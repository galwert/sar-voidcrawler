#!/usr/bin/env python3
"""
Core SLAM + environment for the SAR rubble-exploration demo (pure Python, no ROS / no sudo).

A small robot autonomously explores a maze of rubble void-compartments -- no target, it just
visits every reachable space -- while a 3D map is stitched from synthetic lidar scans.

Pipeline:
  1. ENVIRONMENT  make_compartments(): a pure-rubble maze of small void pockets + an
                 exploration path (DFS) that visits every compartment.
  2. SENSE        raycast a synthetic 3D lidar at each pose (Open3D RaycastingScene).
  3. DEAD-RECKON  noisy wheel odometry that drifts (the naive baseline).
  4. STITCH       odometry-primed point-to-plane scan-to-map ICP -> consistent global map.

This module is imported by explore.py (the 3D wow animation). Run directly for a quick
2D map + metrics:
  ./.venv/bin/python scripts/smoke_demo.py            # writes out/occupancy.png + reconstruction.ply
  ./.venv/bin/python scripts/smoke_demo.py --view     # also opens the interactive 3D viewer
"""
import argparse
import os
import numpy as np
import open3d as o3d

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(__file__), "..", "out")
RNG = np.random.default_rng(0)
MAX_RANGE = 12.0


# --------------------------------------------------------------------- 1. ENVIRONMENT + PATH
def build_box(ext, center, R=None):
    """Open3D box of full size `ext`, rotated by 3x3 `R`, centered at `center`."""
    m = o3d.geometry.TriangleMesh.create_box(*ext)
    m.translate((-ext[0] / 2, -ext[1] / 2, -ext[2] / 2))
    if R is not None:
        m.rotate(R, center=(0, 0, 0))
    m.translate(center)
    return m


def boxes_to_mesh(specs):
    """Efficiently merge many (ext, center, R) boxes into ONE TriangleMesh (no O(n^2) +=)."""
    V, F, off = [], [], 0
    for ext, c, R in specs:
        b = o3d.geometry.TriangleMesh.create_box(*ext)
        b.translate((-ext[0] / 2, -ext[1] / 2, -ext[2] / 2))
        if R is not None:
            b.rotate(R, center=(0, 0, 0))
        b.translate(c)
        V.append(np.asarray(b.vertices))
        F.append(np.asarray(b.triangles) + off)
        off += len(V[-1])
    m = o3d.geometry.TriangleMesh()
    m.vertices = o3d.utility.Vector3dVector(np.vstack(V))
    m.triangles = o3d.utility.Vector3iVector(np.vstack(F))
    m.compute_vertex_normals()
    return m


def _path_from_waypoints(wp, n, smooth=8, z=0.3):
    """Arc-length-resample + smooth a waypoint polyline -> (centerline_xy, poses)."""
    wp = np.asarray(wp, dtype=float)
    seg = np.diff(wp, axis=0)
    seglen = np.linalg.norm(seg, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seglen)])
    s = np.linspace(0, cum[-1], n)
    cl = np.zeros((n, 2))
    for i, si in enumerate(s):
        k = min(int(np.searchsorted(cum, si, side="right") - 1), len(seg) - 1)
        cl[i] = wp[k] + (si - cum[k]) / seglen[k] * seg[k]
    for _ in range(smooth):                                # round corners -> smooth heading
        cl[1:-1] = 0.5 * cl[1:-1] + 0.25 * cl[:-2] + 0.25 * cl[2:]
    poses = []
    for i in range(n):
        a = cl[min(i + 1, n - 1)] - cl[max(i - 1, 0)]
        T = np.eye(4)
        T[:3, :3] = o3d.geometry.get_rotation_matrix_from_xyz((0, 0, np.arctan2(a[1], a[0])))
        T[:3, 3] = (cl[i, 0], cl[i, 1], z)
        poses.append(T)
    return cl, poses


def make_compartments(nx=7, ny=5, cs=1.4, H=1.8, door=0.7, extra=0.12, seed=5):
    """Pure-rubble maze of small void compartments connected by tight gaps, plus an
    exploration path that visits EVERY compartment (DFS traversal, with backtracking).

    Organic, not grid-regular: compartment sizes are jittered and a few extra openings are
    carved beyond the spanning tree. Rubble walls block line-of-sight, so the map reveals
    compartment-by-compartment as the robot crawls. Returns (mesh, poses)."""
    rng = np.random.default_rng(seed)
    xs = np.concatenate([[0.0], np.cumsum(cs * rng.uniform(0.8, 1.3, nx))]); xs -= xs.mean()
    ys = np.concatenate([[0.0], np.cumsum(cs * rng.uniform(0.8, 1.3, ny))]); ys -= ys.mean()
    cx = lambda i: (xs[i] + xs[i + 1]) / 2
    cy = lambda j: (ys[j] + ys[j + 1]) / 2

    # --- random maze (DFS spanning tree); used for the exploration path ---
    def grid_nbrs(c):
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            a, b = c[0] + di, c[1] + dj
            if 0 <= a < nx and 0 <= b < ny:
                yield (a, b)
    passages, visited, stack = set(), {(0, 0)}, [(0, 0)]
    while stack:
        un = [x for x in grid_nbrs(stack[-1]) if x not in visited]
        if un:
            nxt = un[rng.integers(len(un))]
            passages.add(frozenset((stack[-1], nxt)))
            visited.add(nxt); stack.append(nxt)
        else:
            stack.pop()

    # --- extra openings -> organic, interconnected voids (geometry only, path keeps the tree) ---
    openings = set(passages)
    for i in range(1, nx):
        for j in range(ny):
            pr = frozenset(((i - 1, j), (i, j)))
            if pr not in openings and rng.random() < extra:
                openings.add(pr)
    for j in range(1, ny):
        for i in range(nx):
            pr = frozenset(((i, j - 1), (i, j)))
            if pr not in openings and rng.random() < extra:
                openings.add(pr)

    # --- rubble walls (tumbled debris) on every boundary, with a gap where an opening exists ---
    specs = [([xs[-1] - xs[0], ys[-1] - ys[0], 0.1], [0, 0, -0.05], None)]   # floor

    def rubble_wall(axis, c, lo, hi, gap=None):
        spans = [(lo, hi)] if gap is None else [(lo, gap - door / 2), (gap + door / 2, hi)]
        for a, b in spans:
            L, segs = b - a, max(1, int((b - a) / 0.35))
            for s in range(segs):
                t = a + (s + 0.5) * L / segs
                ext = [rng.uniform(0.3, 0.55), rng.uniform(0.3, 0.55), rng.uniform(0.5, H)]
                R = o3d.geometry.get_rotation_matrix_from_xyz(
                    (rng.uniform(-0.3, 0.3), rng.uniform(-0.3, 0.3), rng.uniform(0, np.pi)))
                ctr = ([c + rng.normal(0, 0.06), t, ext[2] / 2] if axis == "v"
                       else [t, c + rng.normal(0, 0.06), ext[2] / 2])
                specs.append((ext, ctr, R))

    for i in range(1, nx):
        for j in range(ny):
            op = frozenset(((i - 1, j), (i, j))) in openings
            rubble_wall("v", xs[i], ys[j], ys[j + 1], cy(j) if op else None)
    for j in range(1, ny):
        for i in range(nx):
            op = frozenset(((i, j - 1), (i, j))) in openings
            rubble_wall("h", ys[j], xs[i], xs[i + 1], cx(i) if op else None)
    for j in range(ny):                                    # perimeter (solid rubble)
        rubble_wall("v", xs[0], ys[j], ys[j + 1]); rubble_wall("v", xs[-1], ys[j], ys[j + 1])
    for i in range(nx):
        rubble_wall("h", ys[0], xs[i], xs[i + 1]); rubble_wall("h", ys[-1], xs[i], xs[i + 1])
    mesh = boxes_to_mesh(specs)

    # --- exploration path: DFS traversal visiting every compartment, backtracking included ---
    adj = {}
    for p in passages:
        a, b = tuple(p)
        adj.setdefault(a, []).append(b); adj.setdefault(b, []).append(a)
    order = []

    def dfs(c, parent):
        order.append(c)
        for nb in adj.get(c, []):
            if nb != parent:
                dfs(nb, c); order.append(c)                # re-enter parent on the way back
    dfs((0, 0), None)

    def door_mid(A, B):
        if A[0] != B[0]:
            return (xs[max(A[0], B[0])], cy(A[1]))
        return (cx(A[0]), ys[max(A[1], B[1])])
    wp = [(cx(order[0][0]), cy(order[0][1]))]
    for A, B in zip(order, order[1:]):
        wp += [door_mid(A, B), (cx(B[0]), cy(B[1]))]
    _, poses = _path_from_waypoints(wp, min(420, max(200, len(wp))), smooth=3)
    return mesh, poses


def _resample_smooth3d(wp, n, smooth=8):
    """Arc-length-resample a 3D waypoint polyline to n points and smooth it."""
    wp = np.asarray(wp, float)
    seg = np.diff(wp, axis=0)
    seglen = np.linalg.norm(seg, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seglen)])
    s = np.linspace(0, cum[-1], n)
    cl = np.zeros((n, 3))
    for i, si in enumerate(s):
        k = min(int(np.searchsorted(cum, si, side="right") - 1), len(seg) - 1)
        cl[i] = wp[k] + (si - cum[k]) / seglen[k] * seg[k]
    for _ in range(smooth):
        cl[1:-1] = 0.5 * cl[1:-1] + 0.25 * cl[:-2] + 0.25 * cl[2:]
    return cl


def _poses_from_path3d(cl):
    """6-DOF poses along a 3D centerline: x-axis = path tangent (so the robot pitches on
    slopes), z-axis kept near world-up (minimal roll)."""
    n = len(cl)
    up0 = np.array([0.0, 0.0, 1.0])
    poses = []
    for i in range(n):
        t = cl[min(i + 1, n - 1)] - cl[max(i - 1, 0)]
        nt = np.linalg.norm(t)
        t = t / nt if nt > 1e-9 else np.array([1.0, 0.0, 0.0])
        side = np.cross(up0, t)
        ns = np.linalg.norm(side)
        side = side / ns if ns > 1e-6 else np.array([0.0, 1.0, 0.0])
        up = np.cross(t, side); up /= np.linalg.norm(up)
        T = np.eye(4)
        T[:3, :3] = np.column_stack([t, side, up])         # forward, left, up
        T[:3, 3] = cl[i]
        poses.append(T)
    return poses


def make_tunnels_3d(nx=6, ny=5, cell=1.5, clearance=0.62, amp=2.4, seed=7):
    """A mostly-2D maze on GENTLY UNDULATING terrain: the layout reads as a 2D tunnel maze, but
    z is a smooth height-field of (x,y), so the robot rises and falls only on shallow ramps
    (~20 deg, no vertical shafts). Rubble is packed in a shell hugging the tube. 6-DOF but easy
    to read. Returns (mesh, poses)."""
    rng = np.random.default_rng(seed)

    # smooth, low-frequency terrain height field -> gentle slopes
    fr = rng.uniform(0.18, 0.38, (4, 2))
    ph = rng.uniform(0, 2 * np.pi, 4)
    aw = rng.uniform(0.5, 1.0, 4)

    def terrain(x, y):
        z = sum(aw[k] * np.sin(fr[k, 0] * x + fr[k, 1] * y + ph[k]) for k in range(4))
        return z * (amp / aw.sum())

    xs = (np.arange(nx) + 0.5) * cell; xs = xs - xs.mean()
    ys = (np.arange(ny) + 0.5) * cell; ys = ys - ys.mean()
    P = {(i, j): np.array([xs[i] + rng.normal(0, 0.12), ys[j] + rng.normal(0, 0.12)])
         for i in range(nx) for j in range(ny)}

    def nbrs(c):
        for d in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            m = (c[0] + d[0], c[1] + d[1])
            if m in P:
                yield m
    passages, visited, stack = set(), {(0, 0)}, [(0, 0)]
    while stack:
        un = [x for x in nbrs(stack[-1]) if x not in visited]
        if un:
            m = un[rng.integers(len(un))]
            passages.add(frozenset((stack[-1], m))); visited.add(m); stack.append(m)
        else:
            stack.pop()

    adj = {}
    for p in passages:
        a, b = tuple(p)
        adj.setdefault(a, []).append(b); adj.setdefault(b, []).append(a)
    order = []

    def dfs(c, par):
        order.append(c)
        for nb in adj.get(c, []):
            if nb != par:
                dfs(nb, c); order.append(c)
    dfs((0, 0), None)

    wp = np.array([P[c] for c in order])
    cl = _resample_smooth3d(np.column_stack([wp, np.zeros(len(wp))]),
                            min(440, max(260, len(order) * 4)), smooth=8)
    cl[:, 2] = terrain(cl[:, 0], cl[:, 1]) + 0.3            # ride the terrain (smooth -> gentle pitch)

    poses = _poses_from_path3d(cl)

    # rubble packed into a shell hugging the tube -> tunnel walls in every direction
    lo = np.array([xs[0] - 1.2, ys[0] - 1.2, cl[:, 2].min() - 1.2])
    hi = np.array([xs[-1] + 1.2, ys[-1] + 1.2, cl[:, 2].max() + 1.2])
    specs, placed, tries = [], 0, 0
    while placed < 1300 and tries < 40000:
        tries += 1
        q = rng.uniform(lo, hi)
        d = float(np.min(np.linalg.norm(cl - q, axis=1)))
        if d < clearance or d > clearance + 0.6:
            continue                                       # keep the tube void; thin shell = clean walls
        ext = list(rng.uniform(0.2, 0.7, 3))
        R = o3d.geometry.get_rotation_matrix_from_xyz(rng.uniform(0, np.pi, 3))
        specs.append((ext, list(q), R)); placed += 1
    return boxes_to_mesh(specs), poses


def _terrain_mesh(xlo, xhi, ylo, yhi, terrain, res=0.25):
    """Triangulated height-field floor over the footprint."""
    xg, yg = np.arange(xlo, xhi, res), np.arange(ylo, yhi, res)
    X, Y = np.meshgrid(xg, yg)
    verts = np.column_stack([X.ravel(), Y.ravel(), terrain(X, Y).ravel()])
    w = len(xg)
    tris = []
    for j in range(len(yg) - 1):
        for i in range(w - 1):
            a = j * w + i
            tris += [[a, a + 1, a + w + 1], [a, a + w + 1, a + w]]
    m = o3d.geometry.TriangleMesh()
    m.vertices = o3d.utility.Vector3dVector(verts)
    m.triangles = o3d.utility.Vector3iVector(np.array(tris))
    return m


def make_compartments_3d(nx=7, ny=5, cs=1.4, H=1.6, door=0.7, extra=0.12, amp=2.0, seed=5):
    """The readable v2 compartment maze, laid on GENTLE undulating terrain: clear thin rubble
    walls + a 2D maze layout (easy to read), but the floor and the robot rise/fall on shallow
    ramps -> genuinely 6-DOF. Returns (mesh, poses)."""
    rng = np.random.default_rng(seed)
    fr = rng.uniform(0.18, 0.38, (4, 2)); ph = rng.uniform(0, 2 * np.pi, 4); aw = rng.uniform(0.5, 1.0, 4)

    def terrain(x, y):
        z = sum(aw[k] * np.sin(fr[k, 0] * x + fr[k, 1] * y + ph[k]) for k in range(4))
        return z * (amp / aw.sum())

    xs = np.concatenate([[0.0], np.cumsum(cs * rng.uniform(0.8, 1.3, nx))]); xs -= xs.mean()
    ys = np.concatenate([[0.0], np.cumsum(cs * rng.uniform(0.8, 1.3, ny))]); ys -= ys.mean()
    cx = lambda i: (xs[i] + xs[i + 1]) / 2
    cy = lambda j: (ys[j] + ys[j + 1]) / 2

    def grid_nbrs(c):
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            a, b = c[0] + di, c[1] + dj
            if 0 <= a < nx and 0 <= b < ny:
                yield (a, b)
    passages, visited, stack = set(), {(0, 0)}, [(0, 0)]
    while stack:
        un = [x for x in grid_nbrs(stack[-1]) if x not in visited]
        if un:
            m = un[rng.integers(len(un))]
            passages.add(frozenset((stack[-1], m))); visited.add(m); stack.append(m)
        else:
            stack.pop()
    openings = set(passages)
    for i in range(1, nx):
        for j in range(ny):
            if rng.random() < extra:
                openings.add(frozenset(((i - 1, j), (i, j))))
    for j in range(1, ny):
        for i in range(nx):
            if rng.random() < extra:
                openings.add(frozenset(((i, j - 1), (i, j))))

    specs = []                                             # walls only (floor is the terrain mesh)

    def rubble_wall(axis, c, lo, hi, gap=None):
        spans = [(lo, hi)] if gap is None else [(lo, gap - door / 2), (gap + door / 2, hi)]
        for a, b in spans:
            L, segs = b - a, max(1, int((b - a) / 0.35))
            for s in range(segs):
                t = a + (s + 0.5) * L / segs
                ext = [rng.uniform(0.3, 0.55), rng.uniform(0.3, 0.55), rng.uniform(0.5, H)]
                R = o3d.geometry.get_rotation_matrix_from_xyz(
                    (rng.uniform(-0.3, 0.3), rng.uniform(-0.3, 0.3), rng.uniform(0, np.pi)))
                wx, wy = (c, t) if axis == "v" else (t, c)
                specs.append((ext, [wx, wy, terrain(wx, wy) + ext[2] / 2], R))   # sit on terrain

    for i in range(1, nx):
        for j in range(ny):
            op = frozenset(((i - 1, j), (i, j))) in openings
            rubble_wall("v", xs[i], ys[j], ys[j + 1], cy(j) if op else None)
    for j in range(1, ny):
        for i in range(nx):
            op = frozenset(((i, j - 1), (i, j))) in openings
            rubble_wall("h", ys[j], xs[i], xs[i + 1], cx(i) if op else None)
    for j in range(ny):
        rubble_wall("v", xs[0], ys[j], ys[j + 1]); rubble_wall("v", xs[-1], ys[j], ys[j + 1])
    for i in range(nx):
        rubble_wall("h", ys[0], xs[i], xs[i + 1]); rubble_wall("h", ys[-1], xs[i], xs[i + 1])

    adj = {}
    for p in passages:
        a, b = tuple(p)
        adj.setdefault(a, []).append(b); adj.setdefault(b, []).append(a)
    order = []

    def dfs(c, par):
        order.append(c)
        for nb in adj.get(c, []):
            if nb != par:
                dfs(nb, c); order.append(c)
    dfs((0, 0), None)

    def door_mid(A, B):
        return (xs[max(A[0], B[0])], cy(A[1])) if A[0] != B[0] else (cx(A[0]), ys[max(A[1], B[1])])
    wp = [(cx(order[0][0]), cy(order[0][1]))]
    for A, B in zip(order, order[1:]):
        wp += [door_mid(A, B), (cx(B[0]), cy(B[1]))]
    cl, _ = _path_from_waypoints(wp, min(420, max(200, len(wp))), smooth=3)
    cl3 = np.column_stack([cl, terrain(cl[:, 0], cl[:, 1]) + 0.3])
    poses = _poses_from_path3d(cl3)

    mesh = boxes_to_mesh(specs) + _terrain_mesh(xs[0] - 0.2, xs[-1] + 0.3, ys[0] - 0.2, ys[-1] + 0.3, terrain)
    mesh.compute_vertex_normals()
    return mesh, poses, terrain                            # terrain() exposed for planning


def _pt_seg_dist(p, A, B):
    """Distance from point p to each segment A[i]->B[i]. Returns (E,)."""
    AB = B - A
    t = np.clip(((p - A) * AB).sum(1) / ((AB * AB).sum(1) + 1e-9), 0, 1)
    return np.linalg.norm(A + t[:, None] * AB - p, axis=1)


def _mst(nodes):
    """Euclidean minimum spanning tree (Prim) -> list of (i, j) edges."""
    n = len(nodes)
    D = np.linalg.norm(nodes[:, None] - nodes[None], axis=2)
    in_tree, rest, edges = [0], set(range(1, n)), []
    while rest:
        best = min((D[u, v], u, v) for u in in_tree for v in rest)
        edges.append((best[1], best[2])); in_tree.append(best[2]); rest.discard(best[2])
    return edges


def make_maze_3d(gx=6, gy=4, cell=2.3, corridor=0.62, amp=1.8, loops=4, seed=5):
    """Larger ORGANIC maze: jittered junction nodes connected by a spanning tree (+a few loops)
    -> winding corridors at varied, non-orthogonal angles, carved through a rubble mass on gentle
    terrain. Returns (mesh, terrain, nodes_xy, edges)."""
    rng = np.random.default_rng(seed)
    fr = rng.uniform(0.18, 0.38, (4, 2)); ph = rng.uniform(0, 2 * np.pi, 4); aw = rng.uniform(0.5, 1.0, 4)

    def terrain(x, y):
        z = sum(aw[k] * np.sin(fr[k, 0] * x + fr[k, 1] * y + ph[k]) for k in range(4))
        return z * (amp / aw.sum())

    nodes = np.array([[(i + 0.5) * cell + rng.uniform(-0.55, 0.55) * cell,
                       (j + 0.5) * cell + rng.uniform(-0.55, 0.55) * cell]
                      for i in range(gx) for j in range(gy)], float)
    nodes -= nodes.mean(0)
    edges = _mst(nodes)
    D = np.linalg.norm(nodes[:, None] - nodes[None], axis=2)         # add a few short loop edges
    eset = {frozenset(e) for e in edges}
    extra = sorted((D[i, j], i, j) for i in range(len(nodes)) for j in range(i + 1, len(nodes))
                   if frozenset((i, j)) not in eset)
    for _, i, j in extra[:loops]:
        edges.append((i, j))
    A, B = nodes[[e[0] for e in edges]], nodes[[e[1] for e in edges]]

    lo, hi = nodes.min(0) - 1.0, nodes.max(0) + 1.0                  # rubble fill except corridors
    specs = []
    for x in np.arange(lo[0], hi[0], 0.34):
        for y in np.arange(lo[1], hi[1], 0.34):
            q = np.array([x, y]) + rng.normal(0, 0.05, 2)
            if _pt_seg_dist(q, A, B).min() < corridor:
                continue                                            # keep the corridor void
            ext = [rng.uniform(0.3, 0.6), rng.uniform(0.3, 0.6), rng.uniform(0.5, 1.6)]
            R = o3d.geometry.get_rotation_matrix_from_xyz(
                (rng.uniform(-0.3, 0.3), rng.uniform(-0.3, 0.3), rng.uniform(0, np.pi)))
            specs.append((ext, [q[0], q[1], terrain(q[0], q[1]) + ext[2] / 2], R))
    mesh = boxes_to_mesh(specs) + _terrain_mesh(lo[0], hi[0], lo[1], hi[1], terrain)
    mesh.compute_vertex_normals()
    return mesh, terrain, nodes, edges


# --------------------------------------------------------------------------------- 2. SENSE
def lidar_rays(channels=16, h_samples=360, v_fov_deg=40.0):
    """Unit ray directions in the sensor frame (x forward). Shape (channels*h_samples, 3)."""
    az = np.linspace(-np.pi, np.pi, h_samples, endpoint=False)
    el = np.deg2rad(np.linspace(-v_fov_deg / 2, v_fov_deg / 2, channels))
    az, el = np.meshgrid(az, el)
    az, el = az.ravel(), el.ravel()
    return np.stack([np.cos(el) * np.cos(az),
                     np.cos(el) * np.sin(az),
                     np.sin(el)], axis=1).astype(np.float32)


def simulate_scan(rcscene, pose, local_dirs, noise=0.01, dropout=0.05):
    """Return scan points in the SENSOR frame (Nx3) as seen from `pose`."""
    R, t = pose[:3, :3], pose[:3, 3]
    world_dirs = local_dirs @ R.T
    origins = np.broadcast_to(t, world_dirs.shape)
    rays = np.hstack([origins, world_dirs]).astype(np.float32)
    hit = rcscene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy()

    ok = np.isfinite(hit) & (hit < MAX_RANGE)
    ok &= RNG.random(ok.shape) > dropout                   # random dropouts
    rng = hit[ok] + RNG.normal(0, noise, ok.sum())         # range noise
    return (local_dirs[ok] * rng[:, None]).astype(np.float64)


# ------------------------------------------------------------------------- 3. DEAD-RECKONING
def dead_reckon(poses, pos_sigma=0.02, yaw_sigma=1.3):
    """Integrate noisy relative motion -> odometry that drifts from truth (yaw deg/step)."""
    odom = [poses[0].copy()]
    for i in range(1, len(poses)):
        rel = np.linalg.inv(poses[i - 1]) @ poses[i]       # true relative motion
        n = np.eye(4)
        n[:3, :3] = o3d.geometry.get_rotation_matrix_from_xyz((0, 0, RNG.normal(0, np.deg2rad(yaw_sigma))))
        n[:3, 3] = RNG.normal(0, pos_sigma, 3)
        odom.append(odom[-1] @ rel @ n)                    # accumulate error
    return odom


# --------------------------------------------------------------------------------- 4. STITCH
def fused_frames(scans, odom, T0, voxel=0.08, max_corr=0.25, min_fitness=0.3):
    """Generator yielding (pose, world_points) per frame as the map is built incrementally.

    Odometry-primed scan-to-map ICP (a real SLAM front-end). Each frame: predict the pose from
    wheel odometry, then refine against the accumulated map with point-to-PLANE ICP (normals
    constrain rotation -> kill yaw drift) + a robust Tukey kernel. If ICP is unconfident we
    fall back to odometry -- this guard prevents degenerate blow-ups in tight rubble.
    """
    p2l = o3d.pipelines.registration.TransformationEstimationPointToPlane(
        o3d.pipelines.registration.TukeyLoss(k=0.1))
    crit = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=40)
    nn = o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 3, max_nn=30)
    gmap, poses = o3d.geometry.PointCloud(), []
    for i, scan in enumerate(scans):
        src = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(scan)).voxel_down_sample(voxel)
        if i == 0:
            T = T0.copy()
        else:
            init = poses[-1] @ (np.linalg.inv(odom[i - 1]) @ odom[i])   # odometry prediction
            reg = o3d.pipelines.registration.registration_icp(src, gmap, max_corr, init, p2l, crit)
            T = reg.transformation if reg.fitness > min_fitness else init  # guard
        poses.append(T)
        src.transform(T)
        world = np.asarray(src.points).copy()              # this scan, in world frame
        gmap += src
        gmap = gmap.voxel_down_sample(voxel)
        gmap.estimate_normals(nn)                          # map normals for next point-to-plane
        yield T, world


def stitch_fused(scans, odom, T0, **kw):
    return [T for T, _ in fused_frames(scans, odom, T0, **kw)]


def build_map(scans, world_poses, voxel=0.05):
    gm = o3d.geometry.PointCloud()
    for scan, T in zip(scans, world_poses):
        gm += o3d.geometry.PointCloud(o3d.utility.Vector3dVector(scan)).transform(T)
    return gm.voxel_down_sample(voxel)


# --------------------------------------------------------------------------------- OUTPUT
def traj_xy(poses):
    return np.array([T[:3, 3] for T in poses])


def rmse(a, b):
    return float(np.sqrt(np.mean(np.sum((traj_xy(a) - traj_xy(b)) ** 2, axis=1))))


def occupancy_png(pcd, est, path, res=0.05, z_band=(0.15, 1.4)):
    pts = np.asarray(pcd.points)
    xy = pts[(pts[:, 2] > z_band[0]) & (pts[:, 2] < z_band[1])][:, :2]
    lo, hi = xy.min(0) - 0.5, xy.max(0) + 0.5
    grid, xe, ye = np.histogram2d(xy[:, 0], xy[:, 1],
                                  bins=[np.arange(lo[0], hi[0], res), np.arange(lo[1], hi[1], res)])
    t = traj_xy(est)
    plt.figure(figsize=(8, 6))
    plt.imshow(np.log1p(grid.T), origin="lower",
               extent=[xe[0], xe[-1], ye[0], ye[-1]], cmap="bone_r")
    plt.plot(t[:, 0], t[:, 1], "-", color="tab:red", lw=2, label="exploration path")
    plt.legend(); plt.title("Explored rubble compartments (2D occupancy)")
    plt.xlabel("x [m]"); plt.ylabel("y [m]"); plt.tight_layout()
    plt.savefig(path, dpi=130); plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--view", action="store_true", help="open interactive 3D viewer (needs a display)")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    print("1/4 building rubble compartment maze + exploration path ...")
    mesh, gt = make_compartments()
    rcscene = o3d.t.geometry.RaycastingScene()
    rcscene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))

    print(f"2/4 simulating lidar over {len(gt)} exploration poses ...")
    dirs = lidar_rays()
    scans = [simulate_scan(rcscene, T, dirs) for T in gt]
    odom = dead_reckon(gt)

    print("3/4 stitching (odometry-primed scan-to-map ICP) ...")
    est = stitch_fused(scans, odom, gt[0])
    global_map = build_map(scans, est)
    print(f"      map: {len(global_map.points)} points")
    print(f"      trajectory RMSE -- dead-reckoning: {rmse(odom, gt):.3f} m | fused: {rmse(est, gt):.3f} m")

    os.makedirs(os.path.join(OUT, "models"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "demos"), exist_ok=True)
    o3d.io.write_point_cloud(os.path.join(OUT, "models", "reconstruction.ply"), global_map)
    occupancy_png(global_map, est, os.path.join(OUT, "demos", "occupancy.png"))
    print("4/4 wrote out/models/reconstruction.ply, out/demos/occupancy.png")

    if args.view:
        o3d.visualization.draw_geometries([global_map], window_name="SAR rubble exploration")


if __name__ == "__main__":
    main()
