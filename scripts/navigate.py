#!/usr/bin/env python3
"""
Goal-directed navigation: A* heuristic planning to a target through the *discovered* 3D rubble
maze (v2-style clean compartments on gentle terrain).

The target may lie OFF the currently-known map: the planner treats unknown space optimistically
(traversable, but penalized), heads toward the goal, senses with the lidar, and REPLANS as
rubble is revealed -- routing around obstacles it discovers. Supports a PRIOR map shared by
another scouting robot (merged into the occupancy before planning).

This is the A* layer of [A* route] -> [IRIS convex safe corridors] -> [trajectory opt].

Outputs out/demos/navigate.mp4 (+ .png). Run:
  ./.venv/bin/python scripts/navigate.py
  ./.venv/bin/python scripts/navigate.py --prior     # another robot pre-scouted the goal area
"""
import argparse
import glob
import heapq
from collections import deque
import os
import shutil
import subprocess
import sys
import numpy as np
import open3d as o3d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.colors import ListedColormap

sys.path.insert(0, os.path.dirname(__file__))
import smoke_demo as sd

OUT = os.path.join(os.path.dirname(__file__), "..", "out", "navigation")
UNKNOWN, FREE, OCC = 0, 1, 2
RES = 0.12
CHECK_R = 6        # cells (~0.7 m): "passing near" a hotspot (with line-of-sight) checks it off
REPLAN = 5         # replan every N steps (not every step)
NB = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
      (-1, -1, 1.414), (-1, 1, 1.414), (1, -1, 1.414), (1, 1, 1.414)]
PART_COLORS = {"head": "#ff4545", "torso": "#ffa31a", "arm": "#37e06a", "leg": "#3aa0ff"}


def _cyl(p0, p1, r):
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    v = p1 - p0; L = float(np.linalg.norm(v))
    m = o3d.geometry.TriangleMesh.create_cylinder(radius=r, height=max(L, 1e-3), resolution=12)
    vn = v / (L or 1.0); z = np.array([0, 0, 1.0]); a = np.cross(z, vn); s = np.linalg.norm(a); c = float(np.dot(z, vn))
    if s > 1e-6:
        m.rotate(o3d.geometry.get_rotation_matrix_from_axis_angle(a / s * np.arctan2(s, c)), center=(0, 0, 0))
    elif c < 0:
        m.rotate(o3d.geometry.get_rotation_matrix_from_axis_angle([np.pi, 0, 0]), center=(0, 0, 0))
    m.translate((p0 + p1) / 2); return m


def _sph(c, r):
    m = o3d.geometry.TriangleMesh.create_sphere(r); m.translate(c); return m


def _obox(center, sx, sy, sz, ex, ey):
    m = o3d.geometry.TriangleMesh.create_box(sx, sy, sz); m.translate((-sx / 2, -sy / 2, -sz / 2))
    m.rotate(np.column_stack([ex, ey, np.cross(ex, ey)]), center=(0, 0, 0)); m.translate(center); return m


def bresenham(a, b):
    (x0, y0), (x1, y1) = a, b
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    sx, sy = (1 if x0 < x1 else -1), (1 if y0 < y1 else -1)
    err = dx - dy
    out = []
    while (x0, y0) != (x1, y1):
        out.append((x0, y0))
        e2 = 2 * err
        if e2 > -dy:
            err -= dy; x0 += sx
        if e2 < dx:
            err += dx; y0 += sy
    return out


def astar(grid, start, goal):
    """8-connected A*; unknown cells are traversable but penalized (optimistic for off-map goals)."""
    W, H = grid.shape
    gx, gy = goal
    h = lambda c: np.hypot(c[0] - gx, c[1] - gy)
    openq = [(h(start), 0.0, start)]
    came, gsc = {}, {start: 0.0}
    while openq:
        _, gc, c = heapq.heappop(openq)
        if c == goal:
            p = [c]
            while c in came:
                c = came[c]; p.append(c)
            return p[::-1]
        if gc > gsc.get(c, 1e18):
            continue
        for dx, dy, base in NB:
            nb = (c[0] + dx, c[1] + dy)
            if not (0 <= nb[0] < W and 0 <= nb[1] < H):
                continue
            cc = 1.0 if grid[nb] == FREE else (1.6 if grid[nb] == UNKNOWN else 1000.0)  # OCC = last resort (never stuck)
            ng = gc + base * cc
            if ng < gsc.get(nb, 1e18):
                gsc[nb] = ng; came[nb] = c
                heapq.heappush(openq, (ng + h(nb), ng, nb))
    return None


def line_of_sight(grid, a, b):
    return all(grid[c] != OCC for c in bresenham(a, b))


def shortcut(grid, path):
    """String-pull: keep only the corners where line-of-sight breaks -> straight segments."""
    if len(path) < 3:
        return path
    out, i = [path[0]], 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1 and not line_of_sight(grid, path[i], path[j]):
            j -= 1
        out.append(path[j]); i = j
    return out


def cells_along(sp):
    """Flatten a corner list into the grid cells to walk (start .. end)."""
    if not sp:
        return []
    out = []
    for a, b in zip(sp, sp[1:]):
        out += bresenham(a, b)
    out.append(sp[-1])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prior", action="store_true", help="seed a prior map from another scouting robot")
    ap.add_argument("--fps", type=int, default=12)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    print("building v1-style compartment maze (2D) ...")
    mesh, poses = sd.make_compartments(nx=12, ny=8, door=1.1)  # larger maze -> more turns
    terrain = lambda x, y: 0.0                              # flat floor (2D)
    rc = o3d.t.geometry.RaycastingScene()
    rc.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    dirs = sd.lidar_rays(channels=20, h_samples=400, v_fov_deg=80)

    aabb = mesh.get_axis_aligned_bounding_box()
    lo = np.asarray(aabb.get_min_bound())[:2] - 0.3
    hi = np.asarray(aabb.get_max_bound())[:2] + 0.3
    W = int((hi[0] - lo[0]) / RES) + 1
    Hh = int((hi[1] - lo[1]) / RES) + 1
    to_cell = lambda p: (int((p[0] - lo[0]) / RES), int((p[1] - lo[1]) / RES))
    to_world = lambda c: np.array([lo[0] + (c[0] + 0.5) * RES, lo[1] + (c[1] + 0.5) * RES])
    grid = np.zeros((W, Hh), dtype=np.uint8)
    rng = np.random.default_rng(0)
    chunks = []                                            # accumulated lidar points (for the v1-style map)

    pp = np.array([T[:3, 3] for T in poses])
    start = to_cell(pp[0])
    hotspots = [to_cell(pp[int(len(pp) * f)]) for f in (0.30, 0.55, 0.78, 0.95)]   # KNOWN target points

    def sense(cell, g, collect=True):
        wx, wy = to_world(cell)
        z = terrain(wx, wy) + 0.3
        T = np.eye(4); T[:3, 3] = [wx, wy, z]
        scan = sd.simulate_scan(rc, T, dirs) + np.array([wx, wy, z])   # R=I -> world = scan + t
        if len(scan) > 800:
            scan = scan[rng.choice(len(scan), 800, replace=False)]
        free_cells, hits, wallpts = set(), {}, []
        for p in scan:
            c = to_cell(p)
            if not (0 <= c[0] < W and 0 <= c[1] < Hh):
                continue
            free_cells.update(bresenham(cell, c))           # space along each ray is free
            if 0.22 < p[2] - terrain(p[0], p[1]) < 1.3:     # a wall-height return
                hits[c] = hits.get(c, 0) + 1
                wallpts.append(p)                           # full 3D point (for turbo height-coloring)
        for fc in free_cells:
            if g[fc] == UNKNOWN:
                g[fc] = FREE
        for c, n in hits.items():
            if n >= 2:                                      # >=2 returns -> real wall; stray points can't seal a doorway
                g[c] = OCC
        g[cell] = FREE
        if collect and wallpts:
            chunks.append(np.array(wallpts))                # keep the raw lidar points for the v1-style map

    # TRUE navigable free area (geometric, from the mesh) -> honest coverage denominator.
    # Mark wall cells from the rubble surface in the robot's height band, then flood-fill the
    # reachable open space from the start (bounded by the perimeter walls).
    mpts = np.asarray(mesh.sample_points_uniformly(300000).points)
    band = mpts[(mpts[:, 2] > 0.20) & (mpts[:, 2] < 1.40)]     # rubble at robot height = walls
    wall = np.zeros((W, Hh), dtype=bool)
    ci = ((band[:, 0] - lo[0]) / RES).astype(int)
    cj = ((band[:, 1] - lo[1]) / RES).astype(int)
    ok = (ci >= 0) & (ci < W) & (cj >= 0) & (cj < Hh)
    wall[ci[ok], cj[ok]] = True
    wi, wj = np.where(wall)                                    # non-wall cells inside the building footprint
    open_mask = np.zeros((W, Hh), dtype=bool)
    open_mask[wi.min():wi.max() + 1, wj.min():wj.max() + 1] = ~wall[wi.min():wi.max() + 1, wj.min():wj.max() + 1]
    total_free = max(1, int(open_mask.sum()))

    if args.prior:                                          # another robot already scouted the goal area
        print("merging prior map from another robot ...")
        for q in pp[int(len(pp) * 0.70):int(len(pp) * 0.92):2]:
            sense(to_cell(q), grid)

    print(f"visiting {len(hotspots)} known hotspots from {start} ...")
    robot, frames, visited = start, [], [False] * len(hotspots)
    sense(robot, grid)
    near = lambda a, b: max(abs(a[0] - b[0]), abs(a[1] - b[1])) <= 1

    def nearest_path(r):                                    # nearest UNVISITED hotspot by actual path length
        best_i, best_p = None, None
        for i, h in enumerate(hotspots):
            if visited[i]:
                continue
            p = astar(grid, r, h)
            if p and (best_p is None or len(p) < len(best_p)):
                best_i, best_p = i, p
        return best_i, best_p

    cellpath, ptr, ti, sp, steps = [], 1, None, None, 0
    while sum(visited) < len(hotspots) and steps < 9000:
        # every REPLAN steps (or when blocked / current target reached): re-pick the nearest hotspot
        if steps % REPLAN == 0 or ti is None or visited[ti] or ptr >= len(cellpath) or grid[cellpath[ptr]] == OCC:
            ti, p = nearest_path(robot)
            if ti is None:
                break
            if near(robot, hotspots[ti]):
                visited[ti] = True; ti = None; continue
            sp = shortcut(grid, p)
            cellpath, ptr = cells_along(sp), 1
            if ptr >= len(cellpath):
                visited[ti] = True; ti = None; continue
        cov = 100.0 * int(((grid == FREE) & open_mask).sum()) / total_free
        frames.append((sum(len(c) for c in chunks), robot, sp, list(visited), ti, cov))
        robot = cellpath[ptr]; ptr += 1
        sense(robot, grid); steps += 1
        if near(robot, hotspots[ti]):                       # reached the current target
            visited[ti] = True
        for j, h in enumerate(hotspots):                    # opportunistic: pass near any star (LOS) -> checked off
            if not visited[j] and max(abs(robot[0] - h[0]), abs(robot[1] - h[1])) <= CHECK_R \
               and line_of_sight(grid, robot, h):
                visited[j] = True
    obs_free = int(((grid == FREE) & open_mask).sum())
    print(f"   reached {sum(visited)}/{len(hotspots)} hotspots in {steps} steps | "
          f"mapped {frames[-1][5]:.0f}% of the building")
    print(f"   [coverage check] observed_free={obs_free}  total_free={total_free}  "
          f"ratio={100.0 * obs_free / total_free:.1f}%")

    # --- animation (top-down occupancy + plan + robot) ---
    sx, sy = to_world(start)
    hw = np.array([to_world(h) for h in hotspots])
    # --- render identically to the v1 explore demo: 3D turbo-by-height cloud + cyan path + inset ---
    cloud = np.vstack(chunks) if chunks else np.zeros((1, 3))
    rw3 = np.array([[*to_world(fr[1]), 0.35] for fr in frames])      # robot world pos per frame
    hs3 = np.array([[*to_world(h), 0.45] for h in hotspots])
    full_xy = cloud[:, :2]
    lo3, hi3 = cloud.min(0) - 0.3, cloud.max(0) + 0.3
    ctr, half = (lo3 + hi3) / 2, (hi3 - lo3) / 2
    zmax = max(2.0, hi3[2])
    # ---- arriving at the last hotspot, do a dense 360° scan and segment a buried survivor ----
    #  Only parts not hidden under the collapsed slab return points -> partial visibility, like the
    #  Gemini photos (a hand / forearm reaching out; head and body occluded).
    eye = np.array([rw3[-1, 0], rw3[-1, 1], 0.35])
    hd0 = rw3[-1, :2] - rw3[max(0, len(rw3) - 4), :2]
    hd0 = hd0 / (np.linalg.norm(hd0) or 1.0)
    rcell = to_cell(rw3[-1, :2])
    best = None                                            # place the survivor in a free, visible spot near the robot
    for a in np.linspace(0, 2 * np.pi, 24, endpoint=False):
        d = np.array([np.cos(a), np.sin(a)])
        cand = rw3[-1, :2] + d * 0.85
        cc = to_cell(cand)
        if 0 <= cc[0] < W and 0 <= cc[1] < Hh and grid[cc] != OCC and line_of_sight(grid, rcell, cc):
            sc = float(d @ hd0)
            if best is None or sc > best[0]:
                best = (sc, d, cand)
    hd = best[1] if best else hd0
    hc = best[2] if best else rw3[-1, :2] + hd * 0.85
    ex = np.array([-hd[1], hd[0], 0.0]); ey = np.array([-hd[0], -hd[1], 0.0])   # body axis / toward-robot

    def P(ap, at, z):                                      # ap = along body, at = toward robot
        return (hc[0] + ap * ex[0] - at * hd[0], hc[1] + ap * ex[1] - at * hd[1], z)
    z0 = 0.12
    surv = [(_obox(P(0, 0, z0), 0.50, 0.30, 0.22, ex, ey), "torso"),
            (_sph(P(0.40, 0.0, z0 + 0.02), 0.12), "head"),
            (_cyl(P(-0.08, 0.06, z0), P(-0.58, 0.08, z0 - 0.02), 0.07), "leg"),
            (_cyl(P(-0.08, -0.06, z0), P(-0.58, -0.08, z0 - 0.02), 0.07), "leg"),
            (_cyl(P(0.10, 0.10, z0), P(0.04, 0.42, z0 - 0.02), 0.05), "arm"),
            (_cyl(P(0.04, 0.42, z0 - 0.02), P(-0.02, 0.74, z0 - 0.03), 0.045), "arm"),
            (_sph(P(-0.02, 0.74, z0 - 0.03), 0.05), "arm"),                     # hand reaching out
            (_obox(P(0.0, -0.05, 0.30), 0.72, 0.70, 0.24, ex, ey), "occluder")]  # collapsed slab resting on the body
    rc2 = o3d.t.geometry.RaycastingScene(); slabels = ["occluder"]
    rc2.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))            # walls occlude too
    for m, lab in surv:
        rc2.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(m)); slabels.append(lab)
    saz = np.linspace(-np.pi, np.pi, 1000, endpoint=False)
    sel = np.deg2rad(np.linspace(-55, 12, 52))
    AA, EE = np.meshgrid(saz, sel); AA, EE = AA.ravel(), EE.ravel()
    sdir = np.stack([np.cos(EE) * np.cos(AA), np.cos(EE) * np.sin(AA), np.sin(EE)], 1).astype(np.float32)
    rays = np.hstack([np.broadcast_to(eye.astype(np.float32), sdir.shape), sdir]).astype(np.float32)
    sans = rc2.cast_rays(o3d.core.Tensor(rays))
    st = sans["t_hit"].numpy(); sgid = sans["geometry_ids"].numpy()
    sok = np.isfinite(st) & (st < 6.0)
    spts = eye + sdir[sok] * st[sok, None] + rng.normal(0, 0.008, (int(sok.sum()), 3))
    slab = np.array([slabels[g] for g in sgid[sok]])
    keep = np.isin(slab, ["head", "torso", "arm", "leg"])
    HP, HL = spts[keep], slab[keep]
    print("   survivor parts visible:", {p: int((HL == p).sum()) for p in ["head", "torso", "arm", "leg"]})
    haz = np.arctan2(HP[:, 1] - eye[1], HP[:, 0] - eye[0]) if len(HP) else np.zeros(0)
    hord = np.argsort(haz); HP, HL, HA = HP[hord], HL[hord], (haz[hord] if len(HP) else haz)

    F_nav = min(140, len(frames))
    F_scan = 74                                            # zoom-in + sweep + a long hold on the segmented body
    F = F_nav + F_scan
    idx = np.linspace(0, len(frames) - 1, F_nav).astype(int)

    fig = plt.figure(figsize=(9.5, 7.2)); fig.patch.set_facecolor("black")
    ax = fig.add_subplot(111, projection="3d")
    inset = fig.add_axes([0.70, 0.04, 0.29, 0.30]); inset.set_facecolor("#0a0a0a")

    def hcol(i, vis, ti):
        return "lime" if vis[i] else ("magenta" if i == ti else "orange")

    def draw(f):
        ax.cla(); ax.set_facecolor("black"); ax.set_axis_off()
        ax.set_position([-0.04, -0.02, 1.08, 1.06])
        inset.cla(); inset.set_facecolor("#0a0a0a"); inset.set_xticks([]); inset.set_yticks([]); inset.set_aspect("equal")
        for sp in inset.spines.values():
            sp.set_color("0.4")
        inset.set_xlim(lo3[0], hi3[0]); inset.set_ylim(lo3[1], hi3[1])

        if f < F_nav:
            prog = f / max(1, F_nav - 1)
            n, _rc, _path, vis, ti, cov = frames[idx[f]]
            c = cloud[:max(1, n)]
            if len(c) > 14000:
                c = c[::max(1, len(c) // 14000)]
            tr = rw3[:idx[f] + 1]
            zoom = 1.3 - 0.3 * min(1.0, prog / 0.4)
            L, Hh = ctr - half * zoom, ctr + half * zoom
            ax.set_xlim(L[0], Hh[0]); ax.set_ylim(L[1], Hh[1]); ax.set_zlim(0, zmax)
            ax.set_box_aspect((Hh[0] - L[0], Hh[1] - L[1], zmax * 2.6))
            ax.scatter(c[:, 0], c[:, 1], c[:, 2], c=c[:, 2], cmap="turbo", s=2.2, depthshade=False, vmin=0, vmax=zmax)
            ax.plot(tr[:, 0], tr[:, 1], tr[:, 2], color="cyan", lw=2.2, alpha=0.95)
            for i in range(len(hotspots)):
                ax.scatter([hs3[i, 0]], [hs3[i, 1]], [hs3[i, 2]], c=hcol(i, vis, ti), marker="*",
                           s=420, edgecolors="black", depthshade=False)
            ax.scatter([tr[-1, 0]], [tr[-1, 1]], [0.5], c="white", edgecolors="red", s=110, depthshade=False)
            ax.view_init(elev=70 - 28 * min(1.0, prog / 0.4), azim=-80 + 130 * prog)

            inset.scatter(c[:, 0], c[:, 1], s=0.6, c=c[:, 2], cmap="turbo", vmin=0, vmax=zmax, linewidths=0)
            inset.plot(tr[:, 0], tr[:, 1], color="cyan", lw=0.9, alpha=0.6)
            if _path and len(_path) > 1:
                pw = np.array([to_world(cc) for cc in _path])
                inset.plot(pw[:, 0], pw[:, 1], color="yellow", lw=1.7)
                a, b = pw[0], pw[min(10, len(pw) - 1)]
                d = b - a; nd = np.linalg.norm(d)
                if nd > 1e-6:
                    d = d / nd * 0.11 * (hi3[0] - lo3[0])
                    inset.annotate("", xy=(a[0] + d[0], a[1] + d[1]), xytext=(a[0], a[1]),
                                   arrowprops=dict(arrowstyle="-|>", color="yellow", lw=2.4))
            for i in range(len(hotspots)):
                inset.scatter([hs3[i, 0]], [hs3[i, 1]], c=hcol(i, vis, ti), marker="*", s=110, edgecolors="black", zorder=5)
            inset.scatter([tr[-1, 0]], [tr[-1, 1]], c="white", edgecolors="red", s=34, zorder=6)
            fig.suptitle(f"Reaching known hotspots — checked {sum(vis)}/{len(hotspots)}  |  mapped {cov:.0f}%"
                         f"   ({'prior-aided' if args.prior else 'no prior'})", color="white", fontsize=12, y=0.97)
        else:
            k = f - F_nav
            near = (np.abs(cloud[:, 0] - hc[0]) < 3.0) & (np.abs(cloud[:, 1] - hc[1]) < 3.0)
            c = cloud[near]                                                   # only nearby walls -> clean scan view
            if len(c) > 8000:
                c = c[::max(1, len(c) // 8000)]
            zt = min(1.0, k / 16.0); zt = zt * zt * (3 - 2 * zt)              # ease the zoom-in
            cx = ctr[0] * (1 - zt) + hc[0] * zt; cy = ctr[1] * (1 - zt) + hc[1] * zt
            hr = (half[0] + half[1]) / 2 * (1 - zt) + 1.15 * zt              # zoom in TIGHT on the body
            ax.set_xlim(cx - hr, cx + hr); ax.set_ylim(cy - hr, cy + hr); ax.set_zlim(0, zmax)
            ax.set_box_aspect((2 * hr, 2 * hr, zmax * 2.6))
            if len(c):
                ax.scatter(c[:, 0], c[:, 1], c[:, 2], c=c[:, 2], cmap="turbo", s=1.6, depthshade=False,
                           vmin=0, vmax=zmax, alpha=0.16)                     # map faded to the background
            ax.scatter([eye[0]], [eye[1]], [0.45], c="white", edgecolors="red", s=150, depthshade=False)
            reveal = min(1.0, k / 46.0)
            nrev = int(reveal * len(HP))
            ang = HA[min(nrev, len(HP) - 1)] if len(HP) else 0.0
            ax.plot([eye[0], eye[0] + 2.6 * np.cos(ang)], [eye[1], eye[1] + 2.6 * np.sin(ang)],
                    [0.12, 0.12], color="cyan", lw=2.2, alpha=0.85)           # rotating scan beam
            for lab in ["leg", "arm", "torso", "head"]:
                m = HL[:nrev] == lab
                if m.any():
                    ax.scatter(HP[:nrev][m, 0], HP[:nrev][m, 1], HP[:nrev][m, 2], s=80,
                               c=PART_COLORS[lab], depthshade=False, alpha=0.16)          # glow
                    ax.scatter(HP[:nrev][m, 0], HP[:nrev][m, 1], HP[:nrev][m, 2], s=28,
                               c=PART_COLORS[lab], depthshade=False, label=lab)
            ax.view_init(elev=58 - 20 * min(1.0, k / F_scan), azim=40 + 55 * (k / F_scan))
            if nrev > 0:
                ax.legend(loc="upper left", fontsize=11, facecolor="0.1", labelcolor="white",
                          markerscale=1.5, framealpha=0.85)

            inset.scatter(cloud[::6, 0], cloud[::6, 1], s=0.5, c="0.4", linewidths=0)     # full-map context
            inset.plot(rw3[:, 0], rw3[:, 1], color="cyan", lw=0.8, alpha=0.5)
            for lab in ["leg", "arm", "torso", "head"]:
                m = HL[:nrev] == lab
                if m.any():
                    inset.scatter(HP[:nrev][m, 0], HP[:nrev][m, 1], s=7, c=PART_COLORS[lab], linewidths=0)
            inset.scatter([eye[0]], [eye[1]], c="white", edgecolors="red", s=30, zorder=6)
            if reveal < 1.0:
                ttl = "● 360° SCAN at hotspot — segmenting survivor…"
            else:
                seen = sorted(set(HL), key=lambda x: ["head", "torso", "arm", "leg"].index(x))
                ttl = "SURVIVOR FOUND — parts: " + ", ".join(seen)
            fig.suptitle(ttl, color=("#ff6060" if reveal >= 1 else "white"), fontsize=13, y=0.97)
        return []

    out = os.path.join(OUT, "navigate_prior.mp4" if args.prior else "navigate.mp4")
    ff = shutil.which("ffmpeg") or next(
        (p for p in glob.glob(os.path.expanduser("~/miniforge3/envs/*/bin/ffmpeg")) if os.path.exists(p)), None)
    print(f"rendering {F} frames ({F_nav} nav + {F_scan} scan) -> {out} ...")
    anim = FuncAnimation(fig, draw, frames=F, blit=False)
    if ff:
        plt.rcParams["animation.ffmpeg_path"] = ff
        anim.save(out, writer=FFMpegWriter(fps=args.fps, extra_args=[
            "-pix_fmt", "yuv420p", "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2"]),
            savefig_kwargs={"facecolor": "black"})
    draw(F - 1); fig.savefig(out.replace(".mp4", ".png"), dpi=150, facecolor="black"); plt.close()

    # --- extra view formats of the same route ---
    suffix = "_prior" if args.prior else ""
    route = rw3[:, :2]
    cov_series = [fr[5] for fr in frames]

    f2 = plt.figure(figsize=(10, 8)); f2.patch.set_facecolor("black")          # 1) top-down white route
    a2 = f2.add_axes([0, 0, 1, 1]); a2.set_facecolor("black"); a2.axis("off"); a2.set_aspect("equal")
    a2.scatter(cloud[:, 0], cloud[:, 1], s=0.5, c="white", linewidths=0)
    a2.plot(route[:, 0], route[:, 1], color="cyan", lw=2.0)
    a2.scatter(hs3[:, 0], hs3[:, 1], c="lime", marker="*", s=320, edgecolors="black", zorder=5)
    a2.scatter([route[0, 0]], [route[0, 1]], c="yellow", s=70, zorder=5)
    f2.savefig(os.path.join(OUT, f"nav_topdown{suffix}.png"), dpi=180, facecolor="black"); plt.close(f2)

    f3 = plt.figure(figsize=(10, 8)); f3.patch.set_facecolor("black")          # 2) 3D turbo hero of the route
    a3 = f3.add_subplot(111, projection="3d"); a3.set_facecolor("black"); a3.set_axis_off()
    a3.set_position([-0.05, -0.05, 1.1, 1.1])
    a3.scatter(cloud[:, 0], cloud[:, 1], cloud[:, 2], c=cloud[:, 2], cmap="turbo", s=2.0, depthshade=False, vmin=0, vmax=zmax)
    a3.plot(route[:, 0], route[:, 1], 0.35 + 0 * route[:, 0], color="cyan", lw=2.0)
    a3.scatter(hs3[:, 0], hs3[:, 1], hs3[:, 2], c="lime", marker="*", s=320, edgecolors="black", depthshade=False)
    a3.set_box_aspect((np.ptp(cloud[:, 0]), np.ptp(cloud[:, 1]), zmax * 2.2)); a3.view_init(elev=46, azim=-60)
    f3.savefig(os.path.join(OUT, f"nav_hero3d{suffix}.png"), dpi=180, facecolor="black"); plt.close(f3)

    f4, a4 = plt.subplots(figsize=(8, 5))                                      # 3) accurate coverage curve
    x = np.linspace(0, 100, len(cov_series))
    a4.plot(x, cov_series, color="tab:blue", lw=2.5); a4.fill_between(x, cov_series, alpha=0.15)
    a4.set_xlabel("mission progress [%]"); a4.set_ylabel("building mapped [%]")
    a4.set_title(f"Coverage while reaching hotspots (final {cov_series[-1]:.0f}%)"); a4.grid(alpha=0.3)
    a4.set_ylim(0, 101); f4.tight_layout()
    f4.savefig(os.path.join(OUT, f"nav_coverage{suffix}.png"), dpi=160); plt.close(f4)

    f5 = plt.figure(figsize=(11, 8.5)); f5.patch.set_facecolor("black")        # 4) 2D map of ALL lidar findings
    a5 = f5.add_axes([0, 0, 1, 1]); a5.set_facecolor("black"); a5.axis("off"); a5.set_aspect("equal")
    a5.scatter(cloud[:, 0], cloud[:, 1], s=0.4, c="white", linewidths=0)
    f5.savefig(os.path.join(OUT, f"nav_map2d{suffix}.png"), dpi=200, facecolor="black"); plt.close(f5)

    print("done:", out, "+ nav_topdown / nav_hero3d / nav_coverage / nav_map2d")


if __name__ == "__main__":
    main()
