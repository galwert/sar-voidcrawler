#!/usr/bin/env python3
"""
Human-part segmentation on a 360° LiDAR scan.

At an interest point the robot does a dense 360° scan of the room. The scene contains a person
(built from labeled parts: head / torso / arms / legs) plus rubble + floor. We raycast the lidar;
because each body part is a separate geometry, Open3D's cast_rays returns which part each point
hit -> every lidar point is segmented by body part (perfect synthetic ground truth).

Outputs (out/segment/): seg_3d.png (full scan), seg_person.png (zoom), and points+labels .npz
for the later interactive click-to-inspect viewer.

Run:  ./.venv/bin/python scripts/segment.py
"""
import os
import numpy as np
import open3d as o3d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(__file__), "..", "out", "segment")
COLORS = {"head": "#ff4545", "torso": "#ffa31a", "arm": "#37e06a", "leg": "#3aa0ff",
          "rubble": "#6e6e6e", "floor": "#333333"}
RNG = np.random.default_rng(1)


def bone(p0, p1, r):
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    v = p1 - p0; L = float(np.linalg.norm(v))
    m = o3d.geometry.TriangleMesh.create_cylinder(radius=r, height=L, resolution=14)
    vn = v / L; z = np.array([0, 0, 1.0]); ax = np.cross(z, vn); s = np.linalg.norm(ax); c = float(np.dot(z, vn))
    if s > 1e-6:
        m.rotate(o3d.geometry.get_rotation_matrix_from_axis_angle(ax / s * np.arctan2(s, c)), center=(0, 0, 0))
    elif c < 0:
        m.rotate(o3d.geometry.get_rotation_matrix_from_axis_angle([np.pi, 0, 0]), center=(0, 0, 0))
    m.translate((p0 + p1) / 2)
    return m


def make_scene():
    """Returns list of (mesh, label). A supine person + rubble + floor."""
    items = []
    items.append((box_mesh(6, 6, 0.1, (0.7, 0, -0.05)), "floor"))

    # --- person (lying on back, along +x), labeled parts ---
    items.append((o3d.geometry.TriangleMesh.create_sphere(0.12).translate((0.0, 0.0, 0.16)), "head"))
    items.append((box_mesh(0.52, 0.36, 0.22, (0.42, 0.0, 0.15)), "torso"))
    for sy in (1, -1):                                      # arms
        items.append((bone((0.2, 0.18 * sy, 0.15), (0.45, 0.30 * sy, 0.12), 0.05), "arm"))
        items.append((bone((0.45, 0.30 * sy, 0.12), (0.68, 0.34 * sy, 0.10), 0.045), "arm"))
    for sy in (1, -1):                                      # legs
        items.append((bone((0.62, 0.10 * sy, 0.15), (1.02, 0.13 * sy, 0.12), 0.075), "leg"))
        items.append((bone((1.02, 0.13 * sy, 0.12), (1.42, 0.14 * sy, 0.08), 0.06), "leg"))

    for _ in range(22):                                    # rubble around the room
        s = RNG.uniform(0.18, 0.6, 3)
        x, y = RNG.uniform(-2.2, 3.0), RNG.uniform(-2.5, 2.5)
        if abs(x - 0.7) < 1.0 and abs(y) < 0.8:
            continue                                       # keep clear around the person
        R = o3d.geometry.get_rotation_matrix_from_xyz(RNG.uniform(0, np.pi, 3))
        m = o3d.geometry.TriangleMesh.create_box(*s); m.translate(-s / 2); m.rotate(R, center=(0, 0, 0))
        m.translate((x, y, RNG.uniform(s[2] / 2, 0.6)))
        items.append((m, "rubble"))
    return items


def box_mesh(sx, sy, sz, c):
    m = o3d.geometry.TriangleMesh.create_box(sx, sy, sz); m.translate((-sx / 2, -sy / 2, -sz / 2)); m.translate(c)
    return m


def scan_360(items, eye, channels=40, h=720, vmin=-52, vmax=14, rmax=8.0, noise=0.008):
    scene = o3d.t.geometry.RaycastingScene()
    labels = []
    for m, lab in items:
        scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(m)); labels.append(lab)
    az = np.linspace(-np.pi, np.pi, h, endpoint=False)
    el = np.deg2rad(np.linspace(vmin, vmax, channels))
    A, E = np.meshgrid(az, el); A, E = A.ravel(), E.ravel()
    dirs = np.stack([np.cos(E) * np.cos(A), np.cos(E) * np.sin(A), np.sin(E)], 1).astype(np.float32)
    rays = np.hstack([np.broadcast_to(eye, dirs.shape), dirs]).astype(np.float32)
    ans = scene.cast_rays(o3d.core.Tensor(rays))
    t = ans["t_hit"].numpy(); gid = ans["geometry_ids"].numpy()
    ok = np.isfinite(t) & (t < rmax)
    pts = (np.asarray(eye) + dirs[ok] * t[ok, None]) + RNG.normal(0, noise, (ok.sum(), 3))
    labs = np.array([labels[g] for g in gid[ok]])
    return pts, labs


def panel(ax, pts, labs, elev, azim, title, zoom=False):
    ax.set_facecolor("black"); ax.set_axis_off()
    if zoom:
        order = ["leg", "arm", "torso", "head"]              # human points only, clean figure
        for lab in order:
            m = labs == lab
            if m.any():
                ax.scatter(pts[m, 0], pts[m, 1], pts[m, 2], s=16, c=COLORS[lab],
                           depthshade=False, label=lab)
        hm = np.isin(labs, order)
        c = pts[hm].mean(0)
        ax.set_xlim(c[0] - 0.95, c[0] + 0.95); ax.set_ylim(c[1] - 0.75, c[1] + 0.75)
        ax.set_zlim(0, 0.55); ax.set_box_aspect((1.9, 1.5, 0.55))
    else:
        for lab in ["floor", "rubble", "leg", "arm", "torso", "head"]:
            m = labs == lab
            if not m.any():
                continue
            s = 1.2 if lab in ("floor", "rubble") else 6
            a = 0.5 if lab in ("floor", "rubble") else 1.0
            ax.scatter(pts[m, 0], pts[m, 1], pts[m, 2], s=s, c=COLORS[lab], alpha=a,
                       depthshade=False, label=lab if lab != "floor" else None)
        ax.set_xlim(-2.5, 3); ax.set_ylim(-2.7, 2.7); ax.set_zlim(0, 1.2)
        ax.set_box_aspect((5.5, 5.4, 1.6))
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(title, color="white", fontsize=12)


def main():
    os.makedirs(OUT, exist_ok=True)
    items = make_scene()
    eye = np.array([0.4, -1.7, 0.55], np.float32)          # robot scan position
    pts, labs = scan_360(items, eye)
    for lab in ["head", "torso", "arm", "leg", "rubble", "floor"]:
        print(f"   {lab:7s}: {(labs == lab).sum()} pts")
    np.savez(os.path.join(OUT, "scan.npz"), points=pts, labels=labs, eye=eye)

    # full scan
    fig = plt.figure(figsize=(11, 8)); fig.patch.set_facecolor("black")
    ax = fig.add_subplot(111, projection="3d")
    panel(ax, pts, labs, 32, -60, "360° LiDAR scan — human-part segmentation")
    leg = ax.legend(loc="upper right", fontsize=10, facecolor="0.12", labelcolor="white", markerscale=2)
    fig.savefig(os.path.join(OUT, "seg_3d.png"), dpi=150, facecolor="black"); plt.close(fig)

    # zoom on the person
    fig = plt.figure(figsize=(10, 8)); fig.patch.set_facecolor("black")
    ax = fig.add_subplot(111, projection="3d")
    panel(ax, pts, labs, 68, -88, "Segmented body parts (lidar points)", zoom=True)
    ax.legend(loc="upper right", fontsize=11, facecolor="0.12", labelcolor="white", markerscale=2)
    fig.savefig(os.path.join(OUT, "seg_person.png"), dpi=150, facecolor="black"); plt.close(fig)
    print("wrote out/segment/seg_3d.png, seg_person.png, scan.npz")


if __name__ == "__main__":
    main()
