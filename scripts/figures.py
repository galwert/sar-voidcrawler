#!/usr/bin/env python3
"""
Generate presentation-ready STILL figures from the v2 compartment-maze SLAM run.
Everything is written to out/presentation/.

  pres_topdown_white.png        - top-down, all lidar points white on black ("scan" look)
  pres_topdown_white_trans.png  - same, transparent background (overlay on a dark slide)
  pres_topdown_white_poster.png - 300 dpi poster version
  pres_blueprint.png            - dark points on white + red path (floor plan)
  pres_blueprint_trans.png      - dark points, transparent bg (overlay on a light slide)
  pres_trajectory.png           - ground truth vs odometry vs SLAM, with RMSE (result slide)
  pres_compare.png              - ground-truth map vs SLAM reconstruction (accuracy)
  pres_hero3d.png / _trans / _poster - height-colored 3D reconstruction (hero image)
  pres_coverage.png             - % of the space mapped vs time
  pres_pipeline.png             - environment | one scan | stitched map

Run:  ./.venv/bin/python scripts/figures.py
"""
import os
import sys
import numpy as np
import open3d as o3d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
import smoke_demo as sd

OUT = os.path.join(os.path.dirname(__file__), "..", "out", "presentation")


def traj(poses):
    return np.array([T[:3, 3] for T in poses])


def main():
    os.makedirs(OUT, exist_ok=True)
    print("running v2 pipeline ...")
    mesh, gt = sd.make_compartments()
    rc = o3d.t.geometry.RaycastingScene()
    rc.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    dirs = sd.lidar_rays()
    scans = [sd.simulate_scan(rc, T, dirs) for T in gt]
    odom = sd.dead_reckon(gt)
    frames = list(sd.fused_frames(scans, odom, gt[0], voxel=0.05))
    est = [T for T, _ in frames]
    worlds = [w for _, w in frames]
    rec = np.vstack(worlds)
    gtm = np.asarray(sd.build_map(scans, gt).points)
    rmse = lambda a, b: float(np.sqrt(np.mean(np.sum((traj(a) - traj(b)) ** 2, axis=1))))
    print(f"   fused RMSE {rmse(est, gt):.3f} m | dead-reckoning {rmse(odom, gt):.3f} m")

    def P(name):
        return os.path.join(OUT, name)

    # 1) top-down white on black (+ transparent + poster) ----------------------------------
    for tag, dpi, transp in [("", 200, False), ("_trans", 200, True), ("_poster", 300, False)]:
        fig = plt.figure(figsize=(12, 9.6) if tag == "_poster" else (10, 8))
        if not transp:
            fig.patch.set_facecolor("black")
        ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off"); ax.set_aspect("equal")
        if not transp:
            ax.set_facecolor("black")
        ax.scatter(rec[:, 0], rec[:, 1], s=0.5, c="white", linewidths=0)
        fig.savefig(P(f"pres_topdown_white{tag}.png"), dpi=dpi, transparent=transp,
                    facecolor=("none" if transp else "black"))
        plt.close(fig)

    # 2) blueprint: dark points on white / transparent + red path --------------------------
    t = traj(est)
    for tag, transp in [("", False), ("_trans", True)]:
        fig = plt.figure(figsize=(10, 8)); ax = fig.add_axes([0.02, 0.02, 0.96, 0.96]); ax.set_aspect("equal")
        ax.scatter(rec[:, 0], rec[:, 1], s=0.5, c="#102040", linewidths=0)
        ax.plot(t[:, 0], t[:, 1], color="crimson", lw=1.6)
        ax.set_xticks([]); ax.set_yticks([])
        if not transp:
            ax.set_title("Discovered floor plan + robot path", fontsize=13)
        fig.savefig(P(f"pres_blueprint{tag}.png"), dpi=200, transparent=transp); plt.close(fig)

    # 3) trajectory result -----------------------------------------------------------------
    g, o, e = traj(gt), traj(odom), traj(est)
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.plot(g[:, 0], g[:, 1], "-", color="k", lw=3, label="ground truth")
    ax.plot(o[:, 0], o[:, 1], "--", color="tab:orange", lw=2, label=f"odometry only ({rmse(odom, gt):.2f} m)")
    ax.plot(e[:, 0], e[:, 1], "-", color="tab:blue", lw=2, label=f"lidar SLAM ({rmse(est, gt):.2f} m)")
    ax.axis("equal"); ax.grid(alpha=0.3); ax.legend(fontsize=11)
    ax.set_title("Trajectory: lidar stitching corrects odometry drift")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); fig.tight_layout()
    fig.savefig(P("pres_trajectory.png"), dpi=180); plt.close(fig)

    # 4) ground-truth map vs SLAM reconstruction -------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(15, 6)); fig.patch.set_facecolor("black")
    for ax, pts, ttl in [(axes[0], gtm, "ground-truth map"), (axes[1], rec, "SLAM reconstruction")]:
        ax.set_facecolor("black"); ax.axis("off"); ax.set_aspect("equal")
        ax.scatter(pts[:, 0], pts[:, 1], s=0.5, c="white", linewidths=0)
        ax.set_title(ttl, color="white", fontsize=14)
    fig.tight_layout(); fig.savefig(P("pres_compare.png"), dpi=170, facecolor="black"); plt.close(fig)

    # 5) hero 3D still (+ transparent + poster) --------------------------------------------
    for tag, dpi, transp in [("", 190, False), ("_trans", 190, True), ("_poster", 300, False)]:
        fig = plt.figure(figsize=(10, 8))
        if not transp:
            fig.patch.set_facecolor("black")
        ax = fig.add_subplot(111, projection="3d"); ax.set_axis_off()
        if not transp:
            ax.set_facecolor("black")
        ax.set_position([-0.05, -0.05, 1.1, 1.1])
        ax.scatter(rec[:, 0], rec[:, 1], rec[:, 2], c=rec[:, 2], cmap="turbo", s=2.0, depthshade=False)
        ax.set_box_aspect((np.ptp(rec[:, 0]), np.ptp(rec[:, 1]), np.ptp(rec[:, 2]) * 2.2))
        ax.view_init(elev=46, azim=-60)
        fig.savefig(P(f"pres_hero3d{tag}.png"), dpi=dpi, transparent=transp,
                    facecolor=("none" if transp else "black"))
        plt.close(fig)

    # 6) coverage over time ----------------------------------------------------------------
    res, seen, cov = 0.1, set(), []
    for w in worlds:
        seen |= set(map(tuple, np.floor(w[:, :2] / res).astype(int)))
        cov.append(len(seen))
    cov = 100.0 * np.array(cov) / cov[-1]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(np.linspace(0, 100, len(cov)), cov, color="tab:blue", lw=2.5)
    ax.fill_between(np.linspace(0, 100, len(cov)), cov, alpha=0.15)
    ax.set_xlabel("exploration progress [%]"); ax.set_ylabel("area mapped [%]")
    ax.set_title("Autonomous coverage: every compartment visited"); ax.grid(alpha=0.3)
    ax.set_xlim(0, 100); ax.set_ylim(0, 101); fig.tight_layout()
    fig.savefig(P("pres_coverage.png"), dpi=180); plt.close(fig)

    # 7) pipeline: environment | one scan | map --------------------------------------------
    env = np.asarray(mesh.sample_points_uniformly(45000).points)
    k = len(scans) // 2
    scan_w = (np.asarray(scans[k]) @ gt[k][:3, :3].T) + gt[k][:3, 3]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, pts, ttl, col in [(axes[0], env, "1. environment (rubble)", "0.25"),
                              (axes[1], scan_w, "2. one lidar scan", "tab:green"),
                              (axes[2], rec, "3. stitched 3D map", "0.15")]:
        ax.scatter(pts[:, 0], pts[:, 1], s=0.5, c=col, linewidths=0)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([]); ax.set_title(ttl, fontsize=13)
    fig.tight_layout(); fig.savefig(P("pres_pipeline.png"), dpi=170); plt.close(fig)

    print(f"wrote figures to {OUT}")


if __name__ == "__main__":
    main()
