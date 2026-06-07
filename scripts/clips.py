#!/usr/bin/env python3
"""
Generate presentation VIDEO clips from the v2 compartment-maze SLAM run -> out/presentation/.

  clip_buildup.mp4    - top-down white map drawing itself as the robot explores (+ path)
  clip_spin3d.mp4     - clean 360 orbit of the final height-colored 3D map (no labels)
  clip_scan_sweep.mp4 - faint full map + the single current lidar scan sweeping along the path

Run:  ./.venv/bin/python scripts/clips.py
"""
import glob
import os
import sys
import numpy as np
import open3d as o3d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

sys.path.insert(0, os.path.dirname(__file__))
import smoke_demo as sd

OUT = os.path.join(os.path.dirname(__file__), "..", "out", "presentation")
FF = next((p for p in [__import__("shutil").which("ffmpeg")] +
          glob.glob(os.path.expanduser("~/miniforge3/envs/*/bin/ffmpeg")) if p and os.path.exists(p)), None)


def writer(fps):
    return FFMpegWriter(fps=fps, extra_args=["-pix_fmt", "yuv420p",
                                             "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2"])


def main():
    os.makedirs(OUT, exist_ok=True)
    if FF:
        plt.rcParams["animation.ffmpeg_path"] = FF
    else:
        raise SystemExit("no ffmpeg found")

    print("running v2 pipeline ...")
    mesh, gt = sd.make_compartments()
    rc = o3d.t.geometry.RaycastingScene()
    rc.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    dirs = sd.lidar_rays()
    scans = [sd.simulate_scan(rc, T, dirs) for T in gt]
    odom = sd.dead_reckon(gt)
    frames = list(sd.fused_frames(scans, odom, gt[0], voxel=0.06))
    est = np.array([T[:3, 3] for T, _ in frames])
    worlds = [w for _, w in frames]
    allmap = np.vstack(worlds)
    lo, hi = allmap.min(0) - 0.3, allmap.max(0) + 0.3
    ctr, half = (lo + hi) / 2, (hi - lo) / 2
    rng = np.random.default_rng(0)
    N = len(worlds)

    # --- clip 1: white top-down build-up -------------------------------------------------
    print("clip_buildup ...")
    fig = plt.figure(figsize=(10, 8)); fig.patch.set_facecolor("black")
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_facecolor("black"); ax.axis("off"); ax.set_aspect("equal")
    F1 = 80

    def d1(f):
        ax.cla(); ax.set_facecolor("black"); ax.axis("off"); ax.set_aspect("equal")
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1])
        s = max(1, int(round((f + 1) / F1 * N)))
        acc = np.vstack(worlds[:s])
        if len(acc) > 16000:
            acc = acc[rng.choice(len(acc), 16000, replace=False)]
        ax.scatter(acc[:, 0], acc[:, 1], s=0.6, c="white", linewidths=0)
        ax.plot(est[:s, 0], est[:s, 1], color="red", lw=1.6)
        return []
    FuncAnimation(fig, d1, frames=F1, blit=False).save(
        os.path.join(OUT, "clip_buildup.mp4"), writer=writer(12), savefig_kwargs={"facecolor": "black"})
    plt.close(fig)

    # --- clip 2: clean 360 spin of the final 3D map --------------------------------------
    print("clip_spin3d ...")
    disp = allmap if len(allmap) <= 16000 else allmap[rng.choice(len(allmap), 16000, replace=False)]
    fig = plt.figure(figsize=(9, 8)); fig.patch.set_facecolor("black")
    ax = fig.add_subplot(111, projection="3d"); ax.set_axis_off()
    F2 = 90

    def d2(f):
        ax.cla(); ax.set_axis_off(); ax.set_position([-0.05, -0.05, 1.1, 1.1])
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(0, max(2.0, hi[2]))
        ax.set_box_aspect((half[0], half[1], max(2.0, hi[2]) * 1.3))
        ax.scatter(disp[:, 0], disp[:, 1], disp[:, 2], c=disp[:, 2], cmap="turbo", s=2.0,
                   depthshade=False, vmin=0, vmax=max(2.0, hi[2]))
        ax.view_init(elev=38, azim=-180 + 360 * f / F2)
        return []
    FuncAnimation(fig, d2, frames=F2, blit=False).save(
        os.path.join(OUT, "clip_spin3d.mp4"), writer=writer(15), savefig_kwargs={"facecolor": "black"})
    plt.close(fig)

    # --- clip 3: single lidar scan sweeping over the faint map ---------------------------
    print("clip_scan_sweep ...")
    fig = plt.figure(figsize=(10, 8)); fig.patch.set_facecolor("black")
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_facecolor("black"); ax.axis("off"); ax.set_aspect("equal")
    faint = allmap if len(allmap) <= 14000 else allmap[rng.choice(len(allmap), 14000, replace=False)]
    F3 = min(90, N)

    def d3(f):
        ax.cla(); ax.set_facecolor("black"); ax.axis("off"); ax.set_aspect("equal")
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1])
        i = int(f / F3 * N)
        ax.scatter(faint[:, 0], faint[:, 1], s=0.4, c="#444444", linewidths=0)   # full map (context)
        cur = worlds[i]                                                          # current scan
        ax.scatter(cur[:, 0], cur[:, 1], s=2.0, c="cyan", linewidths=0)
        ax.scatter([est[i, 0]], [est[i, 1]], c="white", edgecolors="red", s=90, zorder=5)
        return []
    FuncAnimation(fig, d3, frames=F3, blit=False).save(
        os.path.join(OUT, "clip_scan_sweep.mp4"), writer=writer(12), savefig_kwargs={"facecolor": "black"})
    plt.close(fig)

    print(f"wrote 3 clips to {OUT}")


if __name__ == "__main__":
    main()
