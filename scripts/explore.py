#!/usr/bin/env python3
"""
3D exploration reconstruction -- the wow shot.

A small robot autonomously explores a maze of rubble void-compartments (no target -- it just
visits every reachable space). As it crawls, a dense 3D map of the rubble materializes,
height-colored, while the camera flies in and orbits. An inset top-down map shows the
"fog of war" clearing as space gets explored. Pure-software 3D (matplotlib) -> runs headless.

Outputs: out/exploration_3d.gif + out/exploration_3d.png (final still).

Run:
  ./.venv/bin/python scripts/explore.py
  ./.venv/bin/python scripts/explore.py --frames 120 --fps 15
"""
import argparse
import glob
import os
import shutil
import subprocess
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import open3d as o3d

sys.path.insert(0, os.path.dirname(__file__))
import smoke_demo as sd

OUT = os.path.join(os.path.dirname(__file__), "..", "out", "demos")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=90)
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--max-disp", type=int, default=14000)
    ap.add_argument("--out", default=os.path.join(OUT, "exploration_3d.gif"))
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    print("building rubble compartment maze + exploration path ...")
    mesh, gt = sd.make_compartments()
    rc = o3d.t.geometry.RaycastingScene()
    rc.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    dirs = sd.lidar_rays()
    scans = [sd.simulate_scan(rc, T, dirs) for T in gt]
    odom = sd.dead_reckon(gt)

    print(f"exploring + stitching {len(scans)} frames (dense map) ...")
    frames = list(sd.fused_frames(scans, odom, gt[0], voxel=0.06))   # denser cloud
    worlds = [w for _, w in frames]
    est_xy = np.array([T[:3, 3] for T, _ in frames])

    allmap = np.vstack(worlds)
    _seen, _cov = set(), []                                 # true coverage = unique map cells seen
    for w in worlds:
        _seen |= set(map(tuple, (w[:, :2] // 0.15).astype(int))); _cov.append(len(_seen))
    cov_frac = np.array(_cov) / _cov[-1]
    full_xy = allmap[:, :2]
    lo, hi = allmap.min(0) - 0.3, allmap.max(0) + 0.3
    ctr, half = (lo + hi) / 2, (hi - lo) / 2
    zmax = max(2.0, hi[2])
    rng = np.random.default_rng(0)
    N, F = len(worlds), args.frames

    fig = plt.figure(figsize=(9.5, 7.2)); fig.patch.set_facecolor("black")
    ax = fig.add_subplot(111, projection="3d")
    inset = fig.add_axes([0.70, 0.04, 0.29, 0.33]); inset.set_facecolor("#0a0a0a")

    def draw(f):
        prog = f / max(1, F - 1)
        s = max(1, int(round((f + 1) / F * N)))            # scans revealed so far
        acc = np.vstack(worlds[:s])
        disp = acc if len(acc) <= args.max_disp else acc[rng.choice(len(acc), args.max_disp, replace=False)]

        # --- main 3D panel: dense height-colored map + path, fly-in + orbit ---
        ax.cla(); ax.set_facecolor("black"); ax.set_axis_off()
        ax.set_position([-0.04, -0.02, 1.08, 1.06])
        zoom = 1.32 - 0.32 * min(1.0, prog / 0.4)          # fly in over the first 40%
        L, Hh = ctr - half * zoom, ctr + half * zoom
        ax.set_xlim(L[0], Hh[0]); ax.set_ylim(L[1], Hh[1]); ax.set_zlim(0, zmax)
        ax.set_box_aspect((Hh[0] - L[0], Hh[1] - L[1], zmax * 2.6))
        ax.scatter(disp[:, 0], disp[:, 1], disp[:, 2], c=disp[:, 2], cmap="turbo",
                   s=2.4, depthshade=False, vmin=0, vmax=zmax)
        ax.plot(est_xy[:s, 0], est_xy[:s, 1], 0.32 + 0 * est_xy[:s, 0], color="cyan", lw=2.2, alpha=0.95)
        p = est_xy[s - 1]
        ax.scatter([p[0]], [p[1]], [0.42], c="white", edgecolors="red", s=120, depthshade=False)
        ax.view_init(elev=72 - 30 * min(1.0, prog / 0.4), azim=-80 + 130 * prog)

        # --- inset top-down: fog of war (faint=unknown, bright=explored) ---
        inset.cla(); inset.set_facecolor("#0a0a0a")
        inset.set_xticks([]); inset.set_yticks([]); inset.set_aspect("equal")
        for sp in inset.spines.values():
            sp.set_color("0.4")
        inset.scatter(full_xy[:, 0], full_xy[:, 1], s=0.5, c="#333333", linewidths=0)   # unknown
        d2 = disp
        inset.scatter(d2[:, 0], d2[:, 1], s=0.6, c=d2[:, 2], cmap="turbo", vmin=0, vmax=zmax, linewidths=0)
        inset.plot(est_xy[:s, 0], est_xy[:s, 1], color="cyan", lw=1.0)
        inset.scatter([p[0]], [p[1]], c="white", edgecolors="red", s=30, zorder=5)
        inset.set_title(f"mapped {100.0 * cov_frac[s - 1]:4.0f}%", color="0.8", fontsize=9)

        fig.suptitle(f"Autonomous rubble exploration — live 3D reconstruction   ({s}/{N} scans)",
                     color="white", fontsize=12.5, y=0.97)
        return []

    print(f"rendering {F}-frame fly-in orbit -> {args.out} ...")
    anim = FuncAnimation(fig, draw, frames=F, blit=False)
    anim.save(args.out, writer=PillowWriter(fps=args.fps), savefig_kwargs={"facecolor": "black"})
    draw(F - 1)
    fig.savefig(os.path.join(OUT, "exploration_3d.png"), dpi=140, facecolor="black")
    plt.close()
    print("done:", args.out, "and out/exploration_3d.png")

    # also write an MP4 (smaller, plays everywhere) if an ffmpeg is reachable
    ff = shutil.which("ffmpeg") or next(
        (p for p in glob.glob(os.path.expanduser("~/miniforge3/envs/*/bin/ffmpeg")) if os.path.exists(p)), None)
    mp4 = args.out.replace(".gif", ".mp4")
    if ff:
        subprocess.run([ff, "-y", "-i", args.out, "-movflags", "+faststart", "-pix_fmt", "yuv420p",
                        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", mp4],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("wrote", mp4)
    else:
        print("(no ffmpeg found -> GIF only)")


if __name__ == "__main__":
    main()
