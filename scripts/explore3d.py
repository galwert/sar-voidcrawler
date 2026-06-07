#!/usr/bin/env python3
"""
Full-3D (6-DOF) tunnel exploration -- the robot crawls a genuinely 3D void network carved
through a rubble VOLUME, climbing and dropping through the tunnels while a 3D map is stitched.

Unlike the 2.5D floor-plan demo, here the path varies in height and the robot pitches on the
slopes, so the reconstruction is truly volumetric. Pure-software 3D (matplotlib), headless.

Outputs: out/explore3d.gif + .mp4 + out/explore3d.png

Run:
  ./.venv/bin/python scripts/explore3d.py
  ./.venv/bin/python scripts/explore3d.py --frames 120 --fps 15
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
    ap.add_argument("--max-disp", type=int, default=15000)
    ap.add_argument("--env", choices=["maze", "tunnels"], default="maze",
                    help="maze = readable compartment maze on gentle terrain; tunnels = carved tube network")
    ap.add_argument("--out", default=os.path.join(OUT, "explore3d.gif"))
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    if args.env == "tunnels":
        print("building 3D rubble tunnel network ...")
        mesh, gt = sd.make_tunnels_3d()
        dirs = sd.lidar_rays(channels=24, h_samples=400, v_fov_deg=110)
    else:
        print("building compartment maze on gentle 3D terrain ...")
        mesh, gt, _ = sd.make_compartments_3d()
        dirs = sd.lidar_rays(channels=20, h_samples=400, v_fov_deg=80)
    rc = o3d.t.geometry.RaycastingScene()
    rc.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    scans = [sd.simulate_scan(rc, T, dirs) for T in gt]
    odom = sd.dead_reckon(gt)

    print(f"exploring + stitching {len(scans)} frames (6-DOF) ...")
    frames = list(sd.fused_frames(scans, odom, gt[0], voxel=0.06))
    worlds = [w for _, w in frames]
    est = np.array([T[:3, 3] for T, _ in frames])          # 3D estimated positions

    allmap = np.vstack(worlds)
    _seen, _cov = set(), []                                 # true coverage = unique map cells seen
    for w in worlds:
        _seen |= set(map(tuple, (w[:, :2] // 0.15).astype(int))); _cov.append(len(_seen))
    cov_frac = np.array(_cov) / _cov[-1]
    full_xy = allmap[:, :2]
    lo, hi = allmap.min(0) - 0.3, allmap.max(0) + 0.3
    ctr, half = (lo + hi) / 2, (hi - lo) / 2
    zlo, zhi = lo[2], hi[2]
    rng = np.random.default_rng(0)
    N, F = len(worlds), args.frames

    fig = plt.figure(figsize=(9.5, 7.4)); fig.patch.set_facecolor("black")
    ax = fig.add_subplot(111, projection="3d")
    inset = fig.add_axes([0.71, 0.04, 0.28, 0.30]); inset.set_facecolor("#0a0a0a")

    def draw(f):
        prog = f / max(1, F - 1)
        s = max(1, int(round((f + 1) / F * N)))
        acc = np.vstack(worlds[:s])
        disp = acc if len(acc) <= args.max_disp else acc[rng.choice(len(acc), args.max_disp, replace=False)]

        # --- main 3D panel: height-colored volumetric map + 3D path, fly-in + orbit ---
        ax.cla(); ax.set_facecolor("black"); ax.set_axis_off()
        ax.set_position([-0.04, -0.02, 1.08, 1.06])
        zoom = 1.3 - 0.3 * min(1.0, prog / 0.4)
        L, Hh = ctr - half * zoom, ctr + half * zoom
        ax.set_xlim(L[0], Hh[0]); ax.set_ylim(L[1], Hh[1]); ax.set_zlim(L[2], Hh[2])
        ax.set_box_aspect((Hh[0] - L[0], Hh[1] - L[1], (Hh[2] - L[2]) * 1.8))
        ax.scatter(disp[:, 0], disp[:, 1], disp[:, 2], c=disp[:, 2], cmap="turbo",
                   s=2.2, depthshade=False, vmin=zlo, vmax=zhi)
        ax.plot(est[:s, 0], est[:s, 1], est[:s, 2], color="cyan", lw=2.2, alpha=0.95)
        p = est[s - 1]
        ax.scatter([p[0]], [p[1]], [p[2]], c="white", edgecolors="red", s=120, depthshade=False)
        ax.view_init(elev=54 - 14 * min(1.0, prog / 0.4), azim=-90 + 150 * prog)

        # --- inset top-down: coverage (3D path projected) ---
        inset.cla(); inset.set_facecolor("#0a0a0a")
        inset.set_xticks([]); inset.set_yticks([]); inset.set_aspect("equal")
        for sp in inset.spines.values():
            sp.set_color("0.4")
        inset.scatter(full_xy[:, 0], full_xy[:, 1], s=0.4, c="#333333", linewidths=0)
        inset.scatter(disp[:, 0], disp[:, 1], s=0.5, c=disp[:, 2], cmap="turbo",
                      vmin=zlo, vmax=zhi, linewidths=0)
        inset.plot(est[:s, 0], est[:s, 1], color="cyan", lw=1.0)
        inset.scatter([p[0]], [p[1]], c="white", edgecolors="red", s=28, zorder=5)
        inset.set_title(f"mapped {100.0 * cov_frac[s - 1]:4.0f}%", color="0.8", fontsize=9)

        fig.suptitle(f"Autonomous 3D rubble exploration — 6-DOF reconstruction   ({s}/{N} scans)",
                     color="white", fontsize=12.5, y=0.97)
        return []

    print(f"rendering {F}-frame 3D orbit -> {args.out} ...")
    anim = FuncAnimation(fig, draw, frames=F, blit=False)
    anim.save(args.out, writer=PillowWriter(fps=args.fps), savefig_kwargs={"facecolor": "black"})
    draw(F - 1)
    fig.savefig(os.path.join(OUT, "explore3d.png"), dpi=140, facecolor="black")
    plt.close()
    print("done:", args.out)

    ff = shutil.which("ffmpeg") or next(
        (p for p in glob.glob(os.path.expanduser("~/miniforge3/envs/*/bin/ffmpeg")) if os.path.exists(p)), None)
    mp4 = args.out.replace(".gif", ".mp4")
    if ff:
        subprocess.run([ff, "-y", "-i", args.out, "-movflags", "+faststart", "-pix_fmt", "yuv420p",
                        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", mp4],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("wrote", mp4)


if __name__ == "__main__":
    main()
