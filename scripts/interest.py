#!/usr/bin/env python3
"""
Interest-point inspection demo (camera + seismic vital-signs sensor).

The robot has reached a known interest point. It pans its camera in the dark, lit only by its
own flashlight, finds a shape that looks like a person, and boxes it. The camera can only see a
*shape* -> "possible survivor, unconfirmed". Then it activates the SEISMIC sensor, which detects
a heartbeat (red pulse + ECG) -> "SURVIVOR CONFIRMED".

Renders two videos: out/demos/interest_camera.mp4 (no seismic) and interest_seismic.mp4.

Run:  ./.venv/bin/python scripts/interest.py
"""
import glob
import os
import shutil
import subprocess
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.patches import Rectangle
from PIL import Image, ImageDraw

OUT = os.path.join(os.path.dirname(__file__), "..", "out", "demos")
W, H = 960, 540
FF = next((p for p in [shutil.which("ffmpeg")] +
          glob.glob(os.path.expanduser("~/miniforge3/envs/*/bin/ffmpeg")) if p and os.path.exists(p)), None)


def build_scene(seed=3):
    rng = np.random.default_rng(seed)
    img = Image.new("RGB", (W, H), (12, 10, 9))
    d = ImageDraw.Draw(img)
    for yy in range(int(H * 0.6), H):                      # floor gradient
        c = int(16 + (yy - H * 0.6) / (H * 0.4) * 16)
        d.line([(0, yy), (W, yy)], fill=(c, int(c * 0.82), int(c * 0.62)))

    def rock(rx, ry, s, g):
        n = rng.integers(5, 8); a = np.sort(rng.uniform(0, 2 * np.pi, n))
        d.polygon([(rx + s * np.cos(t) * rng.uniform(0.6, 1), ry + s * 0.7 * np.sin(t) * rng.uniform(0.6, 1))
                   for t in a], fill=(g, int(g * 0.95), int(g * 0.9)))

    for _ in range(80):                                    # scattered rubble
        rock(rng.uniform(0, W), rng.uniform(H * 0.28, H), rng.uniform(14, 58), int(rng.uniform(50, 110)))

    px, py, L = int(W * 0.57), int(H * 0.6), 230           # person, lying, partially buried
    clothing, skin, pants = (74, 84, 116), (203, 161, 131), (60, 55, 72)
    d.ellipse([px - L * 0.42, py - 36, px + L * 0.08, py + 36], fill=clothing)      # torso
    hx = int(px - L * 0.46)
    d.ellipse([hx - 26, py - 26, hx + 26, py + 26], fill=skin)                      # head
    d.ellipse([px + L * 0.02, py - 24, px + L * 0.5, py + 24], fill=pants)          # legs
    d.ellipse([px - L * 0.22, py + 18, px - L * 0.02, py + 58], fill=clothing)      # arm
    d.ellipse([px - L * 0.28, py + 50, px - L * 0.14, py + 70], fill=skin)          # hand
    for _ in range(7):                                     # rubble on top (trapped)
        rock(rng.uniform(px - L * 0.4, px + L * 0.5), rng.uniform(py - 8, py + 46),
             rng.uniform(16, 38), int(rng.uniform(60, 100)))

    scene = np.asarray(img).astype(float) / 255.0
    bbox = (int(px - L * 0.52), int(py - 52), int(px + L * 0.56), int(py + 74))
    return scene, bbox, (px, py)


def render(seismic, scene, bbox, person):
    px, py = person
    x0, y0, x1, y1 = bbox
    yy, xx = np.mgrid[0:H, 0:W]
    pmask = np.exp(-(((xx - px) / 150.0) ** 2 + ((yy - py) / 90.0) ** 2))   # person glow region
    warm = np.array([1.0, 0.93, 0.82])
    rng = np.random.default_rng(0)
    F = 120 if seismic else 78
    fps = 15

    fig = plt.figure(figsize=(9.6, 5.4)); fig.patch.set_facecolor("black")
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")
    vit = fig.add_axes([0.05, 0.05, 0.46, 0.13]); vit.axis("off")

    def status(f):
        if f < 34:
            return "SCANNING…", "0.8"
        if f < 54:
            return "POSSIBLE SURVIVOR  —  VISUAL ONLY", "yellow"
        if not seismic:
            return "CANNOT CONFIRM LIFE  —  shape only", "orange"
        if f < 70:
            return "ACTIVATING SEISMIC SENSOR…", "cyan"
        return "♥ HEARTBEAT 72 BPM  —  SURVIVOR CONFIRMED", "#ff3b3b"

    def draw(f):
        ax.cla(); ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")
        # flashlight position: sweep, then settle on the person
        if f < 25:
            fx = W * (0.13 + 0.75 * f / 25)
        elif f < 34:
            fx = W * 0.88 + (px - W * 0.88) * (f - 25) / 9
        else:
            fx = px
        fyy = py if f >= 34 else H * 0.52 + 14 * np.sin(f * 0.4)
        light = 0.05 + 0.95 * np.exp(-(((xx - fx) / 165.0) ** 2 + ((yy - fyy) / 150.0) ** 2))
        frame = np.clip(scene * (light[..., None] * warm), 0, 1)
        frame *= (1 - 0.04 * (yy % 2))[..., None]          # scanlines
        frame = np.clip(frame + rng.normal(0, 0.012, frame.shape), 0, 1)   # grain

        hb = seismic and f >= 70
        if hb:                                             # red heartbeat pulse on the person
            t = f / fps
            pulse = 0.35 + 0.65 * np.clip(np.sin(2 * np.pi * t / 0.83 - 1.0), 0, 1) ** 3
            a = (0.75 * pulse) * pmask
            frame[..., 0] = np.clip(frame[..., 0] + a, 0, 1)
            frame[..., 1] = np.clip(frame[..., 1] - 0.25 * a, 0, 1)
            frame[..., 2] = np.clip(frame[..., 2] - 0.25 * a, 0, 1)
        ax.imshow(frame, extent=[0, W, H, 0], zorder=0)

        # detection box (after the camera locks on)
        if f >= 40:
            col = "#ff3b3b" if hb else "yellow"
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec=col, lw=2.2, zorder=5))
            lbl = "SURVIVOR ♥" if hb else "PERSON ?  0.84"
            ax.text(x0, y0 - 8, lbl, color=col, fontsize=11, family="monospace", weight="bold")

        # HUD
        for cx, cy, dx, dy in [(18, 18, 1, 1), (W - 18, 18, -1, 1), (18, H - 18, 1, -1), (W - 18, H - 18, -1, -1)]:
            ax.plot([cx, cx + 26 * dx], [cy, cy], color="0.7", lw=1.4)
            ax.plot([cx, cx], [cy, cy + 26 * dy], color="0.7", lw=1.4)
        ax.plot([W / 2 - 12, W / 2 + 12], [H / 2, H / 2], color="0.45", lw=1)
        ax.plot([W / 2, W / 2], [H / 2 - 12, H / 2 + 12], color="0.45", lw=1)
        if f % 2 == 0:
            ax.text(34, 34, "● REC", color="#ff3b3b", fontsize=11, family="monospace")
        ax.text(W - 34, 34, "SAR-BOT // CAM-1 // FLASH", color="0.7", fontsize=9,
                family="monospace", ha="right")
        ax.text(W - 34, H - 26, f"SEISMIC: {'ON' if hb else ('…' if (seismic and f >= 54) else 'OFF')}",
                color=("#ff3b3b" if hb else "0.6"), fontsize=10, family="monospace", ha="right")
        msg, col = status(f)
        ax.text(W / 2, H - 28, msg, color=col, fontsize=13, family="monospace", ha="center", weight="bold")

        # vitals ECG strip (only once seismic confirms)
        vit.cla(); vit.axis("off")
        if hb:
            tt = np.linspace(0, 4, 500) + f * 0.18
            ph = (tt % 0.83) / 0.83
            ecg = (np.exp(-((ph - 0.5) ** 2) / (2 * 0.012 ** 2))
                   - 0.3 * np.exp(-((ph - 0.45) ** 2) / (2 * 0.01 ** 2))
                   - 0.22 * np.exp(-((ph - 0.58) ** 2) / (2 * 0.012 ** 2)))
            vit.plot(np.linspace(0, 1, len(ecg)), ecg, color="#ff3b3b", lw=1.6)
            vit.set_ylim(-0.6, 1.2); vit.set_xlim(0, 1)
            vit.text(0.0, 1.25, "VITALS  72 BPM", color="#ff3b3b", fontsize=9, family="monospace")
        return []

    out = os.path.join(OUT, f"interest_{'seismic' if seismic else 'camera'}.mp4")
    anim = FuncAnimation(fig, draw, frames=F, blit=False)
    if FF:
        plt.rcParams["animation.ffmpeg_path"] = FF
        anim.save(out, writer=FFMpegWriter(fps=fps, extra_args=["-pix_fmt", "yuv420p"]),
                  savefig_kwargs={"facecolor": "black"})
    draw(F - 1); fig.savefig(out.replace(".mp4", ".png"), dpi=130, facecolor="black"); plt.close()
    print("done:", out)


def main():
    os.makedirs(OUT, exist_ok=True)
    scene, bbox, person = build_scene()
    render(False, scene, bbox, person)
    render(True, scene, bbox, person)


if __name__ == "__main__":
    main()
