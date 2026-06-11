#!/usr/bin/env python3
"""
Robot camera inspection on a REAL rubble photo. At an interest point the robot pans its camera
LEFT, then RIGHT, then FOCUSES on the survivor and boxes it. The seismic version then confirms a
heartbeat (red pulse + ECG). Camera-feed look: flashlight vignette, HUD, scanlines.

Two scenes (Gemini-generated):
  --scene person   a survivor lying in a void
  --scene hand     a survivor buried, only a hand reaching out

Outputs: out/demos/<prefix>_camera.mp4 and <prefix>_seismic.mp4.

Run:  ./.venv/bin/python scripts/inspect_real.py --scene hand
"""
import argparse
import glob
import os
import shutil
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.patches import Rectangle
from PIL import Image

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(ROOT, "out", "demos")
FF = next((p for p in [shutil.which("ffmpeg")] +
          glob.glob(os.path.expanduser("~/miniforge3/envs/*/bin/ffmpeg")) if p and os.path.exists(p)), None)
OW, OH = 854, 480                                           # 16:9 camera frame

SCENES = {
    "person": dict(img="Gemini_Generated_Image_wyp7g3wyp7g3wyp7.png",
                   pf=(0.30, 0.36, 0.63, 0.72), noun="PERSON", conf="0.93", prefix="inspect"),
    "hand": dict(img="Gemini_Generated_Image_lmh5crlmh5crlmh5.png",
                 pf=(0.45, 0.60, 0.585, 0.93), noun="HAND", conf="0.79", prefix="inspect_hand"),
}


def ease(t):
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)


def render(seismic, sc, src, vign):
    W, H = src.size
    px0, py0, px1, py1 = sc["pf"][0] * W, sc["pf"][1] * H, sc["pf"][2] * W, sc["pf"][3] * H
    pcx, pcy = (px0 + px1) / 2, (py0 + py1) / 2
    cw0 = min(0.55 * W, H * OW / OH * 0.95)                 # wide scan crop (fits image)
    fw = max((px1 - px0) * 1.9, (py1 - py0) * 1.9 * OW / OH)
    rng = np.random.default_rng(0)
    fps, F = 15, (112 if seismic else 80)
    yy, xx = np.mgrid[0:OH, 0:OW]
    noun, conf = sc["noun"], sc["conf"]

    def viewport(f):
        if f < 22:
            t = ease(f / 22); return (W / 2) * (1 - t) + (cw0 / 2) * t, H / 2, cw0
        if f < 52:
            t = ease((f - 22) / 30); return (cw0 / 2) * (1 - t) + (W - cw0 / 2) * t, H / 2, cw0
        if f < 68:
            t = ease((f - 52) / 16)
            return (W - cw0 / 2) * (1 - t) + pcx * t, (H / 2) * (1 - t) + pcy * t, cw0 * (1 - t) + fw * t
        return pcx, pcy, fw

    fig = plt.figure(figsize=(OW / 100, OH / 100), dpi=100); fig.patch.set_facecolor("black")
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, OW); ax.set_ylim(OH, 0); ax.axis("off")
    vit = fig.add_axes([0.04, 0.06, 0.40, 0.13]); vit.axis("off")

    def draw(f):
        ax.cla(); ax.set_xlim(0, OW); ax.set_ylim(OH, 0); ax.axis("off")
        cx, cy, cw = viewport(f)
        cw = min(cw, W); ch = cw * OH / OW                  # keep crop inside the image
        if ch > H:
            ch = H; cw = ch * OW / OH
        x0 = min(max(cx - cw / 2, 0), W - cw); y0 = min(max(cy - ch / 2, 0), H - ch)
        crop = src.crop((x0, y0, x0 + cw, y0 + ch)).resize((OW, OH), Image.LANCZOS)
        frame = np.asarray(crop).astype(float) / 255.0
        frame *= vign[..., None]
        frame *= (1 - 0.05 * (yy % 2))[..., None]
        frame = np.clip(frame + rng.normal(0, 0.012, frame.shape), 0, 1)

        def to_out(ix, iy):
            return (ix - x0) / cw * OW, (iy - y0) / ch * OH
        ox0, oy0 = to_out(px0, py0); ox1, oy1 = to_out(px1, py1)
        focused = f >= 64
        hb = seismic and f >= 80

        if hb:
            t = f / fps
            pulse = 0.3 + 0.7 * np.clip(np.sin(2 * np.pi * t / 0.83 - 1.0), 0, 1) ** 3
            ocx, ocy = (ox0 + ox1) / 2, (oy0 + oy1) / 2
            a = 0.7 * pulse * np.exp(-(((xx - ocx) / max(40, (ox1 - ox0) * 0.6)) ** 2 +
                                       ((yy - ocy) / max(40, (oy1 - oy0) * 0.6)) ** 2))
            frame[..., 0] = np.clip(frame[..., 0] + a, 0, 1)
            frame[..., 1] = np.clip(frame[..., 1] - 0.3 * a, 0, 1)
            frame[..., 2] = np.clip(frame[..., 2] - 0.3 * a, 0, 1)
        ax.imshow(frame, extent=[0, OW, OH, 0], zorder=0)

        if focused:
            col = "#ff3b3b" if hb else "yellow"
            ax.add_patch(Rectangle((ox0, oy0), ox1 - ox0, oy1 - oy0, fill=False, ec=col, lw=2.4, zorder=5))
            ax.text(ox0, oy0 - 8, "SURVIVOR ♥" if hb else f"{noun}  {conf}",
                    color=col, fontsize=12, family="monospace", weight="bold")

        for cxx, cyy, dx, dy in [(16, 16, 1, 1), (OW - 16, 16, -1, 1), (16, OH - 16, 1, -1), (OW - 16, OH - 16, -1, -1)]:
            ax.plot([cxx, cxx + 24 * dx], [cyy, cyy], color="0.75", lw=1.4)
            ax.plot([cxx, cxx], [cyy, cyy + 24 * dy], color="0.75", lw=1.4)
        ax.plot([OW / 2 - 11, OW / 2 + 11], [OH / 2, OH / 2], color="0.5", lw=1)
        ax.plot([OW / 2, OW / 2], [OH / 2 - 11, OH / 2 + 11], color="0.5", lw=1)
        if f % 2 == 0:
            ax.text(30, 30, "● REC", color="#ff3b3b", fontsize=11, family="monospace")
        ax.text(OW - 30, 30, "SAR-BOT // CAM-1 // FLASH", color="0.8", fontsize=8.5, family="monospace", ha="right")
        ax.text(OW - 30, OH - 22, f"SEISMIC: {'ON' if hb else ('…' if (seismic and f >= 64) else 'OFF')}",
                color=("#ff3b3b" if hb else "0.65"), fontsize=9.5, family="monospace", ha="right")

        if f < 22:
            msg, col = "SCANNING  ◄ LEFT", "0.85"
        elif f < 52:
            msg, col = "SCANNING  RIGHT ►", "0.85"
        elif f < 64:
            msg, col = "CONTACT — FOCUSING…", "yellow"
        elif not seismic:
            msg, col = f"{noun} DETECTED — confirm life?", "yellow"
        elif f < 80:
            msg, col = "ACTIVATING SEISMIC SENSOR…", "cyan"
        else:
            msg, col = "♥ HEARTBEAT — SURVIVOR CONFIRMED", "#ff3b3b"
        ax.text(OW / 2, OH - 24, msg, color=col, fontsize=13, family="monospace", ha="center", weight="bold")

        vit.cla(); vit.axis("off")
        if hb:
            tt = np.linspace(0, 4, 500) + f * 0.18
            ph = (tt % 0.83) / 0.83
            ecg = (np.exp(-((ph - 0.5) ** 2) / (2 * 0.012 ** 2))
                   - 0.3 * np.exp(-((ph - 0.45) ** 2) / (2 * 0.01 ** 2))
                   - 0.22 * np.exp(-((ph - 0.58) ** 2) / (2 * 0.012 ** 2)))
            vit.plot(np.linspace(0, 1, len(ecg)), ecg, color="#ff3b3b", lw=1.6)
            vit.set_xlim(0, 1); vit.set_ylim(-0.6, 1.2)
            vit.text(0.0, 1.25, "VITALS  72 BPM", color="#ff3b3b", fontsize=9, family="monospace")
        return []

    out = os.path.join(OUT, f"{sc['prefix']}_{'seismic' if seismic else 'camera'}.mp4")
    anim = FuncAnimation(fig, draw, frames=F, blit=False)
    if FF:
        plt.rcParams["animation.ffmpeg_path"] = FF
        anim.save(out, writer=FFMpegWriter(fps=fps, extra_args=["-pix_fmt", "yuv420p"]),
                  savefig_kwargs={"facecolor": "black"})
    draw(F - 1); fig.savefig(out.replace(".mp4", ".png"), dpi=110, facecolor="black"); plt.close()
    print("done:", out)


def render_clean(sc, src):
    """Clean inspection: scan left, scan right, lock onto the subject, box + score. No text/HUD."""
    W, H = src.size
    px0, py0, px1, py1 = sc["pf"][0] * W, sc["pf"][1] * H, sc["pf"][2] * W, sc["pf"][3] * H
    pcx, pcy = (px0 + px1) / 2, (py0 + py1) / 2
    bw, bh = px1 - px0, py1 - py0
    cw0 = min(0.55 * W, H * OW / OH * 0.95)                 # wide scan crop
    fw = max(bw * 2.2, bh * 2.2 * OW / OH)                  # focus crop on the subject
    fps, F = 15, 86

    fig = plt.figure(figsize=(OW / 100, OH / 100), dpi=100); fig.patch.set_facecolor("black")
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, OW); ax.set_ylim(OH, 0); ax.axis("off")

    def viewport(f):
        if f < 20:                                          # pan LEFT
            t = ease(f / 20); return (W / 2) * (1 - t) + (cw0 / 2) * t, H / 2, cw0
        if f < 48:                                          # pan RIGHT
            t = ease((f - 20) / 28); return (cw0 / 2) * (1 - t) + (W - cw0 / 2) * t, H / 2, cw0
        if f < 62:                                          # come back + zoom onto the hand
            t = ease((f - 48) / 14)
            return (W - cw0 / 2) * (1 - t) + pcx * t, (H / 2) * (1 - t) + pcy * t, cw0 * (1 - t) + fw * t
        return pcx, pcy, fw                                 # hold

    def draw(f):
        ax.cla(); ax.set_xlim(0, OW); ax.set_ylim(OH, 0); ax.axis("off")
        cx, cy, cw = viewport(f)
        cw = min(cw, W); ch = cw * OH / OW
        if ch > H:
            ch = H; cw = ch * OW / OH
        x0 = min(max(cx - cw / 2, 0), W - cw); y0 = min(max(cy - ch / 2, 0), H - ch)
        crop = src.crop((x0, y0, x0 + cw, y0 + ch)).resize((OW, OH), Image.LANCZOS)
        ax.imshow(np.asarray(crop), extent=[0, OW, OH, 0], zorder=0)
        if f >= 58:                                         # box + score appear as it locks on
            t = min(1.0, (f - 58) / 6.0)
            ox0, oy0 = (px0 - x0) / cw * OW, (py0 - y0) / ch * OH
            ox1, oy1 = (px1 - x0) / cw * OW, (py1 - y0) / ch * OH
            ax.add_patch(Rectangle((ox0, oy0), ox1 - ox0, oy1 - oy0, fill=False, ec="#ffe11a", lw=2.6, alpha=t))
            ax.text(ox0, oy0 - 9, f"HAND  {sc['conf']}", color="#ffe11a", fontsize=15,
                    family="monospace", weight="bold", alpha=t)
        return []

    out = os.path.join(OUT, f"{sc['prefix']}_camera.mp4")
    anim = FuncAnimation(fig, draw, frames=F, blit=False)
    if FF:
        plt.rcParams["animation.ffmpeg_path"] = FF
        anim.save(out, writer=FFMpegWriter(fps=fps, extra_args=["-pix_fmt", "yuv420p"]),
                  savefig_kwargs={"facecolor": "black"})
    draw(F - 1); fig.savefig(out.replace(".mp4", ".png"), dpi=110, facecolor="black"); plt.close()
    print("done (clean):", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", choices=list(SCENES), default="hand")
    ap.add_argument("--clean", action="store_true", help="bare detection: box + score only, no HUD")
    args = ap.parse_args()
    sc = SCENES[args.scene]
    os.makedirs(OUT, exist_ok=True)
    src = Image.open(os.path.join(ROOT, sc["img"])).convert("RGB")
    if args.clean:
        render_clean(sc, src)
        return
    yy, xx = np.mgrid[0:OH, 0:OW]
    r = np.sqrt(((xx - OW / 2) / (OW / 2)) ** 2 + ((yy - OH / 2) / (OH / 2)) ** 2)
    vign = np.clip(1.05 - 0.6 * r ** 2, 0.25, 1.0)
    render(False, sc, src, vign)
    render(True, sc, src, vign)


if __name__ == "__main__":
    main()
