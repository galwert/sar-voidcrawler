#!/usr/bin/env python3
"""
Presentation screenshots of the interactive SAR operator console (out/poc/console.html).
Composites real video frames into the console UI so the panels are never blank.

Outputs: out/poc/shot1_map.png, shot2_inspect_cam.png, shot3_inspect_seismic.png
Run:  ./.venv/bin/python scripts/console_shots.py
"""
import os
import matplotlib
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.join(os.path.dirname(__file__), "..")
POC = os.path.join(ROOT, "out", "poc")
TTF = os.path.join(matplotlib.get_data_path(), "fonts", "ttf")
W, H = 1280, 800
BG, PANEL, HEAD, LINE = (7, 9, 12), (13, 17, 23), (11, 15, 20), (29, 39, 51)
TXT, DIM, WHT = (199, 210, 221), (107, 120, 134), (234, 242, 251)
CYAN, GREEN, AMBER, RED = (54, 211, 255), (55, 224, 106), (255, 176, 32), (255, 59, 70)


def F(sz, bold=False):
    return ImageFont.truetype(os.path.join(TTF, "DejaVuSansMono-Bold.ttf" if bold else "DejaVuSansMono.ttf"), sz)


def fit(im, bw, bh):
    s = min(bw / im.width, bh / im.height)
    return im.resize((max(1, int(im.width * s)), max(1, int(im.height * s))), Image.LANCZOS)


def header(d):
    d.rectangle([0, 0, W, 52], fill=HEAD); d.line([0, 53, W, 53], fill=LINE)
    d.polygon([(22, 26), (29, 16), (43, 16), (50, 26), (43, 36), (29, 36)], outline=CYAN, width=2)  # hex logo
    d.text((60, 17), "SAR-BOT", font=F(18, 1), fill=WHT)
    d.text((158, 19), "· OPERATOR CONSOLE", font=F(15), fill=DIM)
    d.ellipse([330, 21, 342, 33], fill=RED)
    d.text((350, 18), "LIVE", font=F(15), fill=RED)
    d.text((W - 190, 18), "T 14:58:27", font=F(15), fill=DIM)


def panel(d, box, cap):
    d.rounded_rectangle(box, radius=8, fill=PANEL, outline=LINE, width=1)
    d.text((box[0] + 14, box[1] + 9), cap, font=F(12), fill=DIM)
    d.line([box[0] + 1, box[1] + 31, box[2] - 1, box[1] + 31], fill=LINE)


def stat(d, x, y, w, k, v, vc=TXT):
    d.text((x + 14, y), k, font=F(13), fill=DIM)
    d.text((x + w - 14 - d.textlength(v, font=F(13)), y), v, font=F(13), fill=vc)
    d.line([x + 8, y + 26, x + w - 8, y + 26], fill=(19, 26, 34))


def alert_pin(d, cx, cy, label):
    for r, a in [(26, 40), (20, 90)]:                                  # pulse rings
        d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(255, 59, 70), width=2)
    d.ellipse([cx - 11, cy - 11, cx + 11, cy + 11], fill=RED)
    d.text((cx - 3, cy - 9, ), "!", font=F(15, 1), fill=(255, 255, 255))
    tw = d.textlength(label, font=F(11)) + 14
    d.rounded_rectangle([cx - tw / 2, cy + 18, cx + tw / 2, cy + 38], radius=4, fill=(190, 30, 38))
    d.text((cx - tw / 2 + 7, cy + 22), label, font=F(11), fill=(255, 255, 255))


def done_pin(d, cx, cy, label):
    d.ellipse([cx - 8, cy - 8, cx + 8, cy + 8], outline=GREEN, width=2)
    d.line([(cx - 4, cy), (cx - 1, cy + 4), (cx + 5, cy - 4)], fill=GREEN, width=2)
    d.text((cx + 14, cy - 6), label, font=F(10), fill=GREEN)


def build_map():
    img = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(img)
    header(d)
    mp = [14, 64, 938, 786]
    panel(d, mp, "ROBOT FEED · SLAM MAP    SECTOR B / VOID 3")
    ix0, iy0, iw, ih = mp[0] + 10, mp[1] + 40, mp[2] - mp[0] - 20, mp[3] - mp[1] - 50
    frame = fit(Image.open(os.path.join(POC, ".map_frame.png")).convert("RGB"), iw, ih)
    fx, fy = ix0 + (iw - frame.width) // 2, iy0 + (ih - frame.height) // 2
    img.paste(frame, (fx, fy))
    d.rectangle([fx, fy, fx + frame.width, fy + frame.height], outline=(40, 52, 66))
    fw, fh = frame.width, frame.height
    done_pin(d, fx + int(.30 * fw), fy + int(.64 * fh), "IP-01")
    done_pin(d, fx + int(.44 * fw), fy + int(.40 * fh), "IP-02")
    done_pin(d, fx + int(.67 * fw), fy + int(.33 * fh), "IP-03")
    alert_pin(d, fx + int(.55 * fw), fy + int(.50 * fh), "IP-04 · LIFE-SIGN ALERT")

    # sidebar
    sx, sw = 952, 314
    panel(d, [sx, 64, sx + sw, 300], "MISSION STATUS")
    y = 110
    for k, v, c in [("Hotspots reached", "4 / 4", GREEN), ("Building mapped", "84 %", TXT),
                    ("Steps travelled", "893", TXT), ("Comms", "link OK", GREEN),
                    ("Life-sign alerts", "1 active", AMBER)]:
        stat(d, sx, y, sw, k, v, c); y += 36
    panel(d, [sx, 312, sx + sw, 786], "ALERTS")
    ar = [sx + 12, 356, sx + sw - 12, 470]                             # active alert card
    d.rounded_rectangle(ar, radius=7, fill=(26, 14, 17), outline=RED, width=1)
    d.ellipse([ar[0] + 12, ar[1] + 14, ar[0] + 30, ar[1] + 32], outline=RED, width=2)
    d.text((ar[0] + 18, ar[1] + 13), "!", font=F(15, 1), fill=RED)
    d.text((ar[0] + 42, ar[1] + 14), "IP-04 — anomaly", font=F(14, 1), fill=(255, 210, 213))
    d.text((ar[0] + 42, ar[1] + 38), "Camera flagged a shape under a", font=F(11), fill=DIM)
    d.text((ar[0] + 42, ar[1] + 54), "collapsed slab. Life unconfirmed.", font=F(11), fill=DIM)
    d.text((ar[0] + 42, ar[1] + 80), "▶ INSPECT FEED", font=F(11, 1), fill=RED)
    cr = [sx + 12, 484, sx + sw - 12, 548]                             # cleared card
    d.rounded_rectangle(cr, radius=7, fill=(14, 26, 18), outline=(22, 51, 31), width=1)
    done_pin(d, cr[0] + 21, cr[1] + 24, "")
    d.text((cr[0] + 42, cr[1] + 12), "IP-01 / 02 / 03", font=F(14, 1), fill=(189, 236, 203))
    d.text((cr[0] + 42, cr[1] + 36), "Cleared — no life signs.", font=F(11), fill=DIM)
    return img


def modal(base, seismic):
    img = base.convert("RGBA")
    img = Image.alpha_composite(img, Image.new("RGBA", (W, H), (0, 0, 0, 165)))
    d = ImageDraw.Draw(img)
    mx0, my0, mx1, my1 = 196, 96, 1084, 720
    d.rounded_rectangle([mx0, my0, mx1, my1], radius=10, fill=(10, 14, 19, 255), outline=LINE, width=1)
    tx, ty = mx0 + 16, my0 + 14
    d.text((tx, ty), "INSPECTING ", font=F(13), fill=WHT); tx += d.textlength("INSPECTING ", font=F(13))
    d.text((tx, ty), "IP-04", font=F(13, 1), fill=RED); tx += d.textlength("IP-04", font=F(13, 1))
    d.text((tx, ty), "  · robot camera + seismic sensor", font=F(13), fill=DIM)
    d.text((mx1 - 30, my0 + 11), "✕", font=F(18), fill=DIM)
    d.line([mx0 + 1, my0 + 44, mx1 - 1, my0 + 44], fill=LINE)
    dem = os.path.join(ROOT, "out", "demos")
    vid = fit(Image.open(os.path.join(dem, "inspect_hand_seismic.png" if seismic else "inspect_hand_camera.png")
                         ).convert("RGB"), mx1 - mx0 - 28, 470)
    vx, vy = mx0 + (mx1 - mx0 - vid.width) // 2, my0 + 56
    img.paste(vid, (vx, vy))
    # control bar
    cy = vy + vid.height + 16
    d.line([mx0 + 1, cy - 8, mx1 - 1, cy - 8], fill=LINE)

    def tab(x, txt, active, col):
        w = d.textlength(txt, font=F(12)) + 28
        fill = col if active else (16, 22, 29)
        d.rounded_rectangle([x, cy, x + w, cy + 30], radius=6, fill=fill,
                            outline=col if active else LINE, width=1)
        tc = (0, 8, 10) if active and col == CYAN else ((255, 255, 255) if active else DIM)
        d.text((x + 14, cy + 7), txt, font=F(12, 1 if active else 0), fill=tc)
        return x + w + 10
    nx = tab(mx0 + 16, "CAM-1 · VISUAL", not seismic, CYAN)
    tab(nx, "◉ ACTIVATE SEISMIC", seismic, RED)
    if seismic:
        d.ellipse([mx1 - 360, cy + 8, mx1 - 348, cy + 20], fill=RED)
        d.text((mx1 - 340, cy + 6), "HEARTBEAT 72 BPM — SURVIVOR CONFIRMED", font=F(12, 1), fill=RED)
    else:
        d.text((mx1 - 290, cy + 7), "VISUAL ONLY — life unconfirmed", font=F(12), fill=AMBER)
    return img.convert("RGB")


def main():
    m = build_map()
    m.save(os.path.join(POC, "shot1_map.png"))
    modal(m, False).save(os.path.join(POC, "shot2_inspect_cam.png"))
    modal(m, True).save(os.path.join(POC, "shot3_inspect_seismic.png"))
    print("wrote shot1_map.png, shot2_inspect_cam.png, shot3_inspect_seismic.png in out/poc/")


if __name__ == "__main__":
    main()
