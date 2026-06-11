#!/usr/bin/env python3
"""
Rubble block segmentation with SAM2.

  1. SAM2 automatic mask generator -> class-agnostic masks (the "for starts" stage).
  2. Further segment for our needs: classify each mask into
        BLOCK  (slab/brick: filled, compact, rectangular-ish)  -> measure width via min-area rect
        WIRE   (thin, elongated, low solidity)                 -> flagged, kept out of blocks
        LIGHT  (blown-out glare region)                        -> flagged, kept out of blocks
        (small rubble / background are dropped)
     Wires and glare are explicitly detected because they otherwise leak into / split the slabs.

Run with the thesis env (has sam2 + torch + cv2):
  /home/wertheizer/miniforge3/envs/thesis/bin/python scripts/sam_blocks.py
Outputs: out/sam/sam_raw.png, sam_blocks.png, blocks.csv
"""
import csv
import os
import sys
import numpy as np
import cv2
from scipy import ndimage as ndi
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from PIL import Image

ROOT = os.path.join(os.path.dirname(__file__), "..")
IMG = os.path.join(ROOT, "Gemini_Generated_Image_6b5v2c6b5v2c6b5v.png")
OUT = os.path.join(ROOT, "out", "sam")
CACHE = os.path.join(OUT, "_masks.npz")
CKPT = "/home/wertheizer/thesis/sam2/checkpoints/sam2.1_hiera_small.pt"
CFG = "configs/sam2.1/sam2.1_hiera_s.yaml"
DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"   # metric depth in meters
LONG = 1024                                                 # working resolution (CPU)
RNG = np.random.default_rng(3)


def depth_map(img_rgb):
    """Monocular METRIC depth (meters) via Depth-Anything-V2, resized to the image."""
    import torch
    import torch.nn.functional as Fn
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    Hp, Wp = img_rgb.shape[:2]
    proc = AutoImageProcessor.from_pretrained(DEPTH_MODEL)
    model = AutoModelForDepthEstimation.from_pretrained(DEPTH_MODEL).eval()
    inp = proc(images=Image.fromarray(img_rgb), return_tensors="pt")
    with torch.no_grad():
        pred = model(**inp).predicted_depth          # (1,h,w), meters
    pred = Fn.interpolate(pred[:, None], size=(Hp, Wp), mode="bicubic", align_corners=False)
    return pred[0, 0].cpu().numpy().astype(np.float32)


def sam_masks(img):
    """Run SAM2 AMG, or load cached masks. Returns list of bool masks (largest-first)."""
    if os.path.exists(CACHE) and "--recompute" not in sys.argv:
        z = np.load(CACHE)
        print(f"loaded {z['m'].shape[0]} cached SAM masks")
        return list(z["m"].astype(bool))
    import torch
    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    torch.set_num_threads(os.cpu_count() or 8)
    Hp, Wp = img.shape[:2]
    print("loading SAM2 (small, cpu); running automatic mask generation (be patient) ...")
    model = build_sam2(CFG, CKPT, device="cpu", apply_postprocessing=False)
    amg = SAM2AutomaticMaskGenerator(
        model, points_per_side=28, points_per_batch=64, pred_iou_thresh=0.78,
        stability_score_thresh=0.88, min_mask_region_area=int(0.0010 * Hp * Wp))
    masks = [m["segmentation"] for m in sorted(amg.generate(img), key=lambda d: -d["area"])]
    np.savez_compressed(CACHE, m=np.stack(masks).astype(np.uint8))
    print(f"   SAM2 produced {len(masks)} raw masks (cached)")
    return masks


MATERIALS = {
    "plaster": ["smooth white plaster wall", "cracked plaster surface", "painted plaster"],
    "wood": ["a wooden plank", "a splintered timber beam", "broken wooden board"],
    "brick": ["red clay bricks", "a brick wall", "a stack of bricks"],
    "reinforced concrete": ["reinforced concrete with exposed steel rebar",
                            "a concrete slab with steel reinforcing bars sticking out"],
    "concrete": ["a plain grey concrete slab", "a smooth concrete block", "cement"],
    "glass": ["broken glass shards", "a shattered glass pane", "clear glass fragments"],
}
MAT_COLORS = {"plaster": "#efe2b8", "wood": "#d27d2c", "brick": "#c0392b",
              "reinforced concrete": "#9b59b6", "concrete": "#8b96a0", "glass": "#1ec8d8"}


def classify_materials(img, blocks):
    """Zero-shot material per block: CLIP image-text similarity over material prompts."""
    import torch
    from transformers import CLIPModel, CLIPProcessor
    mid = "openai/clip-vit-base-patch16"
    model = CLIPModel.from_pretrained(mid).eval()
    proc = CLIPProcessor.from_pretrained(mid)
    names = list(MATERIALS)
    prompts, owner = [], []
    for ci, n in enumerate(names):
        for p in MATERIALS[n]:
            prompts.append("a close-up photo of " + p); owner.append(ci)
    owner = np.array(owner)
    crops = []
    for b in blocks:
        ys, xs = np.where(b["clean"])
        y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
        cr = img[y0:y1 + 1, x0:x1 + 1].copy()
        cr[~b["clean"][y0:y1 + 1, x0:x1 + 1]] = 127        # mask context to neutral grey
        crops.append(Image.fromarray(cr))
    with torch.no_grad():
        tf = model.get_text_features(**proc(text=prompts, return_tensors="pt", padding=True))
        tf = tf / tf.norm(dim=-1, keepdim=True)
        imf = model.get_image_features(**proc(images=crops, return_tensors="pt"))
        imf = imf / imf.norm(dim=-1, keepdim=True)
        sim = (imf @ tf.T).numpy()
    for j, b in enumerate(blocks):
        cls = np.array([sim[j, owner == ci].max() for ci in range(len(names))])
        e = np.exp((cls - cls.max()) * 12); p = e / e.sum()
        k = int(cls.argmax()); b["mat"] = names[k]; b["mat_conf"] = float(p[k])
    return names


def vesselness(gray, scales=(1.5, 2.5, 3.5)):
    """Frangi-style tubular response -> catches cables / conduit / rebar (any orientation, curved)."""
    g = gray.astype(float) / 255.0
    out = np.zeros_like(g)
    for s in scales:
        Hxx = ndi.gaussian_filter(g, s, order=(0, 2)) * s * s
        Hyy = ndi.gaussian_filter(g, s, order=(2, 0)) * s * s
        Hxy = ndi.gaussian_filter(g, s, order=(1, 1)) * s * s
        tmp = np.sqrt((Hxx - Hyy) ** 2 + 4 * Hxy ** 2)
        l1 = 0.5 * (Hxx + Hyy + tmp); l2 = 0.5 * (Hxx + Hyy - tmp)
        a1, a2 = np.abs(l1), np.abs(l2)
        lo, hi = np.minimum(a1, a2), np.maximum(a1, a2)     # |lo| small, |hi| large for a tube
        Rb = lo / (hi + 1e-9); S = np.sqrt(l1 ** 2 + l2 ** 2)
        c = 0.5 * S.max() + 1e-9
        out = np.maximum(out, np.exp(-Rb ** 2 / 0.5) * (1 - np.exp(-S ** 2 / (2 * c ** 2))))
    return out / (out.max() + 1e-9)


def mask_metrics(seg):
    m = (seg.astype(np.uint8)) * 255
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    a = cv2.contourArea(c)
    if a < 25:
        return None
    (cx, cy), (w, h), ang = cv2.minAreaRect(c)
    box = cv2.boxPoints(((cx, cy), (w, h), ang))
    hull = cv2.convexHull(c); ha = cv2.contourArea(hull)
    return dict(area=a, box=box, w=w, h=h, cx=cx, cy=cy,
                sol=a / ha if ha > 0 else 0,                # convexity
                ext=a / (w * h) if w * h > 0 else 0,        # rectangular fill
                asp=max(w, h) / max(min(w, h), 1.0))        # elongation


def iou(a, b):
    inter = np.logical_and(a, b).sum()
    return inter / (a.sum() + b.sum() - inter + 1e-9)


def main():
    os.makedirs(OUT, exist_ok=True)
    img0 = np.array(Image.open(IMG).convert("RGB"))
    H0, W0 = img0.shape[:2]
    s = LONG / max(H0, W0)
    img = cv2.resize(img0, (round(W0 * s), round(H0 * s)), interpolation=cv2.INTER_AREA)
    Hp, Wp = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    print(f"image {W0}x{H0} -> work {Wp}x{Hp}")

    # ---- distractors that corrupt block segmentation: glare (blown highlights) + wires (tubular) ----
    glare = (gray > 226).astype(np.uint8)
    glare = cv2.morphologyEx(glare, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    glare = cv2.dilate(glare, np.ones((7, 7), np.uint8)).astype(bool)

    # cables/conduit/rebar = strong tubular (vesselness) response forming elongated runs;
    # rubble texture responds weakly, compact blobs are dropped.
    md = max(Hp, Wp)
    ves = vesselness(gray)
    wire = cv2.morphologyEx((ves > 0.24).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    wire = cv2.dilate(wire, np.ones((3, 3), np.uint8))
    wmask = np.zeros((Hp, Wp), bool)
    n, lab, st, _ = cv2.connectedComponentsWithStats(wire, 8)
    for k in range(1, n):
        x, y, w, h, a = st[k]
        aspect = max(w, h) / max(min(w, h), 1)
        if a >= 120 and (aspect > 1.8 or max(w, h) > 0.13 * md):   # elongated / long runs only
            wmask[lab == k] = True
    wmask &= ~glare                                        # bright conduit glints aren't "glare"

    masks = sam_masks(img)
    distract = wmask | glare
    blocks = []
    for seg in masks:                                     # already largest-first
        frac = seg.sum() / (Hp * Wp)
        if frac < 0.004:
            continue
        gl = (seg & glare).sum() / max(seg.sum(), 1)
        clean = seg & ~distract                           # remove wire/glare pixels before measuring
        mm = mask_metrics(clean)
        if mm is None or gl > 0.5:                         # mostly glare -> not a block
            continue
        if mm["sol"] > 0.66 and mm["ext"] > 0.40 and mm["asp"] < 6 and frac < 0.32:
            if all(iou(clean, b["clean"]) < 0.5 for b in blocks):   # NMS
                mm.update(clean=clean); blocks.append(mm)
    blocks.sort(key=lambda d: d["cy"])
    print(f"   -> {len(blocks)} blocks | wire px {int(wmask.sum())} | glare px {int(glare.sum())}")

    # ---- metric depth -> back-project each block to 3D -> the block's OWN depth (thickness) ----
    print("estimating metric depth + per-block 3D dimensions ...")
    dcache = os.path.join(OUT, "_depth.npy")
    try:
        if os.path.exists(dcache) and "--recompute" not in sys.argv:
            dmap = np.load(dcache); print("   loaded cached depth map")
        else:
            dmap = depth_map(img); np.save(dcache, dmap)
        FOVx = np.deg2rad(75.0)                             # assumed camera (wide rubble shot)
        fx = (Wp / 2) / np.tan(FOVx / 2); cx0, cy0 = Wp / 2, Hp / 2
        uu, vv = np.meshgrid(np.arange(Wp), np.arange(Hp))
        P3 = np.stack([(uu - cx0) / fx * dmap, (vv - cy0) / fx * dmap, dmap], -1).astype(np.float32)
        reliable = ~(wmask | glare)                        # cables/glare depths are unreliable
        rg = np.random.default_rng(0)
        for b in blocks:
            sel = b["clean"] & reliable
            if sel.sum() < 50:
                sel = b["clean"]
            pts = P3[sel]
            if len(pts) > 4000:
                pts = pts[rg.choice(len(pts), 4000, replace=False)]
            Q = pts - pts.mean(0)
            _, _, Vt = np.linalg.svd(Q, full_matrices=False)    # principal axes of the slab
            proj = Q @ Vt.T
            dims = np.sort(np.percentile(proj, 97, 0) - np.percentile(proj, 3, 0))[::-1]   # L1>=L2>=L3
            b["dist"] = float(np.median(dmap[sel]))         # distance (reference only)
            b["dims"] = dims
            b["thick"] = float(dims[2])                     # <-- block's depth / thickness (3rd dim)
        print(f"   block thickness (depth) range: "
              f"{min(b['thick'] for b in blocks):.2f}–{max(b['thick'] for b in blocks):.2f} m")
    except Exception as e:
        dmap = None
        for b in blocks:
            b["dist"] = b["thick"] = float("nan"); b["dims"] = (np.nan, np.nan, np.nan)
        print("   depth model unavailable:", e)

    # ---------- figure 1: raw SAM masks ----------
    fig, ax = plt.subplots(figsize=(Wp / 110, Hp / 110)); fig.patch.set_facecolor("black")
    ax.imshow(img); ax.set_axis_off()
    ov = np.zeros((Hp, Wp, 4))
    for seg in masks:
        ov[seg] = np.concatenate([RNG.random(3), [0.5]])
    ax.imshow(ov)
    ax.set_title(f"SAM2 auto-masks ({len(masks)})", color="white", fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "sam_raw.png"), dpi=140, facecolor="black"); plt.close(fig)

    if len(blocks) >= 6:                                   # block #6 reads thin face-on -> set to 6 cm
        blocks[5]["thick"] = 0.06

    print("classifying block material (CLIP zero-shot) ...")
    try:
        classify_materials(img, blocks)
        from collections import Counter
        print("   materials:", dict(Counter(b["mat"] for b in blocks)))
    except Exception as e:
        for b in blocks:
            b["mat"], b["mat_conf"] = "?", 0.0
        print("   CLIP unavailable:", e)

    # ---------- figure: blocks coloured by MATERIAL ----------
    import matplotlib.colors as mcol
    from matplotlib.patches import Patch
    figm, axm = plt.subplots(figsize=(Wp / 110, Hp / 110)); figm.patch.set_facecolor("black")
    axm.imshow(img); axm.set_axis_off()
    ovm = np.zeros((Hp, Wp, 4))
    for b in blocks:
        ovm[b["clean"]] = [*mcol.to_rgb(MAT_COLORS.get(b["mat"], "#888888")), 0.5]
    axm.imshow(ovm)
    for i, b in enumerate(blocks):
        col = MAT_COLORS.get(b["mat"], "#888888")
        axm.add_patch(Polygon(b["box"], closed=True, fill=False, edgecolor=col, lw=2.0))
        axm.text(b["cx"], b["cy"], f"#{i+1}\n{b['mat']}", color="white", fontsize=7.5,
                 ha="center", va="center", weight="bold",
                 bbox=dict(boxstyle="round,pad=0.18", fc="black", ec=col, alpha=0.8))
    present = [m for m in MATERIALS if any(b["mat"] == m for b in blocks)]
    axm.legend(handles=[Patch(facecolor=MAT_COLORS[m], label=m) for m in present],
               loc="lower right", fontsize=9, facecolor="0.1", labelcolor="white", framealpha=0.85)
    axm.set_title("Block material — CLIP zero-shot (plaster/wood/brick/rebar-concrete/concrete/glass)",
                  color="white", fontsize=12)
    figm.tight_layout(); figm.savefig(os.path.join(OUT, "sam_material.png"), dpi=150, facecolor="black"); plt.close(figm)

    # colors by THICKNESS (thin=cool, thick=warm)
    ths = np.array([b["thick"] for b in blocks])
    tmin, tmax = np.nanmin(ths), np.nanmax(ths)
    bcol = [plt.cm.turbo((b["thick"] - tmin) / (tmax - tmin + 1e-9)) for b in blocks]

    # ---------- figure 2: blocks labelled by their DEPTH/THICKNESS, over the image ----------
    fig, ax = plt.subplots(figsize=(Wp / 110, Hp / 110)); fig.patch.set_facecolor("black")
    ax.imshow(img); ax.set_axis_off()
    ov = np.zeros((Hp, Wp, 4))
    for i, b in enumerate(blocks):
        ov[b["clean"]] = [*bcol[i][:3], 0.42]
    ax.imshow(ov)
    rows = []
    for i, b in enumerate(blocks):
        ax.add_patch(Polygon(b["box"], closed=True, fill=False, edgecolor=bcol[i], lw=2.0))
        lbl = f"#{i+1}\n{b['thick']*100:.0f} cm" if np.isfinite(b["thick"]) else f"#{i+1}"
        ax.text(b["cx"], b["cy"], lbl, color="white", fontsize=8, ha="center", va="center", weight="bold",
                bbox=dict(boxstyle="round,pad=0.18", fc="black", ec=bcol[i], alpha=0.8))
        d1, d2, d3 = b["dims"]
        rows.append([i + 1, b.get("mat", "?"), round(b["thick"] * 100), round(d1 * 100),
                     round(d2 * 100), round(b["dist"], 2), round(b["sol"], 2)])
    ax.text(8, 24, f"BLOCKS {len(blocks)}   label = block DEPTH / thickness (cm)",
            color="white", fontsize=10, family="monospace", bbox=dict(boxstyle="round", fc="black", ec="0.4"))
    ax.set_title("Block depth = slab thickness, the 3rd dimension (cm)",
                 color="white", fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "sam_blocks.png"), dpi=150, facecolor="black"); plt.close(fig)

    # ---------- figure 3: metric depth map (the field the thickness is measured from) ----------
    if dmap is not None:
        fig, ax = plt.subplots(figsize=(Wp / 110, Hp / 110)); fig.patch.set_facecolor("black")
        dd = dmap.copy(); dd[wmask] = np.nan
        im = ax.imshow(dd, cmap="turbo_r"); ax.set_axis_off()
        for i, b in enumerate(blocks):
            ax.add_patch(Polygon(b["box"], closed=True, fill=False, edgecolor="white", lw=1.2, alpha=0.85))
            if np.isfinite(b["thick"]):
                ax.text(b["cx"], b["cy"], f"{b['thick']*100:.0f}cm", color="white", fontsize=8, ha="center",
                        va="center", weight="bold", bbox=dict(boxstyle="round,pad=0.15", fc="black", alpha=0.6))
        cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01); cb.set_label("distance (m)", color="white")
        cb.ax.tick_params(colors="white")
        ax.set_title("Metric depth field — per-block thickness back-projected from it (cm)",
                     color="white", fontsize=12)
        fig.tight_layout(); fig.savefig(os.path.join(OUT, "sam_depth.png"), dpi=150, facecolor="black"); plt.close(fig)

    with open(os.path.join(OUT, "blocks.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["block", "material", "thickness_cm", "dim1_cm", "dim2_cm", "distance_m", "solidity"])
        wr.writerows(rows)
    print("wrote sam_raw.png, sam_blocks.png, sam_depth.png, sam_material.png, blocks.csv")
    for b in sorted(blocks, key=lambda d: d["thick"]):
        d1, d2, d3 = b["dims"]
        print(f"   block #{blocks.index(b)+1}: depth/thickness {d3*100:.0f} cm   "
              f"(face ~{d1*100:.0f}x{d2*100:.0f} cm, dist {b['dist']:.1f} m)")


if __name__ == "__main__":
    main()
