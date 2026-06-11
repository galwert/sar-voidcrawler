# SAR VoidCrawler — a navigation & perception brain for rubble search‑and‑rescue

Proof‑of‑concept software stack for a small crawling search‑and‑rescue (SAR) robot that worms
through the **void pockets inside collapsed buildings**. The robot itself (an inchworm‑style
crawler with no wheels — it stretches, contracts and turns on command) is built by the hardware
team; this repo is the **"brain"**: everything that maps the void, decides where to go, and makes
sense of what the sensors see.

Everything here runs **without a real robot, without a dataset, without ROS, and without sudo** —
the world is synthetic (Open3D ray‑casting) or a single still photo, and all heavy models run on
CPU from weights already on the machine.

---

## Table of contents

1. [What the system does](#1-what-the-system-does)
2. [Repository layout](#2-repository-layout)
3. [Environments & setup](#3-environments--setup)
4. [Pipelines (how to run everything)](#4-pipelines-how-to-run-everything)
   - [4.1 Core SLAM — synthetic LiDAR mapping](#41-core-slam--synthetic-lidar-mapping)
   - [4.2 Goal‑directed navigation + 360° survivor scan](#42-goal-directed-navigation--360-survivor-scan)
   - [4.3 Human‑part LiDAR segmentation](#43-human-part-lidar-segmentation)
   - [4.4 Camera + seismic inspection (real photos)](#44-camera--seismic-inspection-real-photos)
   - [4.5 Interactive operator console](#45-interactive-operator-console)
   - [4.6 SAM2 rubble‑block analysis — segment, depth, material](#46-sam2-rubble-block-analysis--segment-depth-material)
   - [4.7 Presentation assets](#47-presentation-assets)
5. [How the methods work](#5-how-the-methods-work)
6. [Honesty notes / known limits](#6-honesty-notes--known-limits)
7. [Future work](#7-future-work)

---

## 1. What the system does

A collapsed building leaves a lattice of small air pockets connected by tight gaps. A human or a
dog can't get in; a sniffer camera on a pole only sees a few metres. The VoidCrawler idea is a
robot thin enough to enter that lattice. For that to be useful it has to:

- **Build a map** of the void it's crawling through, live, from its own LiDAR (no GPS, no prior map).
- **Navigate to goals** — known interest points (last‑seen locations, sounds, gas readings) that
  may lie *off* the currently‑known map — replanning as new rubble is revealed.
- **Find and characterise survivors** — segment a person (or just a protruding hand) out of a
  LiDAR scan, and inspect them with a camera + a seismic "heartbeat" sensor.
- **Understand the rubble** — segment the debris into blocks, estimate each block's **thickness**
  and **material** (concrete / brick / wood / plaster / …) so rescuers know what they're cutting
  through, while ignoring the wires and glare that confuse naïve segmentation.

This repo demonstrates each of those as a standalone, reproducible script.

> **Key insight that makes it work:** dense rubble is the *easy* case for scan‑matching SLAM.
> Texture in every direction constrains the alignment; the degenerate cases for LiDAR SLAM are
> empty rooms and uniform corridors. The SAR scenario plays to the method's strengths.

---

## 2. Repository layout

```
hackathon/
├── README.md                  ← you are here
├── requirements.txt           ← pip deps for the core (.venv) stack
├── scripts/                   ← all generation code
│   ├── smoke_demo.py          ← core engine: world, synthetic LiDAR, ICP SLAM, occupancy map
│   ├── explore.py             ← v1 exploration animation (turbo 3D cloud + fog-of-war inset)
│   ├── explore3d.py           ← fuller-3D exploration variant
│   ├── to_mesh.py             ← point cloud → voxel-cube mesh (.ply for 3dviewer.net)
│   ├── navigate.py            ← A* goal-directed nav + 360° scan revealing a buried survivor
│   ├── segment.py             ← human-part segmentation on a 360° LiDAR scan (ground-truth)
│   ├── inspect_real.py        ← camera + seismic inspection on real Gemini rubble photos
│   ├── interest.py            ← earlier synthetic camera/seismic demo (kept for reference)
│   ├── sam_blocks.py          ← SAM2 block segmentation → thickness (depth) → material (CLIP)
│   ├── console_shots.py       ← renders presentation screenshots of the operator console
│   ├── figures.py             ← 12 presentation stills
│   └── clips.py               ← 3 short presentation MP4 clips
├── poc/
│   └── console.html           ← interactive SAR operator console (clickable; source copy)
├── Gemini_Generated_Image_*.png  ← input rubble photos used by inspect_real.py / sam_blocks.py
├── SAR_VoidCrawler (1).pptx   ← the talk deck
└── out/                       ← ALL generated media (gitignored; regenerate with the scripts)
```

`out/` is intentionally **not** committed — every image, video, mesh and model cache in it is
reproducible by running the scripts. Sub‑folders: `out/demos`, `out/navigation`, `out/segment`,
`out/sam`, `out/poc`, `out/presentation`, `out/models`.

---

## 3. Environments & setup

Two Python environments are used, because the perception models (SAM2 / Depth‑Anything / CLIP)
are heavier than the core SLAM stack.

### A. Core stack — `.venv` (pip, sudo‑free)

Drives everything in §4.1–4.5 and §4.7 (SLAM, navigation, the human‑part scan, the camera/seismic
videos, the console screenshots, the figures/clips).

```bash
cd ~/hackathon
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt      # numpy, open3d>=0.19, matplotlib, pillow
```

Animations are written to MP4 with `ffmpeg`; the scripts auto‑discover a conda `ffmpeg`
(e.g. `~/miniforge3/envs/*/bin/ffmpeg`) or one on `PATH`.

### B. Perception stack — the `thesis` conda env

Only `scripts/sam_blocks.py` (§4.6) needs this. It already has **SAM2** (+ checkpoints under
`~/thesis/sam2/checkpoints/`), **PyTorch (CPU)**, **transformers**, **OpenCV** and **SciPy**.
Run it with that interpreter directly:

```bash
/home/wertheizer/miniforge3/envs/thesis/bin/python scripts/sam_blocks.py
```

The Depth‑Anything‑V2 and CLIP weights download once from HuggingFace and are then cached; the
SAM2 masks and depth map are cached under `out/sam/` so subsequent runs take seconds.

---

## 4. Pipelines (how to run everything)

### 4.1 Core SLAM — synthetic LiDAR mapping

```bash
./.venv/bin/python scripts/smoke_demo.py          # → out/models/reconstruction.ply, out/demos/occupancy.png
./.venv/bin/python scripts/explore.py             # → out/demos/explore*.mp4/.png  (the "v1" wow shot)
./.venv/bin/python scripts/explore3d.py           # fuller-3D variant
./.venv/bin/python scripts/to_mesh.py             # reconstruction.ply → reconstruction_mesh.ply
```

`smoke_demo.py` is the engine the rest import: it builds a procedural rubble world, ray‑casts a
synthetic multi‑beam LiDAR, dead‑reckons noisy odometry, and stitches the scans with point‑to‑plane
ICP into a globally consistent cloud + 2D occupancy grid. `explore.py` renders the signature look —
a turbo (height‑coloured) 3D point cloud on black with a cyan path and a top‑down **fog‑of‑war**
inset that clears as coverage grows.

The `.ply` opens in any 3D viewer (drag‑and‑drop onto <https://3dviewer.net>); `to_mesh.py` turns
the cloud into coloured voxel cubes for a solid‑looking model.

### 4.2 Goal‑directed navigation + 360° survivor scan

```bash
./.venv/bin/python scripts/navigate.py            # → out/navigation/navigate.mp4 (+ stills)
./.venv/bin/python scripts/navigate.py --prior    # seed a partial map shared by another robot
```

The robot heads for known **hotspots** that may sit off the discovered map, planning with A* over
an occupancy grid where unknown space is optimistically traversable, and **replanning every few
steps** as rubble is revealed. It checks a hotspot off when it passes within line‑of‑sight.

The video has two acts in one clip:
1. **Navigation** — robot visits all hotspots; turbo map builds, cyan path grows, coverage ticks up.
2. **360° scan** — at the final hotspot it does a dense spin and **reveals a buried survivor**,
   segmented by body part. The person is under a collapsed slab, so only the **out‑stretched hand/
   forearm** (plus a sliver of head/feet) returns points — exactly like the reference photos. The
   occlusion is *real* ray‑casting, not faked.

Outputs also include `nav_topdown`, `nav_hero3d`, `nav_coverage`, `nav_map2d` (± `_prior`).

### 4.3 Human‑part LiDAR segmentation

```bash
./.venv/bin/python scripts/segment.py             # → out/segment/seg_3d.png, seg_person.png, scan.npz
```

A standalone, clean version of the survivor scan: a person built from **labelled parts**
(head / torso / arms / legs) lies in a rubble room; a 360° LiDAR is ray‑cast; because each part is
a separate geometry, Open3D's `cast_rays` returns **which part each beam hit** → every LiDAR point
is segmented by body part, with *perfect synthetic ground truth* and no trained model. `scan.npz`
(points + per‑point labels + scan origin) is the data a future click‑to‑inspect viewer would use.

### 4.4 Camera + seismic inspection (real photos)

```bash
./.venv/bin/python scripts/inspect_real.py --scene hand           # buried survivor, only a hand out
./.venv/bin/python scripts/inspect_real.py --scene person         # survivor lying in a void
./.venv/bin/python scripts/inspect_real.py --scene hand --clean   # bare detection: box + score, scan L/R, no HUD
```

Runs on the real **Gemini‑generated** rubble photos. The robot pans its camera left, then right,
locks onto the subject and boxes it (`HAND 0.79`). The **seismic** variant then confirms a heartbeat
(red pulse + ECG strip → "SURVIVOR CONFIRMED"). Two reads per scene: a full HUD camera‑feed look,
and a `--clean` detection‑only look used inside the operator console. Outputs land in `out/demos/`.

### 4.5 Interactive operator console

A self‑contained, clickable web page that ties the story together as a rescue‑team console.

- **Source:** `poc/console.html`
- **Runnable copy:** `out/poc/console.html` (sits next to the three videos it plays).

It plays the SLAM map feed with interest‑point pins; the final point **pulses red with a life‑sign
alert**. Click it → an inspection modal opens with the clean hand‑detection feed; a button
**activates the seismic sensor** and the status flips to "HEARTBEAT 72 BPM — SURVIVOR CONFIRMED".

To run: open `out/poc/console.html` in a browser (it needs `navigate_backup.mp4`,
`inspect_hand_camera.mp4`, `inspect_hand_seismic.mp4` beside it — produced by §4.2 and §4.4). To
regenerate presentation screenshots of it:

```bash
./.venv/bin/python scripts/console_shots.py       # → out/poc/shot1_map.png, shot2_inspect_cam.png, shot3_inspect_seismic.png
```

### 4.6 SAM2 rubble‑block analysis — segment, depth, material

```bash
/home/wertheizer/miniforge3/envs/thesis/bin/python scripts/sam_blocks.py
#   add --recompute to rebuild the cached SAM masks / depth map
```

A three‑stage perception pipeline on a single rubble photo:

1. **Segment (SAM2).** The Segment‑Anything‑2 automatic mask generator produces class‑agnostic
   masks; we filter them into slab‑like **blocks** (convex, rectangular‑fill, not wire‑thin), with
   NMS to dedupe. Wires and glare are detected separately and kept *out* of the blocks.
2. **Depth (Depth‑Anything‑V2, metric).** Each block's pixels are back‑projected to **3D** using
   the metric depth map + a pinhole camera model; an oriented box (PCA) gives the block's three
   real dimensions. The **smallest is the block's depth/thickness** — e.g. a 17 cm reinforced slab
   vs. 3 cm panels.
3. **Material (CLIP zero‑shot).** Each block crop is scored against text prompts for
   **plaster / wood / brick / reinforced‑concrete / concrete / glass**; the best match is assigned.

**Distractor handling (the "watch the wires and lights" requirement):**
- **Wires/cables** — SAM leaves them unsegmented, so a Frangi‑style **vesselness** filter finds the
  tubular network independently; those pixels are removed before measuring blocks.
- **Light/glare** — blown highlights are detected by brightness and excluded.

Outputs in `out/sam/`:

| file | what |
|---|---|
| `sam_raw.png`    | all raw SAM2 masks |
| `sam_blocks.png` | blocks coloured & labelled by **thickness (cm)** |
| `sam_depth.png`  | the metric depth field with per‑block thickness |
| `sam_material.png` | blocks coloured by **material** with a legend |
| `blocks.csv`     | `block, material, thickness_cm, dim1_cm, dim2_cm, distance_m, solidity` |

### 4.7 Presentation assets

```bash
./.venv/bin/python scripts/figures.py             # → out/presentation/*.png  (12 stills)
./.venv/bin/python scripts/clips.py               # → out/presentation/*.mp4  (3 clips)
```

---

## 5. How the methods work

**Synthetic LiDAR.** Open3D `RaycastingScene` casts a grid of rays (azimuth × elevation) against the
world mesh; hit distances become a point cloud, with range noise and random dropouts added to mimic
a real sensor. Because each scene object can be a separate geometry, the cast also returns
`geometry_ids` — which we exploit for free ground‑truth segmentation (§4.3) and occlusion (§4.2).

**SLAM / scan stitching.** "Odometry" here is the motion prior from the commanded stretch/turn (the
robot has no wheels, but the algorithmic role is identical). Each scan is predicted from that prior,
then refined against the accumulated map with **point‑to‑plane ICP** (surface normals constrain
rotation, killing yaw drift) under a **Tukey robust kernel**; a fitness guard falls back to the
prior in degenerate spots. Point‑to‑plane error linearises to the 6×6 normal equations
`(Σ aᵢaᵢᵀ)x = Σ aᵢbᵢ` with `aᵢ = [pᵢ×nᵢ ; nᵢ]`, `bᵢ = nᵢ·(qᵢ−pᵢ)`.

**Occupancy & coverage.** Rays mark free space (Bresenham to each hit) and walls (≥2 returns in the
robot's height band → occupied; a single stray point can't seal a doorway). Coverage is measured
honestly against the **geometric** true‑free area (sample the mesh → wall cells → reachable open
mask), not against the robot's own under‑observing path — which is why the numbers are ~84 %, not a
flattering 100 %.

**Goal‑directed navigation.** 8‑connected A* with a Euclidean heuristic; unknown cells are
traversable but penalised (optimistic, so off‑map goals are reachable), occupied cells are a last
resort (cost 1000 → the planner never returns "stuck"). A string‑pull (line‑of‑sight shortcutting)
straightens the path; the robot re‑picks the nearest unvisited hotspot by path cost every few steps.

**Buried‑survivor occlusion.** The survivor is built from labelled parts and placed under a slab
mesh; the LiDAR only returns points from the **exposed** parts (the hand reaching out) because the
slab blocks the rest — the same `cast_rays` that maps the room produces the partial segmentation.

**Block thickness from depth.** Pinhole back‑projection `X=(u−cx)/f·Z, Y=(v−cy)/f·Z, Z=depth`
lifts each block's pixels to 3D; SVD of the centred points gives the principal axes, and the
3rd–97th‑percentile extent along each axis gives the block's size. Two axes span the visible face;
the third is the **thickness**.

**Vesselness wire detector.** A Frangi‑style filter at several scales: for the image Hessian's
eigenvalues `(λ₁,λ₂)`, a tube has one small and one large magnitude → response
`exp(−Rb²/2β²)·(1−exp(−S²/2c²))` with `Rb=|λ_lo|/|λ_hi|`. Thresholding + keeping the elongated
connected components isolates the cable network that SAM ignores.

**Zero‑shot material.** CLIP (ViT‑B/16) embeds each masked block crop and the material text
prompts; cosine similarity (max over each class's prompts) picks the material. No material‑trained
model is needed — it's a sensible first guess, not a certified classification.

---

## 6. Honesty notes / known limits

These are deliberately called out so the talk stays credible:

- **Map accuracy is qualitative.** We claim "a consistent map, good enough to navigate", not a
  certified centimetre figure — real accuracy depends on the actual hardware LiDAR.
- **Coverage is ~84 %**, measured geometrically. The earlier "~100 %" was a denominator bug.
- **Survivor segmentation in §4.2–4.3 is ground truth from simulation** — it proves the *pipeline
  and visualisation*. On real LiDAR this is where a learned point‑cloud segmenter plugs in.
- **Metric depth is an estimate.** The *relative* near/far ordering is reliable; the *absolute*
  metres carry scale error (mono‑depth, indoor model, no scene reference). Give one known distance
  and everything rescales.
- **Block thickness is a lower bound for face‑on slabs** — a single camera can't see a block's back.
  Blocks caught at an angle (broken edge visible) give a true thickness; the rest are lower bounds.
- **Material is zero‑shot.** Clear cases (wood plank, bricks, plaster) are confident; grey slabs can
  be concrete‑vs‑plaster ambiguous. Reinforced concrete really needs an exposed‑rebar close‑up.

The recurring theme: the camera/LiDAR **close‑up** that the robot can take *in situ* is exactly what
resolves the single‑image ambiguities (thickness, rebar, material).

---

## 7. Future work

- On‑board learned segmentation for survivors and materials (replace the ground‑truth/zero‑shot
  stand‑ins) once labelled rubble data exists.
- Larger / organic mazes, frontier‑based incremental exploration, DARPA SubT‑style worlds.
- IRIS convex‑safe‑corridor + trajectory optimisation on top of the A* route.
- Fuse the close‑up sensors (camera, microphone, seismic vital‑signs) into the map as typed
  annotations the operator console can surface.
- Turn the crawler's hollow body into a **delivery conduit** — once a survivor is found, pump water
  / air / medicine through the tube that found them.
