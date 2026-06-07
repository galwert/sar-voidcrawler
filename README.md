# SAR rubble exploration — synthetic LiDAR SLAM demo (sudo-free)

Proof-of-concept for a small search-and-rescue robot that crawls the **void pockets inside
collapsed rubble**. The robot **autonomously explores every reachable compartment** (no
target — pure coverage) while a **3D map** is stitched live from its LiDAR. Rubble walls
block line-of-sight, so the map reveals compartment-by-compartment as it goes.

No real robot, no dataset, no ROS, no sudo — everything is synthetic and pip-only.

## Stack

| Concern | Tool |
|---|---|
| Environment | procedural maze of small rubble void-compartments (Open3D meshes) |
| Synthetic 3D LiDAR | Open3D `RaycastingScene` (ray-cast against the rubble mesh) |
| Exploration | DFS coverage of the compartment graph — visits every void |
| SLAM / stitching | odometry-primed point-to-plane scan-to-map ICP, Tukey robust kernel |
| 3D viz | matplotlib-3D (headless); Open3D interactive viewer (`--view`, needs a display) |

## Setup (one time)

```bash
cd ~/hackathon
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
```

## Run

```bash
# *** THE WOW SHOT *** -> out/exploration_3d.gif + out/exploration_3d.png
./.venv/bin/python scripts/explore.py
./.venv/bin/python scripts/explore.py --frames 120 --fps 15      # smoother / longer

# Quick 2D map + accuracy metrics -> out/occupancy.png, out/reconstruction.ply
./.venv/bin/python scripts/smoke_demo.py

# Live interactive 3D (spin it on stage; needs a display + GPU)
./.venv/bin/python scripts/smoke_demo.py --view
```

`scripts/explore.py` renders the 3D fly-in/orbit reconstruction (height-colored rubble,
cyan exploration path) with an inset top-down **fog-of-war** that clears to 100% coverage.

## How it works

1. **`make_compartments()`** — a random DFS maze of small, size-jittered rubble compartments
   (tumbled-debris walls, tight gaps, a few extra organic openings). Returns the mesh and an
   exploration path that visits **every** compartment.
2. **Sense** — a 16-beam 3D LiDAR is ray-cast at each pose, with range noise + dropouts.
3. **Dead-reckon** — noisy wheel odometry (the drifting baseline).
4. **Stitch** — `fused_frames()`: each scan is predicted from odometry, then refined against
   the accumulated map with point-to-plane ICP (normals constrain rotation → no yaw drift) and
   a robust kernel; a fitness guard falls back to odometry in degenerate spots.

Typical run: ~200 exploration poses, **fused RMSE ~0.15 m vs. ~0.85 m dead-reckoning** — a
crisp, globally consistent 3D map of the explored voids.

## Key lessons (saved so we don't relearn them)

- Scan matching needs **geometric texture in all directions**. Empty rooms (aperture problem)
  and uniform corridors (no along-axis constraint) are degenerate → ICP slides/blows up. Dense
  rubble is the *easy* case — the SAR scenario plays to the method's strengths.
- **point-to-plane + robust kernel** beats point-to-point (normals kill yaw drift).
- Tight passages need enough clearance or near-field occlusion wrecks frame overlap.

## Notes / future

- 3D wow is matplotlib-3D because Open3D's offscreen renderer needs a GPU/EGL we don't have
  headless. For the live talk, `--view` opens the real interactive Open3D viewer on a machine
  with a display.
- Possible next: bigger/organic mazes, frontier-style incremental exploration, DARPA SubT worlds.
