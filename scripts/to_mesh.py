#!/usr/bin/env python3
"""
Convert the point-cloud map (out/reconstruction.ply) into a COLORED MESH so it opens in
mesh-only web viewers like 3dviewer.net (which reject bare point clouds -- "no faces").

Each occupied voxel becomes a small cube, colored by height (turbo). Output has real faces +
vertex colors, so it looks like the animation when opened anywhere.

Run:  ./.venv/bin/python scripts/to_mesh.py [voxel]
Out:  out/reconstruction_mesh.ply
"""
import os
import sys
import numpy as np
import open3d as o3d
import matplotlib.cm as cm

OUT = os.path.join(os.path.dirname(__file__), "..", "out", "models")


def main():
    v = float(sys.argv[1]) if len(sys.argv) > 1 else 0.14
    src = os.path.join(OUT, "reconstruction.ply")
    pcd = o3d.io.read_point_cloud(src)
    pts = np.asarray(pcd.points)
    if len(pts) == 0:
        raise SystemExit(f"no points in {src} -- run scripts/smoke_demo.py first")

    centers = (np.unique(np.floor(pts / v).astype(np.int64), axis=0) + 0.5) * v   # occupied voxels
    M = len(centers)
    off = v / 2 * np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                            [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], float)
    tri = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4],
                    [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]], int)
    verts = (centers[:, None, :] + off[None, :, :]).reshape(-1, 3)
    tris = (tri[None, :, :] + (np.arange(M) * 8)[:, None, None]).reshape(-1, 3)

    z = verts[:, 2]
    colors = cm.turbo((z - z.min()) / (np.ptp(z) + 1e-9))[:, :3]

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(tris)
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    mesh.compute_vertex_normals()

    out = os.path.join(OUT, "reconstruction_mesh.ply")
    o3d.io.write_triangle_mesh(out, mesh)
    print(f"{M} voxel cubes -> {out}  ({len(verts):,} verts, {len(tris):,} tris)")


if __name__ == "__main__":
    main()
