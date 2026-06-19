#!/usr/bin/env python3
# show_pcds.py
import argparse
import os
import sys
import re
import copy
import numpy as np
import open3d as o3d
import open3d.visualization.rendering as rendering


NAMED = {
    "r": (1, 0, 0), "red": (1, 0, 0),
    "g": (0, 1, 0), "green": (0, 1, 0),
    "b": (0, 0, 1), "blue": (0, 0, 1),
    "c": (0, 1, 1), "cyan": (0, 1, 1),
    "m": (1, 0, 1), "magenta": (1, 0, 1),
    "y": (1, 1, 0), "yellow": (1, 1, 0),
    "k": (0, 0, 0), "black": (0, 0, 0),
    "w": (1, 1, 1), "white": (1, 1, 1),
    "orange": (1.0, 0.5, 0.0),
    "purple": (0.5, 0.0, 0.8),
    "grey": (0.5, 0.5, 0.5), "gray": (0.5, 0.5, 0.5),
}

HEX_RE = re.compile(r"^#?[0-9a-fA-F]{3}([0-9a-fA-F]{3})?$")

def parse_color(token: str):
    """Return (r,g,b) in [0,1] if token is a color, else None."""
    if token is None:
        return None
    s = token.strip().lower()

    # Named colors
    if s in NAMED:
        return NAMED[s]

    # Hex colors: #RRGGBB or RRGGBB or #RGB
    if HEX_RE.match(s):
        h = s[1:] if s.startswith("#") else s
        if len(h) == 3:
            h = "".join(ch * 2 for ch in h)
        r = int(h[0:2], 16) / 255.0
        g = int(h[2:4], 16) / 255.0
        b = int(h[4:6], 16) / 255.0
        return (r, g, b)

    # CSV numbers: "255,0,0" or "1,0,0"
    if "," in s:
        parts = s.split(",")
        if len(parts) == 3:
            try:
                vals = [float(p) for p in parts]
                if any(v > 1.0 for v in vals):
                    vals = [v / 255.0 for v in vals]
                vals = [min(max(v, 0.0), 1.0) for v in vals]
                return tuple(vals)
            except ValueError:
                return None
    return None

def parse_entries(tokens):
    """
    Parse sequence like: FILE [COLOR] FILE [COLOR] ...
    Returns list of (path, color_or_None).
    """
    out = []
    i = 0
    while i < len(tokens):
        path = tokens[i]
        i += 1
        if not os.path.exists(path):
            print(f"[warn] '{path}' not found -> skipping.", file=sys.stderr)
            # If the "path" was actually a stray color, just move on
            continue
        color = None
        if i < len(tokens):
            maybe_color = parse_color(tokens[i])
            if maybe_color is not None:
                color = maybe_color
                i += 1
        out.append((path, color))
    return out

def main():
    ap = argparse.ArgumentParser(
        description="Show N PCDs in the same Open3D window. "
                    "Pass as FILE [COLOR] pairs; COLOR is optional."
    )
    ap.add_argument("items", nargs="*", help="Sequence: file1 [color1] file2 [color2] ...")
    ap.add_argument("--voxel", type=float, default=0.0, help="Voxel downsample size (0 to disable).")
    ap.add_argument("--point-size", type=float, default=1.0, help="Point size in viewer.")
    ap.add_argument("--bg", default="1,1,1",
                    help="Background color (name, #hex, or 'R,G,B'). Default: 0,0,0.")
    ap.add_argument("--frame", type=float, default=10.0,
                    help="Add a coordinate frame of this size (0 disables).")
    ap.add_argument("--center", action="store_true",
                    help="Center each cloud at origin (subtract centroid).")
    ap.add_argument("--normalize", action="store_true",
                    help="Normalize each cloud by its bbox diagonal (scale to ~1).")
    args = ap.parse_args()

    bg = parse_color(args.bg) or (0.0, 0.0, 0.0)

    entries = parse_entries(args.items)
    if not entries:
        print("No valid inputs.\n\nExamples:\n"
              "  python show_pcds.py scan1.pcd scan2.pcd\n"
              "  python show_pcds.py scan1.pcd red scan2.pcd 0,255,0 scan3.pcd #00aaff\n",
              file=sys.stderr)
        sys.exit(1)

    geoms = []
    legend = []

    for path, color in entries:
        pcd = o3d.io.read_point_cloud(path)
        if pcd.is_empty():
            print(f"[warn] '{path}' has zero points -> skipping.", file=sys.stderr)
            continue

        if args.voxel > 0.0:
            pcd = pcd.voxel_down_sample(args.voxel)

        if args.center:
            pcd.translate(-pcd.get_center())

        if args.normalize:
            aabb = pcd.get_axis_aligned_bounding_box()
            diag = np.linalg.norm(aabb.get_extent())
            if diag > 0:
                pcd.scale(1.0 / diag, center=(0, 0, 0))

        if color is not None:
            p = copy.deepcopy(pcd)
            p.paint_uniform_color(color)
            geoms.append(p)
            legend.append((path, tuple(int(round(c * 255)) for c in color), "uniform"))
        else:
            geoms.append(pcd)
            legend.append((path, None, "original"))

    if args.frame > 0:
        geoms.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=args.frame))

    # Visualizer with custom background + point size (works across O3D versions)
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="PCD Compare", width=1280, height=800)
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0)
    def mat_mesh():
        m = rendering.MaterialRecord()
        m.shader = "defaultLit"
        m.base_color = (0.7, 0.7, 0.9, 1.0)
        m.base_roughness = 0.8
        return m
    vis.add_geometry(axis)
    for g in geoms:
        vis.add_geometry(g)
    opt = vis.get_render_option()
    opt.point_size = float(args.point_size)
    if hasattr(opt, "background_color"):
        opt.background_color = np.asarray(bg, dtype=np.float64)
    vis.run()
    vis.destroy_window()

    # Print legend to terminal
    print("\nLegend:")
    for path, rgb255, mode in legend:
        if rgb255 is None:
            print(f"  {path} : original colors")
        else:
            print(f"  {path} : RGB{rgb255}")

if __name__ == "__main__":
    main()
