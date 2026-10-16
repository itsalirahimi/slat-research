"""Visualize mapping PCDs, occupancy voxels, meshes, and diagnostic PLYs."""

import argparse
import os
import sys
from pathlib import Path

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + "/../lib")

from mapping.geometry_viewer import show_files


def main():
    parser = argparse.ArgumentParser(
        description="Visualize PCD/PLY mapping output with Open3D"
    )
    parser.add_argument("paths", nargs="+", type=Path, help="One or more .pcd/.ply files")
    voxel_group = parser.add_mutually_exclusive_group()
    voxel_group.add_argument(
        "--voxels",
        dest="render_voxels",
        action="store_true",
        help="Render point centers as voxel cubes",
    )
    voxel_group.add_argument(
        "--points",
        dest="render_voxels",
        action="store_false",
        help="Render an occupancy PCD as points instead of auto-detected cubes",
    )
    parser.set_defaults(render_voxels=None)
    parser.add_argument(
        "--voxel-size",
        type=float,
        help="Cube size in meters (normally read from occupancy JSON)",
    )
    parser.add_argument("--no-axes", action="store_true", help="Hide XYZ axes")
    args = parser.parse_args()
    show_files(
        args.paths,
        render_voxels=args.render_voxels,
        voxel_size=args.voxel_size,
        show_axes=not args.no_axes,
    )


if __name__ == "__main__":
    main()

