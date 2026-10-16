#!/usr/bin/env python3

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import open3d as o3d

try:
    import tkinter as tk
    from tkinter import ttk
except ImportError:
    print("ERROR: tkinter is required for the control panel.")
    print("Ubuntu/Debian:")
    print("  sudo apt install python3-tk")
    sys.exit(1)


# ============================================================
# Colors
# ============================================================

NAMED_COLORS = {
    "red":     (1.0, 0.0, 0.0),
    "green":   (0.0, 1.0, 0.0),
    "blue":    (0.0, 0.0, 1.0),
    "yellow":  (1.0, 1.0, 0.0),
    "cyan":    (0.0, 1.0, 1.0),
    "magenta": (1.0, 0.0, 1.0),
    "white":   (1.0, 1.0, 1.0),
    "black":   (0.0, 0.0, 0.0),
    "gray":    (0.5, 0.5, 0.5),
    "grey":    (0.5, 0.5, 0.5),
    "orange":  (1.0, 0.5, 0.0),
    "purple":  (0.5, 0.0, 0.5),
    "pink":    (1.0, 0.4, 0.7),
    "brown":   (0.5, 0.25, 0.0),
}


def parse_color(value):
    value = value.strip().lower()

    if value in NAMED_COLORS:
        return np.array(NAMED_COLORS[value], dtype=float)

    # Hex
    hex_value = value.lstrip("#")

    if len(hex_value) == 6:
        try:
            return np.array([
                int(hex_value[0:2], 16) / 255.0,
                int(hex_value[2:4], 16) / 255.0,
                int(hex_value[4:6], 16) / 255.0,
            ], dtype=float)
        except ValueError:
            pass

    # R,G,B
    if "," in value:
        try:
            rgb = np.array(
                [float(x.strip()) for x in value.split(",")],
                dtype=float,
            )

            if len(rgb) != 3:
                raise ValueError

            if np.max(rgb) > 1.0:
                rgb /= 255.0

            return np.clip(rgb, 0.0, 1.0)

        except ValueError:
            pass

    raise ValueError(f"Unknown color: {value}")


def is_color(value):
    try:
        parse_color(value)
        return True
    except ValueError:
        return False


# ============================================================
# Parse:
#
#   a.pcd red b.pcd c.pcd green
#
# ============================================================

def parse_pcd_items(items):
    result = []
    i = 0

    while i < len(items):
        filename = items[i]

        if not filename.lower().endswith(".pcd"):
            raise ValueError(
                f"Expected .pcd file, got: {filename}"
            )

        color = None

        if i + 1 < len(items):
            if is_color(items[i + 1]):
                color = parse_color(items[i + 1])
                i += 1

        result.append((filename, color))
        i += 1

    return result


# ============================================================
# Vector / camera math
# ============================================================

def normalize(v):
    v = np.asarray(v, dtype=float)

    n = np.linalg.norm(v)

    if n < 1e-12:
        return v

    return v / n


def rotate_vector(v, axis, angle_deg):
    """
    Rodrigues rotation.
    """

    v = np.asarray(v, dtype=float)
    axis = normalize(axis)

    a = math.radians(angle_deg)

    return (
        v * math.cos(a)
        + np.cross(axis, v) * math.sin(a)
        + axis * np.dot(axis, v) * (1.0 - math.cos(a))
    )


def camera_from_angles(
    target,
    distance,
    yaw,
    pitch,
    roll,
):
    """
    Camera convention:

        yaw = 0
            camera located on +X side

        yaw = 90
            camera located on +Y side

        pitch = 0
            horizontal

        pitch > 0
            camera above target

        roll
            rotation around viewing direction
    """

    target = np.asarray(target, dtype=float)

    pitch = np.clip(
        float(pitch),
        -89.9,
        89.9,
    )

    yaw_rad = math.radians(yaw)
    pitch_rad = math.radians(pitch)

    # Target -> camera
    offset_direction = np.array([
        math.cos(pitch_rad) * math.cos(yaw_rad),
        math.cos(pitch_rad) * math.sin(yaw_rad),
        math.sin(pitch_rad),
    ])

    eye = (
        target
        + float(distance) * offset_direction
    )

    # Camera -> target
    forward = normalize(
        target - eye
    )

    # Use +Z as normal world-up.
    world_up = np.array([
        0.0,
        0.0,
        1.0,
    ])

    # Project world-up onto camera plane.
    up = (
        world_up
        - np.dot(world_up, forward) * forward
    )

    # Handle nearly vertical view.
    if np.linalg.norm(up) < 1e-8:
        world_up = np.array([
            0.0,
            1.0,
            0.0,
        ])

        up = (
            world_up
            - np.dot(world_up, forward) * forward
        )

    up = normalize(up)

    # Apply roll.
    up = rotate_vector(
        up,
        forward,
        roll,
    )

    up = normalize(up)

    return eye, forward, up


def build_extrinsic(
    eye,
    forward,
    up,
):
    """
    Build Open3D/OpenCV style world -> camera extrinsic.

    Camera axes:
        +X = screen right
        +Y = screen down
        +Z = forward
    """

    forward = normalize(forward)
    up = normalize(up)

    right = normalize(
        np.cross(forward, up)
    )

    # Re-orthogonalize up.
    up = normalize(
        np.cross(right, forward)
    )

    down = -up

    R = np.vstack([
        right,
        down,
        forward,
    ])

    t = -R @ np.asarray(
        eye,
        dtype=float,
    )

    extrinsic = np.eye(4)

    extrinsic[:3, :3] = R
    extrinsic[:3, 3] = t

    return extrinsic


def read_camera(view_control, target):
    """
    Read current legacy Open3D camera.

    This continues to work after mouse rotation.
    """

    params = (
        view_control
        .convert_to_pinhole_camera_parameters()
    )

    extrinsic = np.asarray(
        params.extrinsic,
        dtype=float,
    )

    R = extrinsic[:3, :3]
    t = extrinsic[:3, 3]

    # Camera position in world coordinates.
    eye = -R.T @ t

    # OpenCV/Open3D camera axes.
    forward = normalize(R[2, :])
    up = normalize(-R[1, :])

    target = np.asarray(
        target,
        dtype=float,
    )

    offset = eye - target

    distance = np.linalg.norm(offset)

    if distance < 1e-10:
        yaw = 0.0
        pitch = 0.0
    else:
        yaw = math.degrees(
            math.atan2(
                offset[1],
                offset[0],
            )
        )

        horizontal = math.sqrt(
            offset[0] ** 2
            + offset[1] ** 2
        )

        pitch = math.degrees(
            math.atan2(
                offset[2],
                horizontal,
            )
        )

    # --------------------------------------------------------
    # Calculate roll
    # --------------------------------------------------------

    world_up = np.array([
        0.0,
        0.0,
        1.0,
    ])

    reference_up = (
        world_up
        - np.dot(world_up, forward) * forward
    )

    if np.linalg.norm(reference_up) < 1e-8:
        world_up = np.array([
            0.0,
            1.0,
            0.0,
        ])

        reference_up = (
            world_up
            - np.dot(world_up, forward) * forward
        )

    reference_up = normalize(reference_up)

    actual_up = (
        up
        - np.dot(up, forward) * forward
    )

    actual_up = normalize(actual_up)

    sin_roll = np.dot(
        forward,
        np.cross(
            reference_up,
            actual_up,
        ),
    )

    cos_roll = np.dot(
        reference_up,
        actual_up,
    )

    roll = math.degrees(
        math.atan2(
            sin_roll,
            cos_roll,
        )
    )

    # --------------------------------------------------------
    # Calculate vertical FOV
    # --------------------------------------------------------

    K = np.asarray(
        params.intrinsic.intrinsic_matrix
    )

    height = params.intrinsic.height

    fy = K[1, 1]

    if fy > 0 and height > 0:
        fov = math.degrees(
            2.0
            * math.atan(
                height / (2.0 * fy)
            )
        )
    else:
        fov = 0.0

    return {
        "yaw": yaw,
        "pitch": pitch,
        "roll": roll,
        "distance": distance,
        "eye": eye,
        "forward": forward,
        "up": up,
        "fov": fov,
    }


def apply_camera(
    view_control,
    target,
    distance,
    yaw,
    pitch,
    roll,
):
    """
    Apply camera while staying inside the legacy renderer.
    """

    params = (
        view_control
        .convert_to_pinhole_camera_parameters()
    )

    eye, forward, up = camera_from_angles(
        target,
        distance,
        yaw,
        pitch,
        roll,
    )

    params.extrinsic = build_extrinsic(
        eye,
        forward,
        up,
    )

    view_control.convert_from_pinhole_camera_parameters(
        params,
        allow_arbitrary=True,
    )


# ============================================================
# Capture helpers
# ============================================================

def next_capture_path(directory):
    directory = Path(directory)

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    for i in range(1, 100000):
        path = (
            directory
            / f"capture_{i:04d}.png"
        )

        if not path.exists():
            return path

    raise RuntimeError(
        "Too many capture files."
    )


# ============================================================
# Control panel
# ============================================================

class CameraPanel:

    def __init__(
        self,
        vis,
        target,
        capture_dir,
        panel_x,
        panel_y,
    ):
        self.vis = vis
        self.view_control = (
            vis.get_view_control()
        )

        self.target = np.asarray(
            target,
            dtype=float,
        )

        self.capture_dir = capture_dir

        self.root = tk.Tk()

        self.root.title(
            "PCD Camera Control"
        )

        self.root.geometry(
            f"380x650+{panel_x}+{panel_y}"
        )

        self.root.resizable(
            False,
            True,
        )

        self.closed = False

        self.root.protocol(
            "WM_DELETE_WINDOW",
            self.close_panel,
        )

        # ----------------------------------------------------
        # Live view
        # ----------------------------------------------------

        live_frame = ttk.LabelFrame(
            self.root,
            text="Live Camera",
            padding=10,
        )

        live_frame.pack(
            fill="x",
            padx=10,
            pady=10,
        )

        self.live_var = tk.StringVar()

        live_label = ttk.Label(
            live_frame,
            textvariable=self.live_var,
            font=("TkFixedFont", 10),
            justify="left",
        )

        live_label.pack(
            anchor="w"
        )

        # ----------------------------------------------------
        # Camera fields
        # ----------------------------------------------------

        camera_frame = ttk.LabelFrame(
            self.root,
            text="Camera",
            padding=10,
        )

        camera_frame.pack(
            fill="x",
            padx=10,
            pady=5,
        )

        self.yaw_var = tk.StringVar()
        self.pitch_var = tk.StringVar()
        self.roll_var = tk.StringVar()
        self.distance_var = tk.StringVar()

        self.make_field(
            camera_frame,
            "Yaw",
            self.yaw_var,
            0,
        )

        self.make_field(
            camera_frame,
            "Pitch",
            self.pitch_var,
            1,
        )

        self.make_field(
            camera_frame,
            "Roll",
            self.roll_var,
            2,
        )

        self.make_field(
            camera_frame,
            "Distance",
            self.distance_var,
            3,
        )

        # ----------------------------------------------------
        # Target
        # ----------------------------------------------------

        target_frame = ttk.LabelFrame(
            self.root,
            text="Orbit Target",
            padding=10,
        )

        target_frame.pack(
            fill="x",
            padx=10,
            pady=5,
        )

        self.target_x_var = tk.StringVar(
            value=f"{self.target[0]:.6f}"
        )

        self.target_y_var = tk.StringVar(
            value=f"{self.target[1]:.6f}"
        )

        self.target_z_var = tk.StringVar(
            value=f"{self.target[2]:.6f}"
        )

        self.make_field(
            target_frame,
            "X",
            self.target_x_var,
            0,
        )

        self.make_field(
            target_frame,
            "Y",
            self.target_y_var,
            1,
        )

        self.make_field(
            target_frame,
            "Z",
            self.target_z_var,
            2,
        )

        # ----------------------------------------------------
        # Buttons
        # ----------------------------------------------------

        button_frame = ttk.Frame(
            self.root,
            padding=10,
        )

        button_frame.pack(
            fill="x"
        )

        apply_button = ttk.Button(
            button_frame,
            text="Apply Camera",
            command=self.apply_from_fields,
        )

        apply_button.pack(
            fill="x",
            pady=3,
        )

        current_button = ttk.Button(
            button_frame,
            text="Use Current View",
            command=self.copy_current_view,
        )

        current_button.pack(
            fill="x",
            pady=3,
        )

        capture_button = ttk.Button(
            button_frame,
            text="Capture Image",
            command=self.capture,
        )

        capture_button.pack(
            fill="x",
            pady=8,
        )

        # ----------------------------------------------------
        # Status
        # ----------------------------------------------------

        self.status_var = tk.StringVar(
            value=(
                f"Capture folder:\n"
                f"{self.capture_dir}"
            )
        )

        status = ttk.Label(
            self.root,
            textvariable=self.status_var,
            justify="left",
        )

        status.pack(
            fill="x",
            padx=12,
            pady=5,
        )

        # ----------------------------------------------------
        # Help
        # ----------------------------------------------------

        help_text = (
            "Open3D mouse controls\n"
            "\n"
            "Left drag        Rotate\n"
            "Wheel            Zoom\n"
            "Shift + drag     Roll / pan depending on control\n"
            "\n"
            "You can also press P inside the Open3D window\n"
            "for Open3D's built-in screenshot."
        )

        help_label = ttk.Label(
            self.root,
            text=help_text,
            justify="left",
        )

        help_label.pack(
            fill="x",
            padx=12,
            pady=10,
        )

        self.last_update = 0.0

        self.copy_current_view()

    def make_field(
        self,
        parent,
        label,
        variable,
        row,
    ):
        ttk.Label(
            parent,
            text=label,
            width=10,
        ).grid(
            row=row,
            column=0,
            sticky="w",
            padx=4,
            pady=3,
        )

        entry = ttk.Entry(
            parent,
            textvariable=variable,
            width=20,
        )

        entry.grid(
            row=row,
            column=1,
            sticky="ew",
            padx=4,
            pady=3,
        )

        parent.columnconfigure(
            1,
            weight=1,
        )

    def current_target(self):
        return np.array([
            float(self.target_x_var.get()),
            float(self.target_y_var.get()),
            float(self.target_z_var.get()),
        ])

    def apply_from_fields(self):
        try:
            target = self.current_target()

            yaw = float(
                self.yaw_var.get()
            )

            pitch = float(
                self.pitch_var.get()
            )

            roll = float(
                self.roll_var.get()
            )

            distance = float(
                self.distance_var.get()
            )

            if distance <= 0:
                raise ValueError(
                    "distance must be > 0"
                )

            self.target = target

            apply_camera(
                self.view_control,
                target,
                distance,
                yaw,
                pitch,
                roll,
            )

            self.vis.poll_events()
            self.vis.update_renderer()

            self.status_var.set(
                "Camera applied."
            )

        except Exception as e:
            self.status_var.set(
                f"ERROR:\n{e}"
            )

    def copy_current_view(self):
        try:
            state = read_camera(
                self.view_control,
                self.target,
            )

            self.yaw_var.set(
                f"{state['yaw']:.6f}"
            )

            self.pitch_var.set(
                f"{state['pitch']:.6f}"
            )

            self.roll_var.set(
                f"{state['roll']:.6f}"
            )

            self.distance_var.set(
                f"{state['distance']:.6f}"
            )

            self.status_var.set(
                "Copied current camera values."
            )

        except Exception as e:
            self.status_var.set(
                f"ERROR:\n{e}"
            )

    def capture(self):
        try:
            path = next_capture_path(
                self.capture_dir
            )

            self.vis.poll_events()
            self.vis.update_renderer()

            self.vis.capture_screen_image(
                str(path),
                do_render=True,
            )

            state = read_camera(
                self.view_control,
                self.target,
            )

            self.status_var.set(
                f"Saved:\n{path}"
            )

            print()
            print(
                f"Captured: {path}"
            )

            print(
                "Camera:",
                f"--angles "
                f"{state['yaw']:.6f} "
                f"{state['pitch']:.6f} "
                f"{state['roll']:.6f} "
                f"--distance "
                f"{state['distance']:.6f} "
                f"--target "
                f"{self.target[0]:.6f} "
                f"{self.target[1]:.6f} "
                f"{self.target[2]:.6f}"
            )

        except Exception as e:
            self.status_var.set(
                f"ERROR:\n{e}"
            )

    def update_live(self):
        now = time.monotonic()

        # ~10 Hz is enough for camera display.
        if now - self.last_update < 0.1:
            return

        self.last_update = now

        try:
            state = read_camera(
                self.view_control,
                self.target,
            )

            eye = state["eye"]

            self.live_var.set(
                f"Yaw      {state['yaw']:10.3f}°\n"
                f"Pitch    {state['pitch']:10.3f}°\n"
                f"Roll     {state['roll']:10.3f}°\n"
                f"Distance {state['distance']:10.4f}\n"
                f"FOV      {state['fov']:10.3f}°\n"
                f"\n"
                f"Eye X    {eye[0]:10.4f}\n"
                f"Eye Y    {eye[1]:10.4f}\n"
                f"Eye Z    {eye[2]:10.4f}"
            )

        except Exception:
            pass

    def update(self):
        if self.closed:
            return

        try:
            self.update_live()

            self.root.update_idletasks()
            self.root.update()

        except tk.TclError:
            self.closed = True

    def close_panel(self):
        try:
            self.root.destroy()
        except Exception:
            pass

        self.closed = True


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Legacy Open3D PCD viewer with camera controls "
            "and screenshot capture."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Examples
--------

Original behavior:

  python3 viewer.py a.pcd red b.pcd c.pcd green


With origin:

  python3 viewer.py --origin a.pcd red b.pcd


Fixed camera:

  python3 viewer.py \\
      --angles 45 20 0 \\
      a.pcd


Fixed camera and distance:

  python3 viewer.py \\
      --angles 45 20 0 \\
      --distance 10 \\
      a.pcd


Fixed target:

  python3 viewer.py \\
      --angles 45 20 0 \\
      --distance 15 \\
      --target 0 0 0 \\
      a.pcd


Black background:

  python3 viewer.py \\
      --bg black \\
      --origin \\
      a.pcd


Larger points:

  python3 viewer.py \\
      --point-size 4 \\
      a.pcd
"""
    )

    # --------------------------------------------------------
    # Rendering
    # --------------------------------------------------------

    parser.add_argument(
        "--origin",
        action="store_true",
        help="Add XYZ origin at (0,0,0).",
    )

    parser.add_argument(
        "--origin-size",
        type=float,
        default=None,
        help="Origin axis size.",
    )

    parser.add_argument(
        "--bg",
        default="white",
        help="Background color. Default: white",
    )

    parser.add_argument(
        "--point-size",
        type=float,
        default=2.0,
        help="Point size. Default: 2",
    )

    # --------------------------------------------------------
    # Camera
    # --------------------------------------------------------

    parser.add_argument(
        "--angles",
        nargs=3,
        type=float,
        metavar=(
            "YAW",
            "PITCH",
            "ROLL",
        ),
        default=None,
        help="Camera yaw pitch roll in degrees.",
    )

    parser.add_argument(
        "--yaw",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--pitch",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--roll",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--distance",
        type=float,
        default=None,
        help="Camera distance from target.",
    )

    parser.add_argument(
        "--target",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        default=None,
        help="Camera target. Default: scene center.",
    )

    # --------------------------------------------------------
    # Window/capture
    # --------------------------------------------------------

    parser.add_argument(
        "--capture-dir",
        default="captures",
        help="Screenshot output directory.",
    )

    parser.add_argument(
        "--width",
        type=int,
        default=1280,
    )

    parser.add_argument(
        "--height",
        type=int,
        default=720,
    )

    # --------------------------------------------------------

    parser.add_argument(
        "items",
        nargs="+",
        help="PCD files optionally followed by colors.",
    )

    if hasattr(
        parser,
        "parse_intermixed_args",
    ):
        args = (
            parser.parse_intermixed_args()
        )
    else:
        args = parser.parse_args()

    # --------------------------------------------------------
    # Parse colors / files
    # --------------------------------------------------------

    try:
        bg_color = parse_color(
            args.bg
        )

        pcd_items = parse_pcd_items(
            args.items
        )

    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    geometries = []
    cloud_geometries = []

    mins = []
    maxs = []

    # --------------------------------------------------------
    # Load clouds
    # --------------------------------------------------------

    for filename, color in pcd_items:
        path = Path(filename)

        if not path.exists():
            print(
                f"Error: file does not exist: "
                f"{filename}"
            )
            sys.exit(1)

        print(f"Loading: {filename}")

        pcd = o3d.io.read_point_cloud(
            str(path)
        )

        n = len(pcd.points)

        if n == 0:
            print(
                "  WARNING: no points"
            )
            continue

        print(
            f"  points: {n:,}"
        )

        if color is not None:
            pcd.paint_uniform_color(
                color
            )

            print(
                "  color override:",
                np.round(color, 3),
            )

        else:
            if pcd.has_colors():
                print(
                    "  color: original"
                )
            else:
                print(
                    "  color: no color information"
                )

        geometries.append(pcd)
        cloud_geometries.append(pcd)

        bbox = (
            pcd.get_axis_aligned_bounding_box()
        )

        mins.append(
            bbox.get_min_bound()
        )

        maxs.append(
            bbox.get_max_bound()
        )

    if not cloud_geometries:
        print(
            "Error: no valid point clouds."
        )
        sys.exit(1)

    # --------------------------------------------------------
    # Scene dimensions
    # --------------------------------------------------------

    global_min = np.min(
        np.vstack(mins),
        axis=0,
    )

    global_max = np.max(
        np.vstack(maxs),
        axis=0,
    )

    scene_center = (
        global_min + global_max
    ) / 2.0

    scene_size = np.linalg.norm(
        global_max - global_min
    )

    if scene_size < 1e-8:
        scene_size = 1.0

    # --------------------------------------------------------
    # Origin
    # --------------------------------------------------------

    if args.origin:
        origin_size = (
            args.origin_size
            if args.origin_size is not None
            else scene_size * 0.1
        )

        origin = (
            o3d.geometry.TriangleMesh
            .create_coordinate_frame(
                size=origin_size,
                origin=[0, 0, 0],
            )
        )

        geometries.append(
            origin
        )

    # --------------------------------------------------------
    # Create EXACT legacy Open3D visualizer
    # --------------------------------------------------------

    vis = o3d.visualization.Visualizer()

    window_left = 40
    window_top = 50

    success = vis.create_window(
        window_name="Open3D PCD Viewer",
        width=args.width,
        height=args.height,
        left=window_left,
        top=window_top,
    )

    if not success:
        print(
            "Error: could not create "
            "Open3D window."
        )
        sys.exit(1)

    for geometry in geometries:
        vis.add_geometry(
            geometry
        )

    # --------------------------------------------------------
    # SAME render options as original viewer
    # --------------------------------------------------------

    render_option = (
        vis.get_render_option()
    )

    render_option.background_color = (
        bg_color
    )

    render_option.point_size = (
        args.point_size
    )

    # --------------------------------------------------------
    # Camera initialization
    # --------------------------------------------------------

    view_control = (
        vis.get_view_control()
    )

    target = np.array(
        args.target
        if args.target is not None
        else scene_center,
        dtype=float,
    )

    # Force one render first so Open3D initializes camera.
    vis.poll_events()
    vis.update_renderer()

    initial_state = read_camera(
        view_control,
        target,
    )

    # --------------------------------------------------------
    # Resolve CLI camera values
    # --------------------------------------------------------

    cli_yaw = args.yaw
    cli_pitch = args.pitch
    cli_roll = args.roll

    if args.angles is not None:
        if cli_yaw is None:
            cli_yaw = args.angles[0]

        if cli_pitch is None:
            cli_pitch = args.angles[1]

        if cli_roll is None:
            cli_roll = args.angles[2]

    camera_requested = any([
        cli_yaw is not None,
        cli_pitch is not None,
        cli_roll is not None,
        args.distance is not None,
        args.target is not None,
    ])

    if camera_requested:
        yaw = (
            initial_state["yaw"]
            if cli_yaw is None
            else cli_yaw
        )

        pitch = (
            initial_state["pitch"]
            if cli_pitch is None
            else cli_pitch
        )

        roll = (
            initial_state["roll"]
            if cli_roll is None
            else cli_roll
        )

        distance = (
            initial_state["distance"]
            if args.distance is None
            else args.distance
        )

        if distance <= 0:
            print(
                "Error: --distance must be > 0"
            )
            sys.exit(1)

        apply_camera(
            view_control,
            target,
            distance,
            yaw,
            pitch,
            roll,
        )

        vis.poll_events()
        vis.update_renderer()

    # --------------------------------------------------------
    # Control panel
    # --------------------------------------------------------

    panel_x = (
        window_left
        + args.width
        + 20
    )

    panel_y = window_top

    panel = CameraPanel(
        vis,
        target,
        args.capture_dir,
        panel_x,
        panel_y,
    )

    print()
    print("Viewer controls:")
    print("  mouse          = normal Open3D controls")
    print("  Capture Image  = control panel")
    print("  P              = Open3D built-in screenshot")
    print("  Q / Esc        = close Open3D window")
    print()

    # --------------------------------------------------------
    # Manual event loop
    #
    # Important:
    # This keeps the ORIGINAL Open3D legacy visualizer while
    # also allowing the Tk control window to stay responsive.
    # --------------------------------------------------------

    try:
        while True:

            alive = vis.poll_events()

            if not alive:
                break

            vis.update_renderer()

            panel.update()

            # Avoid busy-looping at 100% CPU.
            time.sleep(0.005)

    except KeyboardInterrupt:
        pass

    finally:
        if not panel.closed:
            panel.close_panel()

        vis.destroy_window()


if __name__ == "__main__":
    main()