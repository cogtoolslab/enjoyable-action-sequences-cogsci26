"""Render Flappy Bird trajectory JSON files as videos and visualizations.

This module is the shared visual-output step for the stimuli pipeline. It can:
- render trajectory JSON files as MP4 videos with countdown/end-card overlays
- create static trajectory visualizations
- create animated trajectory visualizations
- compare multiple trajectories side by side
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import cv2
import matplotlib.animation as animation
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent
ASSETS_DIR = PROJECT_ROOT / "assets"
DEFAULT_VIDEO_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "videos"
FRAME_SIZE = (400, 600)
DEFAULT_FPS = 24
DEFAULT_BIRD_SPRITE = "bird-flap-1.png"

sys.path.insert(0, str(PROJECT_ROOT))

from core import config  # noqa: E402
from core.config import (  # noqa: E402
    BIRD_HEIGHT,
    BIRD_WIDTH,
    GAME_SPEED,
    GROUND_HEIGHT,
    PIPE_GAP,
    SCREEN_HEIGHT,
    SCREEN_WIDHT,
)
from recording import game_renderer  # noqa: E402


def load_trajectory_json(trajectory_path: str | Path) -> Dict[str, Any]:
    """Load a full trajectory JSON file."""
    with open(trajectory_path, "r") as f:
        return json.load(f)


def load_trajectory(trajectory_path: str | Path) -> List[Dict[str, Any]]:
    """Load the trajectory step list from a trajectory JSON file."""
    return load_trajectory_json(trajectory_path)["trajectory"]


def trajectory_step_to_frame_data(traj_step: Dict[str, Any], step_idx: int) -> Dict[str, Any]:
    """Convert one trajectory step to the frame format expected by GameRenderer."""
    bird = traj_step.get("bird", {})
    step = int(traj_step.get("step", step_idx))

    return {
        "step": step,
        "turn": step,
        "damage_mode": False,
        "bird": {
            "x": float(bird.get("x", 0)),
            "y": float(bird.get("y", 0)),
            "width": float(config.BIRD_WIDTH),
            "height": float(config.BIRD_HEIGHT),
            "speed": float(bird.get("speed", 0)),
        },
        "pipes": traj_step.get("pipes", []),
        "ground": {
            "x": 0,
            "y": config.SCREEN_HEIGHT - config.GROUND_HEIGHT,
        },
        "action_executed": int(traj_step.get("action", 0)),
        "action_intended": int(traj_step.get("action", 0)),
    }


def trajectory_to_recording(trajectory_data: Dict[str, Any]) -> Dict[str, Any]:
    """Convert trajectory JSON data to a lightweight recording dictionary."""
    trajectory_steps = trajectory_data.get("trajectory", [])
    return {
        "episode_data": [
            trajectory_step_to_frame_data(step, idx)
            for idx, step in enumerate(trajectory_steps)
        ],
        "metadata": trajectory_data.get("metadata", {}).copy(),
    }


def _frame_to_bgr(frame_image: Image.Image) -> np.ndarray:
    frame = cv2.cvtColor(np.array(frame_image), cv2.COLOR_RGB2BGR)
    return cv2.resize(frame, FRAME_SIZE)


def _darken_frame(frame: np.ndarray, alpha: float) -> np.ndarray:
    dark = frame.copy()
    dark[:] = (0, 0, 0)
    return cv2.addWeighted(frame, 1 - alpha, dark, alpha, 0)


def _overlay_logo(frame: np.ndarray, logo: Image.Image, frame_idx: int, frame_count: int) -> np.ndarray:
    """Place a gently pulsing transparent logo at the center of a BGR frame."""
    frame_width, frame_height = FRAME_SIZE
    logo_width = int(frame_width * 0.6)
    logo_height = int(logo.height * (logo_width / logo.width))

    t = frame_idx / max(frame_count, 1)
    scale_variation = 0.015 * math.exp(-0.1 * 0.8 * t) * math.cos(2 * math.pi * 0.8 * t - math.pi / 2)
    scale_factor = 1.0 + scale_variation

    scaled_width = int(logo_width * scale_factor)
    scaled_height = int(logo_height * scale_factor)
    scaled_logo = logo.resize((scaled_width, scaled_height), Image.LANCZOS)
    logo_rgba = np.array(scaled_logo)
    logo_rgb = logo_rgba[:, :, :3]
    logo_alpha = logo_rgba[:, :, 3] / 255.0
    logo_bgr = cv2.cvtColor(logo_rgb, cv2.COLOR_RGB2BGR)

    x1 = max(0, min((frame_width - scaled_width) // 2, frame_width - scaled_width))
    y1 = max(0, min((frame_height - scaled_height) // 2, frame_height - scaled_height))
    x2 = x1 + scaled_width
    y2 = y1 + scaled_height

    if y2 > frame_height or x2 > frame_width:
        return frame

    roi = frame[y1:y2, x1:x2]
    for channel in range(3):
        roi[:, :, channel] = (
            logo_alpha * logo_bgr[:, :, channel]
            + (1 - logo_alpha) * roi[:, :, channel]
        )
    frame[y1:y2, x1:x2] = roi
    return frame


def _write_logo_overlay(
    writer: cv2.VideoWriter,
    base_frame: np.ndarray,
    logo_path: Path,
    fps: int,
    background_alpha: float,
) -> None:
    try:
        logo = Image.open(logo_path).convert("RGBA")
    except Exception as error:
        print(f"Warning: Could not load {logo_path.name}: {error}")
        return

    frame_count = max(1, int(fps))
    darkened_frame = _darken_frame(base_frame, background_alpha)
    for frame_idx in range(frame_count):
        writer.write(_overlay_logo(darkened_frame.copy(), logo, frame_idx, frame_count))


def render_video_from_recording(
    recording: Dict[str, Any],
    output_path: str | Path,
    fps: int = DEFAULT_FPS,
    bird_sprite_name: str = DEFAULT_BIRD_SPRITE,
    include_overlays: bool = True,
) -> None:
    """Render a recording dictionary as an MP4."""
    episode = recording.get("episode_data", [])
    if not episode:
        print(f"Warning: Empty trajectory, skipping video generation for {output_path}")
        return

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    renderer = game_renderer.GameRenderer(bird_sprite_name=bird_sprite_name, hide_ui=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        FRAME_SIZE,
    )

    try:
        if include_overlays:
            first_frame = _frame_to_bgr(renderer.render_frame(episode[0]))
            for countdown_num in (3, 2, 1):
                _write_logo_overlay(
                    writer,
                    first_frame,
                    ASSETS_DIR / f"{countdown_num}.png",
                    fps,
                    background_alpha=0.4,
                )

        for step_data in episode:
            writer.write(_frame_to_bgr(renderer.render_frame(step_data)))

        if include_overlays:
            last_frame = _frame_to_bgr(renderer.render_frame(episode[-1]))
            _write_logo_overlay(
                writer,
                last_frame,
                ASSETS_DIR / "end.png",
                fps,
                background_alpha=0.5,
            )
    finally:
        writer.release()


def determine_bird_sprite(trajectory_data: Dict[str, Any], default: str = DEFAULT_BIRD_SPRITE) -> str:
    """Use an explicit trajectory bird sprite if one is present, otherwise default."""
    metadata = trajectory_data.get("metadata", {})
    if metadata.get("bird_variant"):
        return f"bird-upflap-{metadata['bird_variant']}.png"
    return metadata.get("bird_sprite_name", default)


def generate_videos_from_trajectories(
    trajectory_paths: List[str],
    output_dir: str,
    fps: int = DEFAULT_FPS,
    bird_sprite: Optional[str] = None,
    filename_suffix: Optional[str] = None,
    include_overlays: bool = True,
) -> List[str]:
    """Generate videos for trajectory JSON paths and return the output paths."""
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    generated_videos = []

    print(f"Generating videos from {len(trajectory_paths)} trajectories...")
    print(f"Output directory: {output_dir_path}")

    for idx, trajectory_path in enumerate(trajectory_paths, 1):
        trajectory_path = Path(trajectory_path)
        print(f"\n[{idx}/{len(trajectory_paths)}] Processing: {trajectory_path.name}")

        try:
            trajectory_data = load_trajectory_json(trajectory_path)
            recording = trajectory_to_recording(trajectory_data)
            sprite_name = bird_sprite or determine_bird_sprite(trajectory_data)

            suffix = f"_{filename_suffix}" if filename_suffix else ""
            output_path = output_dir_path / f"{trajectory_path.stem}{suffix}.mp4"

            print(f"  Rendering video with sprite: {sprite_name}")
            render_video_from_recording(
                recording,
                output_path,
                fps=fps,
                bird_sprite_name=sprite_name,
                include_overlays=include_overlays,
            )

            generated_videos.append(str(output_path))
            print(f"  Generated: {output_path}")
        except Exception as error:
            print(f"  Error processing {trajectory_path}: {error}")
            continue

    print(f"\nGenerated {len(generated_videos)}/{len(trajectory_paths)} videos")
    return generated_videos


def visualize_frame(
    ax: Any,
    trajectory: List[Dict[str, Any]],
    frame_idx: int,
    show_bird_path: bool = False,
    path_length: int = 20,
) -> None:
    """Visualize a single trajectory frame on a Matplotlib axis."""
    ax.clear()

    step = trajectory[frame_idx]
    bird = step["bird"]
    pipes = step["pipes"]

    ax.set_xlim(-50, SCREEN_WIDHT + 50)
    ax.set_ylim(0, SCREEN_HEIGHT)
    ax.set_aspect("equal")
    ax.invert_yaxis()

    ground_rect = patches.Rectangle(
        (0, SCREEN_HEIGHT - GROUND_HEIGHT),
        SCREEN_WIDHT,
        GROUND_HEIGHT,
        linewidth=0,
        edgecolor="none",
        facecolor="#DEB887",
        alpha=0.8,
    )
    ax.add_patch(ground_rect)

    for pipe in pipes:
        color = "#2E8B57" if pipe["inverted"] else "#228B22"
        pipe_rect = patches.Rectangle(
            (pipe["x"], pipe["y"]),
            pipe["width"],
            pipe["height"],
            linewidth=0.5,
            edgecolor="darkgreen",
            facecolor=color,
            alpha=1.0,
        )
        ax.add_patch(pipe_rect)

        if not pipe["inverted"]:
            gap_rect = patches.Rectangle(
                (pipe["x"], pipe["y"] - PIPE_GAP),
                pipe["width"],
                PIPE_GAP,
                linewidth=0.5,
                edgecolor="yellow",
                facecolor="lightgreen",
                linestyle="-",
                alpha=0.5,
            )
            ax.add_patch(gap_rect)

    if show_bird_path and frame_idx > 0:
        start_idx = max(0, frame_idx - path_length)
        path_x = [trajectory[i]["bird"]["x"] + BIRD_WIDTH / 2 for i in range(start_idx, frame_idx + 1)]
        path_y = [trajectory[i]["bird"]["y"] + BIRD_HEIGHT / 2 for i in range(start_idx, frame_idx + 1)]
        ax.plot(path_x, path_y, "b-", alpha=0.3, linewidth=1)

    bird_rect = patches.Rectangle(
        (bird["x"], bird["y"]),
        BIRD_WIDTH,
        BIRD_HEIGHT,
        linewidth=0.5,
        edgecolor="darkblue",
        facecolor="#4169E1",
        alpha=0.8,
        zorder=10,
    )
    ax.add_patch(bird_rect)

    bird_center_x = bird["x"] + BIRD_WIDTH / 2
    bird_center_y = bird["y"] + BIRD_HEIGHT / 2
    ax.plot(bird_center_x, bird_center_y, "ro", markersize=3, zorder=11)

    is_colliding = False
    bird_y = bird["y"]
    bird_bottom = bird_y + BIRD_HEIGHT
    if bird_y < 0 or bird_bottom > SCREEN_HEIGHT - GROUND_HEIGHT:
        is_colliding = True

    for pipe in pipes:
        if pipe["inverted"]:
            continue
        if pipe["x"] - BIRD_WIDTH < bird["x"] < pipe["x"] + pipe["width"]:
            if pipe["y"] < bird_bottom or pipe["y"] - PIPE_GAP > bird_y:
                is_colliding = True
                break

    collision_status = "COLLISION" if is_colliding else "Clear"
    info_text = f"Step: {step['step']}  |  Score: {step['score']}  |  {collision_status}\n"
    info_text += (
        f"Bird Y: {bird['y']:.1f} to {bird['y'] + BIRD_HEIGHT:.1f}  |  "
        f"Speed: {bird['speed']:.1f}  |  Action: {'JUMP' if step['action'] == 1 else 'NONE'}"
    )
    ax.text(10, 30, info_text, fontsize=10, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

    ax.set_title("Flappy Bird Trajectory Visualization", fontsize=14, fontweight="bold")
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)


def create_animation(
    trajectory_path: str | Path,
    output_path: str | Path | None = None,
    interval: int = 50,
    show_path: bool = True,
) -> None:
    """Create an animated visualization of a trajectory."""
    print(f"Loading trajectory: {trajectory_path}")
    trajectory = load_trajectory(trajectory_path)
    print(f"Loaded {len(trajectory)} frames")

    fig, ax = plt.subplots(figsize=(12, 10))

    def animate(frame_idx: int):
        visualize_frame(ax, trajectory, frame_idx, show_bird_path=show_path)
        return ax.patches + ax.lines

    anim = animation.FuncAnimation(
        fig,
        animate,
        frames=len(trajectory),
        interval=interval,
        blit=False,
        repeat=True,
    )

    if output_path:
        output_path = str(output_path)
        print(f"Saving animation to: {output_path}")
        if output_path.endswith(".gif"):
            anim.save(output_path, writer="pillow", fps=20)
        elif output_path.endswith(".mp4"):
            anim.save(output_path, writer="ffmpeg", fps=20)
        else:
            raise ValueError("Animation output must end with .gif or .mp4")
        print("Animation saved.")
    else:
        print("Showing animation. Close the window to exit.")
        plt.show()


def _collect_world_space_pipes(trajectory: List[Dict[str, Any]]) -> Dict[tuple, Dict[str, Any]]:
    pipes_world_space = {}
    for step_idx, step in enumerate(trajectory):
        step_offset = step_idx * GAME_SPEED
        for pipe in step["pipes"]:
            pipe_world_x = pipe["x"] + step_offset
            pipe_key = (round(pipe_world_x / 10) * 10, pipe["y"], pipe["inverted"])
            if pipe_key not in pipes_world_space:
                pipes_world_space[pipe_key] = {
                    "x": pipe_world_x,
                    "y": pipe["y"],
                    "width": pipe["width"],
                    "height": pipe["height"],
                    "inverted": pipe["inverted"],
                }
    return pipes_world_space


def create_static_visualization(
    trajectory_path: str | Path,
    output_path: str | Path | None = None,
    sample_rate: int = 10,
) -> None:
    """Create a static visualization showing the full bird path."""
    trajectory = load_trajectory(trajectory_path)

    pipes_world_space = _collect_world_space_pipes(trajectory)
    bird_start_x = 70
    bird_end_x = 70 + len(trajectory) * GAME_SPEED
    min_x = bird_start_x - 100
    max_x = bird_end_x + 100
    x_range = max_x - min_x

    fig_width = max(16, x_range / 50)
    fig_height = 10
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    ax.set_xlim(min_x, max_x)
    ax.set_ylim(0, SCREEN_HEIGHT)
    ax.set_aspect("equal")
    ax.invert_yaxis()

    ground_rect = patches.Rectangle(
        (min_x, SCREEN_HEIGHT - GROUND_HEIGHT),
        x_range,
        GROUND_HEIGHT,
        linewidth=0,
        edgecolor="none",
        facecolor="#DEB887",
        alpha=0.8,
    )
    ax.add_patch(ground_rect)

    for pipe in pipes_world_space.values():
        color = "#2E8B57" if pipe["inverted"] else "#228B22"
        pipe_rect = patches.Rectangle(
            (pipe["x"], pipe["y"]),
            pipe["width"],
            pipe["height"],
            linewidth=1,
            edgecolor="darkgreen",
            facecolor=color,
            alpha=0.3,
        )
        ax.add_patch(pipe_rect)

    for step in trajectory[::max(1, sample_rate)]:
        bird_world_x = 70 + step["step"] * GAME_SPEED
        bird_y = step["bird"]["y"]
        hitbox_rect = patches.Rectangle(
            (bird_world_x, bird_y),
            BIRD_WIDTH,
            BIRD_HEIGHT,
            linewidth=0,
            edgecolor="none",
            facecolor="#acc9e6",
            alpha=1.0,
            zorder=4,
        )
        ax.add_patch(hitbox_rect)

    start_x = 70 + BIRD_WIDTH / 2
    end_x = 70 + trajectory[-1]["step"] * GAME_SPEED + BIRD_WIDTH / 2

    ax.plot(start_x, trajectory[0]["bird"]["y"] + BIRD_HEIGHT / 2, "go", markersize=10, label="Start", zorder=10)
    ax.plot(end_x, trajectory[-1]["bird"]["y"] + BIRD_HEIGHT / 2, "ro", markersize=10, label="End", zorder=10)

    total_steps = len(trajectory)
    final_score = trajectory[-1]["score"]
    ax.set_title(f"Flappy Bird Trajectory - {total_steps} steps, Score: {final_score}",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)

    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def create_side_by_side_comparison(
    trajectory_paths: Sequence[str | Path],
    output_path: str | Path | None = None,
    sample_rate: int = 10,
) -> None:
    """Create a side-by-side comparison of multiple trajectories."""
    num_trajectories = len(trajectory_paths)
    max_x_range = 0
    for trajectory_path in trajectory_paths:
        trajectory = load_trajectory(trajectory_path)
        bird_end_x = 70 + len(trajectory) * GAME_SPEED
        max_x_range = max(max_x_range, bird_end_x + 200)

    fig_width = max(8 * num_trajectories, max_x_range / 80 * num_trajectories)
    fig, axes = plt.subplots(1, num_trajectories, figsize=(fig_width, 10))
    if num_trajectories == 1:
        axes = [axes]

    for idx, trajectory_path in enumerate(trajectory_paths):
        print(f"Loading trajectory {idx + 1}: {trajectory_path}")
        trajectory = load_trajectory(trajectory_path)
        ax = axes[idx]

        bird_start_x = 70
        bird_end_x = 70 + len(trajectory) * GAME_SPEED
        min_x = bird_start_x - 100
        max_x = bird_end_x + 100
        x_range = max_x - min_x

        ax.set_xlim(min_x, max_x)
        ax.set_ylim(0, SCREEN_HEIGHT)
        ax.set_aspect("equal")
        ax.invert_yaxis()

        ground_rect = patches.Rectangle(
            (min_x, SCREEN_HEIGHT - GROUND_HEIGHT),
            x_range,
            GROUND_HEIGHT,
            linewidth=0,
            edgecolor="none",
            facecolor="#DEB887",
            alpha=0.8,
        )
        ax.add_patch(ground_rect)

        for pipe in _collect_world_space_pipes(trajectory).values():
            color = "#2E8B57" if pipe["inverted"] else "#228B22"
            pipe_rect = patches.Rectangle(
                (pipe["x"], pipe["y"]),
                pipe["width"],
                pipe["height"],
                linewidth=1,
                edgecolor="darkgreen",
                facecolor=color,
                alpha=0.6,
            )
            ax.add_patch(pipe_rect)

        path_x = [70 + step["step"] * GAME_SPEED + BIRD_WIDTH / 2 for step in trajectory[::max(1, sample_rate)]]
        path_y = [step["bird"]["y"] + BIRD_HEIGHT / 2 for step in trajectory[::max(1, sample_rate)]]
        ax.plot(path_x, path_y, "b-", alpha=0.5, linewidth=2)

        end_x = 70 + trajectory[-1]["step"] * GAME_SPEED + BIRD_WIDTH / 2
        ax.plot(end_x, trajectory[-1]["bird"]["y"] + BIRD_HEIGHT / 2, "ro", markersize=10, zorder=10)

        filename = Path(trajectory_path).stem
        final_score = trajectory[-1]["score"]
        ax.set_title(f"{filename}\nScore: {final_score}", fontsize=10, fontweight="bold")
        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")
        ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)

    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Saving comparison to: {output_path}")
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("Comparison saved.")
    else:
        plt.show()


def find_trajectory_by_map_number(map_number: int, outputs_dir: str | Path | None = None) -> Path | None:
    """Find a trajectory file by map number."""
    outputs_dir = Path(outputs_dir) if outputs_dir is not None else PROJECT_ROOT / "outputs"
    matches = list(outputs_dir.glob(f"*map_{map_number}.json"))
    if matches:
        return matches[0]

    for traj_file in outputs_dir.glob("*.json"):
        if f"map_{map_number}" in traj_file.name:
            return traj_file
    return None


def visualize_map_by_number(
    map_number: int,
    outputs_dir: str | Path | None = None,
    output_path: str | Path | None = None,
    sample_rate: int = 10,
    output_name_suffix: str | None = None,
    output_name_prefix: str | None = None,
) -> Path | None:
    """Visualize a trajectory by map number."""
    outputs_dir = Path(outputs_dir) if outputs_dir is not None else PROJECT_ROOT / "outputs"
    trajectory_path = find_trajectory_by_map_number(map_number, outputs_dir)

    if trajectory_path is None:
        print(f"Map {map_number} not found in {outputs_dir}")
        return None
    if not trajectory_path.exists():
        print(f"Trajectory file not found: {trajectory_path}")
        return None

    if output_path is None:
        viz_dir = outputs_dir / "visualizations"
        viz_dir.mkdir(exist_ok=True)
        prefix = f"{output_name_prefix}_" if output_name_prefix else ""
        suffix = f"_{output_name_suffix}" if output_name_suffix else ""
        output_path = viz_dir / f"{prefix}map_{map_number}{suffix}.png"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        create_static_visualization(
            str(trajectory_path),
            str(output_path),
            sample_rate=sample_rate,
        )
        print(f"Visualization saved to: {output_path}")
        return Path(output_path)
    except Exception as error:
        print(f"Error creating visualization: {error}")
        return None


def visualize_maps_by_numbers(
    map_numbers: Sequence[int],
    outputs_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    sample_rate: int = 10,
) -> List[tuple[int, Path]]:
    """Visualize multiple maps by their map numbers."""
    outputs_dir = Path(outputs_dir) if outputs_dir is not None else PROJECT_ROOT / "outputs"
    output_dir = Path(output_dir) if output_dir is not None else outputs_dir / "visualizations"

    results = []
    print(f"Visualizing {len(map_numbers)} maps...\n")

    for i, map_num in enumerate(map_numbers, 1):
        print(f"[{i}/{len(map_numbers)}] ", end="")
        output_path = output_dir / f"map_{map_num}_visualization.png"
        result = visualize_map_by_number(
            map_num,
            outputs_dir=outputs_dir,
            output_path=output_path,
            sample_rate=sample_rate,
        )
        if result:
            results.append((map_num, result))
        print()

    print(f"\nSuccessfully visualized {len(results)}/{len(map_numbers)} maps")
    return results


def _default_example_trajectory() -> Path:
    trajectories = sorted((PROJECT_ROOT / "initial_trajectories").glob("*trajectory*.json"))
    if not trajectories:
        raise FileNotFoundError("No seed trajectory JSON files found in initial_trajectories/")
    return trajectories[0]


def _add_video_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("video", help="Render trajectory JSON files as MP4 videos.")
    parser.add_argument("trajectories", nargs="*", help="Trajectory JSON files to render.")
    parser.add_argument("--output-dir", default=str(DEFAULT_VIDEO_OUTPUT_DIR), help="Directory for generated videos.")
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS, help="Frames per second.")
    parser.add_argument("--bird-sprite", default=DEFAULT_BIRD_SPRITE, help="Bird sprite filename from assets/.")
    parser.add_argument("--no-overlays", action="store_true", help="Skip countdown and final overlay frames.")


def _add_visualize_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("visualize", help="Create static, animated, or comparison visualizations.")
    parser.add_argument("trajectory", help="Path to trajectory JSON file.")
    parser.add_argument("--mode", choices=["animate", "static", "compare"], default="static", help="Visualization mode.")
    parser.add_argument("--output", help="Output file path.")
    parser.add_argument("--interval", type=int, default=50, help="Milliseconds between animation frames.")
    parser.add_argument("--sample-rate", type=int, default=10, help="Sample rate for static/comparison visualizations.")
    parser.add_argument("--no-path", action="store_true", help="Disable bird path trail in animation.")
    parser.add_argument("--compare", nargs="+", help="Additional trajectories for comparison mode.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render Flappy Bird trajectory visual outputs.")
    subparsers = parser.add_subparsers(dest="command")
    _add_video_parser(subparsers)
    _add_visualize_parser(subparsers)

    args = parser.parse_args()

    if args.command == "video":
        trajectory_paths = args.trajectories or [str(_default_example_trajectory())]
        generate_videos_from_trajectories(
            trajectory_paths=trajectory_paths,
            output_dir=args.output_dir,
            fps=args.fps,
            bird_sprite=args.bird_sprite,
            filename_suffix=None,
            include_overlays=not args.no_overlays,
        )
        return

    if args.command == "visualize":
        if args.mode == "animate":
            create_animation(
                args.trajectory,
                output_path=args.output,
                interval=args.interval,
                show_path=not args.no_path,
            )
        elif args.mode == "static":
            create_static_visualization(
                args.trajectory,
                output_path=args.output,
                sample_rate=args.sample_rate,
            )
        elif args.mode == "compare":
            trajectories = [args.trajectory]
            if args.compare:
                trajectories.extend(args.compare)
            create_side_by_side_comparison(
                trajectories,
                output_path=args.output,
                sample_rate=args.sample_rate,
            )
        return

    parser.print_help()


if __name__ == "__main__":
    main()
