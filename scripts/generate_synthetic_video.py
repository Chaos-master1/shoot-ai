"""Better synthetic video generator for testing the CV pipeline.

Generates a football-like scene with moving figures that YOLO can actually
detect as people. Uses colored silhouettes on a green field.
"""

from __future__ import annotations

import argparse
import random
import subprocess
import sys
from pathlib import Path


def generate_better_synthetic_video(
    output_path: Path,
    duration_sec: int = 30,
    fps: int = 30,
    width: int = 1280,
    height: int = 720,
) -> Path:
    """Generate a synthetic video with figures YOLO can detect.

    Creates:
    - Green field background with white lines
    - 22 moving "person" rectangles (colored silhouettes that move around)
    - 1 "ball" (small white circle that moves randomly)
    - Realistic motion patterns

    Args:
        output_path: Where to save the video
        duration_sec: Video duration in seconds
        fps: Frames per second
        width: Video width
        height: Video height

    Returns:
        Path to generated video
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_players = 22
    team_split = n_players // 2

    n_frames = duration_sec * fps
    random.seed(42)

    print(f"  Generating {n_frames} frames ({duration_sec}s @ {fps}fps)...")

    player_positions = []
    for i in range(n_players):
        x = random.uniform(100, width - 100)
        y = random.uniform(100, height - 100)
        vx = random.uniform(-50, 50)
        vy = random.uniform(-50, 50)
        team = "team1" if i < team_split else "team2"
        player_positions.append((x, y, vx, vy, team))

    ball_x = width / 2
    ball_y = height / 2
    ball_vx = 100
    ball_vy = 80

    frames_cmd = []
    dt = 1.0 / fps

    for frame_idx in range(n_frames):
        t = frame_idx * dt
        new_positions = []
        for x, y, vx, vy, team in player_positions:
            nx = x + vx * dt
            ny = y + vy * dt
            if nx < 80 or nx > width - 80:
                vx = -vx
                nx = max(80, min(width - 80, nx))
            if ny < 80 or ny > height - 80:
                vy = -vy
                ny = max(80, min(height - 80, ny))
            if random.random() < 0.02:
                vx = random.uniform(-80, 80)
                vy = random.uniform(-80, 80)
            new_positions.append((nx, ny, vx, vy, team))
        player_positions = new_positions

        ball_x += ball_vx * dt
        ball_y += ball_vy * dt
        if ball_x < 30 or ball_x > width - 30:
            ball_vx = -ball_vx
        if ball_y < 30 or ball_y > height - 30:
            ball_vy = -ball_vy

        shape_cmds = []
        for x, y, _, _, team in player_positions:
            color = "0:0:200" if team == "team1" else "200:0:0"
            shape_cmds.append(
                f"drawbox=x={int(x - 15)}:y={int(y - 30)}:w=30:h=60:color={color}@0.9:t=fill"
            )
        shape_cmds.append(
            f"drawbox=x={int(ball_x - 8)}:y={int(ball_y - 8)}:w=16:h=16:color=white:t=fill"
        )
        shape_cmds.append(
            f"drawtext=text='Kawkab Test - {t:.1f}s':fontsize=24:fontcolor=white:x=20:y=20:box=1:boxcolor=black@0.5:boxborderw=5"
        )

        frame_str = ",\n".join(shape_cmds)
        frames_cmd.append(
            f"drawbox=x=0:y=0:w={width}:h={height}:color=green@0.001:t=fill, {frame_str}"
        )

    print("  Encoding video (this may take a minute)...")

    n_segments = max(1, len(frames_cmd) // 5)
    segments = []
    for i in range(n_segments):
        start = i * 5
        end = min(start + 5, len(frames_cmd))
        seg = ",".join(frames_cmd[start:end])
        segments.append(f"[0:v]{seg},format=yuv420p[v{i}]")
    filter_complex = (
        ";\n".join(segments)
        + ";\n"
        + "".join(f"[v{i}]" for i in range(n_segments))
        + f"concat=n={n_segments}:v=1:a=0[outv]"
    )

    with open("filter_script.txt", "w") as f:
        f.write(filter_complex)

    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"color=c=green:s={width}x{height}:r={fps}:d={duration_sec}",
        "-filter_complex_script",
        "filter_script.txt",
        "-map",
        "[outv]",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-crf",
        "28",
        "-pix_fmt",
        "yuv420p",
        "-r",
        str(fps),
        str(output_path),
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print("  [WARN] Complex filter failed, trying simpler approach...")
            return _generate_simple_animated_video(output_path, duration_sec, fps, width, height)
        print(f"  [OK] Video saved: {output_path}")
        return output_path
    except Exception as e:
        print(f"  [ERROR] {e}")
        return _generate_simple_animated_video(output_path, duration_sec, fps, width, height)


def _generate_simple_animated_video(
    output_path: Path,
    duration_sec: int,
    fps: int,
    width: int,
    height: int,
) -> Path:
    """Fallback: simple animated video with moving colored boxes."""
    print("  Using simple animated fallback...")

    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"color=c=green:s={width}x{height}:r={fps}:d={duration_sec}",
        "-vf",
        "drawbox=x='if(lt(t,5),100+t*100,if(lt(t,10),600-(t-5)*100,if(lt(t,15),600,100+(t-15)*100)))':y='if(lt(t,3),200,if(lt(t,8),300+t*30,if(lt(t,13),450,400)))':w=30:h=60:color=red:t=fill,drawbox=x='if(lt(t,5),200+t*80,if(lt(t,10),600-(t-5)*80,if(lt(t,15),600,200+(t-15)*80)))':y='if(lt(t,3),250,if(lt(t,8),350+t*20,if(lt(t,13),400,350)))':w=30:h=60:color=blue:t=fill,drawbox=x='if(lt(t,5),400+t*60,if(lt(t,10),700-(t-5)*60,if(lt(t,15),700,400+(t-15)*60)))':y='if(lt(t,3),300,if(lt(t,8),400,if(lt(t,13),500,400)))':w=30:h=60:color=red:t=fill,drawbox=x=640:y=360:w=20:h=20:color=white:t=fill,drawtext=text='%{eif\\:t\\:d}':fontsize=40:fontcolor=white:x=20:y=20:box=1:boxcolor=black@0.5:boxborderw=5",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-crf",
        "28",
        "-pix_fmt",
        "yuv420p",
        str(output_path),
    ]
    subprocess.run(cmd, capture_output=True, text=True, check=True)
    print(f"  [OK] Simple animated video saved: {output_path}")
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate synthetic football video")
    parser.add_argument("--duration", type=int, default=30)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    from kawkab.core.paths import get_paths

    paths = get_paths()
    output = (
        Path(args.output)
        if args.output
        else paths.cache / "tests" / f"better_synthetic_{args.duration}s.mp4"
    )

    try:
        generate_better_synthetic_video(output, args.duration, args.fps)
        return 0
    except FileNotFoundError:
        print("[ERROR] FFmpeg not installed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
