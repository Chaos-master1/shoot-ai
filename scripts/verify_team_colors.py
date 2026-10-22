"""Verify team color assignment matches reality.

Sweden wears YELLOW. Tunisia wears RED. Larger cluster = home (heuristic).
So we expect:
- home_cluster should be YELLOW (Sweden)
- away_cluster should be RED (Tunisia)
"""

import asyncio
import os
import sys
import time

os.environ["PYTHONIOENCODING"] = "utf-8"
from pathlib import Path


async def main() -> int:
    import numpy as np

    from kawkab.services import CVService

    cv = CVService(model_size="l", gpu_enabled=True)
    await cv.initialize()

    # 1-min clip, frame_skip=2
    print("Loading 1-min clip...")
    t0 = time.time()
    track_data = await cv.process_video(Path("data/sweden_1min.mp4"), frame_skip=2)
    print(f"  {time.time() - t0:.1f}s")

    # Re-run color collection logic to get the actual RGB
    # This mimics what the team detection does
    import cv2

    cap = cv2.VideoCapture(str(Path("data/sweden_1min.mp4").resolve()))
    fps = cap.get(cv2.CAP_PROP_FPS)
    _ = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    valid_tracks = set(track_data.track_registry.keys())
    home_set = {tid for tid, team in track_data.player_teams.items() if team == "home"}
    away_set = {tid for tid, team in track_data.player_teams.items() if team == "away"}

    home_colors = []
    away_colors = []
    sample_every = max(1, int(fps / 2))

    fno = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if fno % (sample_every * 2) == 0:
            results = cv._model.track(frame, persist=True, conf=0.4, classes=[0], verbose=False)
            if results and len(results) > 0:
                boxes = results[0].boxes
                if boxes is not None and len(boxes) > 0:
                    for i in range(len(boxes)):
                        if boxes.id is None:
                            continue
                        tid = int(boxes.id[i].cpu().numpy())
                        if tid not in valid_tracks:
                            continue
                        bbox = boxes.xyxy[i].cpu().numpy()
                        x1, y1, x2, y2 = map(int, bbox)
                        torso = cv._extract_torso(frame, bbox)
                        if torso is None:
                            continue
                        color = cv._get_dominant_color(torso)
                        if color is None:
                            continue
                        if tid in home_set:
                            home_colors.append(color)
                        elif tid in away_set:
                            away_colors.append(color)
        fno += 1
    cap.release()

    def color_name(bgr):
        b, g, r = bgr
        if r > 150 and g > 150 and b < 100:
            return "YELLOW"
        elif r > 150 and g < 100 and b < 100:
            return "RED"
        elif r < 100 and g > 150 and b < 100:
            return "GREEN"
        elif r < 100 and g < 100 and b > 150:
            return "BLUE"
        elif r > 150 and g > 100 and b < 100:
            return "ORANGE"
        elif r < 80 and g < 80 and b < 80:
            return "BLACK"
        elif r > 200 and g > 200 and b > 200:
            return "WHITE"
        return f"RGB({r},{g},{b})"

    print(f"\nHome team (cluster 0, n={len(home_set)} players, {len(home_colors)} samples):")
    if home_colors:
        avg = np.mean(home_colors, axis=0).astype(int)
        print(f"  Avg BGR: {tuple(avg)}, color={color_name(tuple(avg))}")
        # Show distribution
        for c in home_colors[:5]:
            print(f"  Sample: {c} -> {color_name(c)}")

    print(f"\nAway team (cluster 1, n={len(away_set)} players, {len(away_colors)} samples):")
    if away_colors:
        avg = np.mean(away_colors, axis=0).astype(int)
        print(f"  Avg BGR: {tuple(avg)}, color={color_name(tuple(avg))}")
        for c in away_colors[:5]:
            print(f"  Sample: {c} -> {color_name(c)}")

    print("\nExpected: home=YELLOW (Sweden), away=RED (Tunisia)")
    if home_colors and away_colors:
        h = np.mean(home_colors, axis=0)
        a = np.mean(away_colors, axis=0)
        if h[0] > 150 and h[1] > 100 and h[2] < 100:
            print("[OK] Home is YELLOW (Sweden)")
        else:
            print(f"[WARN] Home is {color_name(tuple(h.astype(int)))} - might be wrong team")
        if a[2] > 150 and a[1] < 100 and a[0] < 100:
            print("[OK] Away is RED (Tunisia)")
        else:
            print(f"[WARN] Away is {color_name(tuple(a.astype(int)))} - might be wrong team")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
