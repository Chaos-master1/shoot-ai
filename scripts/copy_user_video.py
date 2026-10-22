"""Copy the user's video with a simple name."""

import os
import shutil

base = r"C:\Users\Expert Gaming\Downloads\football coach ai"
files = [f for f in os.listdir(base) if f.endswith(".mp4")]
print(f"MP4 files found: {len(files)}")
for f in files:
    print(f"  - {f}")

# Find the real match video (not real_match.mp4 or sample_football.mp4)
target = None
for f in files:
    if "real_match" not in f and "sample" not in f:
        target = f
        break

if target is None:
    print("No real match video found")
else:
    src = os.path.join(base, target)
    dst = os.path.join(base, "data", "real_sweden_tunisia.mp4")
    print(f"\nCopying: {target}")
    print(f"From: {src}")
    print(f"To:   {dst}")
    shutil.copy(src, dst)
    print(f"\nSize: {os.path.getsize(dst) / 1e6:.1f} MB")
    print("Done!")
