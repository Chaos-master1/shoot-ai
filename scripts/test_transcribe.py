"""Test faster-whisper transcription on match segment."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


async def main():
    from kawkab.services.audio_service import AudioService

    svc = AudioService(enable_transcription=True, whisper_model="base", gpu_enabled=True)
    await svc.initialize()

    video = Path(__file__).resolve().parent.parent / "France vs Sweden_match.mp4"
    print(f"Transcribing: {video.name}")
    results = await svc.transcribe_video(video)
    print(f"\nGot {len(results)} segments")
    for r in results[:20]:
        print(f"[{r['start']:.1f}s-{r['end']:.1f}s] {r['text'][:100]}")

    # Save full transcript
    out = video.parent / f"{video.stem}_transcript.txt"
    with open(out, "w", encoding="utf-8") as f:
        for r in results:
            f.write(f"[{r['start']:.1f}s-{r['end']:.1f}s] {r['text']}\n")
    print(f"\nSaved full transcript to {out}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
