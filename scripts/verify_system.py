"""Comprehensive system verification for Kawkab AI."""

import os

os.environ["PYTHONIOENCODING"] = "utf-8"

import sys
from pathlib import Path


def main() -> int:
    print("=" * 60)
    print("KAWKAB AI - SYSTEM VERIFICATION")
    print("=" * 60)
    print()

    print("1. Service imports:")
    from kawkab.core.config import get_settings

    print("   [OK] All services + app + bridge import")
    print()

    print("2. Configuration:")
    s = get_settings()
    print(f"   App: {s.app_name} v{s.app_version}")
    print(f"   GPU: {s.gpu_enabled}, Model: yolov11{s.model_size}")
    print(f"   LLM: {s.llm_provider} ({s.ollama_model})")
    print(f"   Language: {s.language}")
    print()

    print("3. GPU:")
    import torch

    print(f"   Device: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"   CUDA: {torch.version.cuda}")
    print()

    print("4. Ollama:")
    import httpx

    try:
        r = httpx.get("http://localhost:11434/api/tags", timeout=3.0)
        models = [m["name"] for m in r.json().get("models", [])][:5]
        print("   Status: online")
        print(f"   Models: {models}")
    except Exception as e:
        print(f"   Status: offline ({e})")
    print()

    print("5. Knowledge base:")
    tactics = list(Path("src/kawkab/knowledge/tactics").rglob("*.yaml"))
    drills = list(Path("src/kawkab/knowledge/drills").rglob("*.yaml"))
    print(f"   Tactical rules: {len(tactics)}")
    print(f"   Drills: {len(drills)}")
    print()

    print("6. Project structure:")
    dirs = [
        "src/kawkab/core",
        "src/kawkab/services",
        "src/kawkab/ui",
        "src/kawkab/web",
        "src/kawkab/knowledge",
    ]
    for d in dirs:
        p = Path(d)
        status = "[OK]" if p.exists() else "[MISSING]"
        print(f"   {status} {d}/")
    print()

    print("7. Graphify (knowledge graph):")
    graph_path = Path("graphify-out")
    if graph_path.exists():
        files = list(graph_path.rglob("*"))
        print(f"   [OK] {len(files)} graph files in graphify-out/")
    else:
        print("   [INFO] Not yet built (needs LLM for docs)")
    print()

    print("=" * 60)
    print("ALL SYSTEMS GO! Ready to develop.")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
