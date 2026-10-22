from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from kawkab.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TelestrationLayer:
    id: str
    name: str = ""
    visible: bool = True
    locked: bool = False
    opacity: float = 1.0
    elements: list = field(default_factory=list)


@dataclass
class TelestrationPreset:
    name: str
    layers: list[TelestrationLayer] = field(default_factory=list)
    canvas_width: int = 1920
    canvas_height: int = 1080
    created_at: str = ""
    updated_at: str = ""


class TelestrationService:
    def __init__(self, presets_dir: str | None = None):
        self.presets_dir = Path(presets_dir or (Path.home() / ".kawkab" / "telestration_presets"))
        self.presets_dir.mkdir(parents=True, exist_ok=True)
        self._layers: dict[str, TelestrationLayer] = {}
        self._current_preset: str | None = None

    def add_layer(self, layer_id: str, name: str = "") -> str:
        try:
            if layer_id in self._layers:
                return json.dumps({"error": f"Layer {layer_id} already exists"})
            self._layers[layer_id] = TelestrationLayer(id=layer_id, name=name or layer_id)
            return json.dumps({"ok": True, "layer_id": layer_id})
        except Exception as e:
            return json.dumps({"error": str(e)})

    def remove_layer(self, layer_id: str) -> str:
        try:
            if layer_id in self._layers:
                del self._layers[layer_id]
            return json.dumps({"ok": True})
        except Exception as e:
            return json.dumps({"error": str(e)})

    def toggle_layer_visibility(self, layer_id: str) -> str:
        try:
            layer = self._layers.get(layer_id)
            if not layer:
                return json.dumps({"error": "Layer not found"})
            layer.visible = not layer.visible
            return json.dumps({"ok": True, "layer_id": layer_id, "visible": layer.visible})
        except Exception as e:
            return json.dumps({"error": str(e)})

    def set_layer_opacity(self, layer_id: str, opacity: float) -> str:
        try:
            layer = self._layers.get(layer_id)
            if not layer:
                return json.dumps({"error": "Layer not found"})
            layer.opacity = max(0.0, min(1.0, opacity))
            return json.dumps({"ok": True, "layer_id": layer_id, "opacity": layer.opacity})
        except Exception as e:
            return json.dumps({"error": str(e)})

    def get_layers(self) -> str:
        try:
            return json.dumps(
                {
                    "layers": [
                        {
                            "id": lid,
                            "name": layer.name,
                            "visible": layer.visible,
                            "locked": layer.locked,
                            "opacity": layer.opacity,
                            "elements": len(layer.elements),
                        }
                        for lid, layer in self._layers.items()
                    ]
                }
            )
        except Exception as e:
            return json.dumps({"error": str(e)})

    # ── Presets ──

    def save_preset(self, name: str, layers_json: str) -> str:
        try:
            layers_data = json.loads(layers_json)
            preset = TelestrationPreset(
                name=name,
                layers=[TelestrationLayer(**layer) for layer in layers_data],
                created_at=datetime.now().isoformat(),
                updated_at=datetime.now().isoformat(),
            )
            filepath = self.presets_dir / f"{name.replace(' ', '_')}.json"
            self._preset_to_file(preset, filepath)
            return json.dumps({"ok": True, "preset": name, "path": str(filepath)})
        except Exception as e:
            return json.dumps({"error": str(e)})

    def load_preset(self, name: str) -> str:
        try:
            filepath = self.presets_dir / f"{name.replace(' ', '_')}.json"
            if not filepath.exists():
                # Try scanning for it
                for f in self.presets_dir.glob("*.json"):
                    data = json.loads(f.read_text())
                    if data.get("name") == name:
                        self._layers = {
                            layer["id"]: TelestrationLayer(**layer)
                            for layer in data.get("layers", [])
                        }
                        self._current_preset = name
                        return json.dumps(
                            {"ok": True, "preset": name, "layers": list(self._layers.keys())}
                        )
                return json.dumps({"error": f"Preset '{name}' not found"})
            data = json.loads(filepath.read_text())
            self._layers = {
                layer["id"]: TelestrationLayer(**layer) for layer in data.get("layers", [])
            }
            self._current_preset = name
            return json.dumps({"ok": True, "preset": name, "layers": list(self._layers.keys())})
        except Exception as e:
            return json.dumps({"error": str(e)})

    def list_presets(self) -> str:
        try:
            presets = []
            for f in self.presets_dir.glob("*.json"):
                try:
                    data = json.loads(f.read_text())
                    presets.append(
                        {
                            "name": data.get("name", f.stem),
                            "layers": len(data.get("layers", [])),
                            "updated_at": data.get("updated_at", ""),
                        }
                    )
                except Exception:
                    pass
            return json.dumps({"presets": presets})
        except Exception as e:
            return json.dumps({"error": str(e)})

    def delete_preset(self, name: str) -> str:
        try:
            filepath = self.presets_dir / f"{name.replace(' ', '_')}.json"
            if filepath.exists():
                filepath.unlink()
            return json.dumps({"ok": True})
        except Exception as e:
            return json.dumps({"error": str(e)})

    # ── Export annotated video ──

    def export_annotated_video(
        self, video_path: str, layers_json: str, output_path: str = ""
    ) -> str:
        try:
            layers = json.loads(layers_json)
            if not layers:
                return json.dumps({"error": "No layers to export"})
            video = Path(video_path)
            if not video.exists():
                return json.dumps({"error": f"Video not found: {video_path}"})
            out = output_path or str(video.parent / f"{video.stem}_annotated{video.suffix}")

            # Generate FFmpeg drawtext filter from layer elements
            filters = []
            for layer in layers:
                for el in layer.get("elements", []):
                    if el.get("type") == "text":
                        text = el.get("text", "")
                        x = el.get("x", 0)
                        y = el.get("y", 0)
                        filters.append(
                            f"drawtext=text='{text}':x={x}:y={y}:fontsize=24:fontcolor=white"
                        )

            if not filters:
                # Copy video if no drawable elements
                cmd = ["ffmpeg", "-y", "-i", str(video), "-c", "copy", str(out)]
            else:
                filter_str = ",".join(filters)
                cmd = [
                    "ffmpeg",
                    "-y",
                    "-i",
                    str(video),
                    "-vf",
                    filter_str,
                    "-c:a",
                    "copy",
                    str(out),
                ]

            subprocess.run(cmd, capture_output=True, timeout=300)
            return json.dumps({"ok": True, "output": out})
        except subprocess.TimeoutExpired:
            return json.dumps({"error": "Export timed out"})
        except Exception as e:
            return json.dumps({"error": str(e)})

    def _preset_to_file(self, preset: TelestrationPreset, path: Path):
        data = {
            "name": preset.name,
            "canvas_width": preset.canvas_width,
            "canvas_height": preset.canvas_height,
            "created_at": preset.created_at,
            "updated_at": preset.updated_at,
            "layers": [
                {
                    "id": layer.id,
                    "name": layer.name,
                    "visible": layer.visible,
                    "locked": layer.locked,
                    "opacity": layer.opacity,
                    "elements": layer.elements,
                }
                for layer in preset.layers
            ],
        }
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
