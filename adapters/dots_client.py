import json
import os
import subprocess
import tempfile
from typing import Any, Dict, List, Optional


def _run_parser(img_path: str, prompt: str, bbox: Optional[List[float]] = None) -> Dict[str, Any]:
    """Invoke dots_ocr/parser.py and return parsed layout data.

    The parser writes a JSONL file containing metadata about the run. The
    metadata includes a ``layout_info_path`` field pointing to a JSON file with
    the actual layout blocks. This helper runs the parser inside a temporary
    directory, loads the metadata, then reads the layout JSON to return a
    consolidated object of the form ``{"blocks": [...], "meta": {...}}``.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        cmd = [
            "python",
            "dots_ocr/parser.py",
            img_path,
            "--prompt",
            prompt,
            "--output",
            tmpdir,
        ]
        if bbox is not None:
            x1, y1, x2, y2 = [str(int(v)) for v in bbox]
            cmd.extend(["--bbox", x1, y1, x2, y2])
        subprocess.run(cmd, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        base = os.path.splitext(os.path.basename(img_path))[0]
        jsonl_path = os.path.join(tmpdir, f"{base}.jsonl")
        with open(jsonl_path, "r", encoding="utf-8") as f:
            lines = [json.loads(line) for line in f if line.strip()]

        meta: Dict[str, Any] = lines[0] if lines else {}
        blocks: List[Dict[str, Any]] = []
        layout_path = meta.get("layout_info_path")
        if layout_path and os.path.exists(layout_path):
            with open(layout_path, "r", encoding="utf-8") as layout_file:
                blocks = json.load(layout_file)

        return {"blocks": blocks, "meta": meta}


def run_layout(img_path: str, prompt: str) -> Dict[str, Any]:
    """Run dots.ocr once to obtain the full-page layout.

    Returns a dictionary containing ``blocks`` and ``meta``.
    """
    return _run_parser(img_path, prompt)


def run_grounding(img_path: str, bbox_xyxy: List[float], prompt: str) -> Dict[str, Any]:
    """Run dots.ocr within a bounding box to extract picture text.

    Returns a dictionary containing ``blocks`` and ``meta``.
    """
    return _run_parser(img_path, prompt, bbox=bbox_xyxy)
