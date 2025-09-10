import json
import os
import subprocess
import tempfile
from typing import Any, Dict, List, Optional


def _run_parser(img_path: str, prompt: str, bbox: Optional[List[float]] = None) -> Dict[str, Any]:
    """Invoke dots_ocr/parser.py and return parsed JSON output.

    The parser writes its results to a JSONL file inside the provided output
    directory. This helper runs the parser inside a temporary directory and
    loads the first JSON object from the resulting file.
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
        return lines[0] if lines else {}


def run_layout(img_path: str, prompt: str) -> Dict[str, Any]:
    """Run dots.ocr once to obtain the full-page layout."""
    return _run_parser(img_path, prompt)


def run_grounding(img_path: str, bbox_xyxy: List[float], prompt: str) -> Dict[str, Any]:
    """Run dots.ocr within a bounding box to extract picture text."""
    return _run_parser(img_path, prompt, bbox=bbox_xyxy)
