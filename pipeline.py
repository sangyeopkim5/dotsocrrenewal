"""Orchestrate two-pass OCR with hierarchical PictureText merging."""
import argparse
import json
import sys
from typing import Any, Dict, List, Tuple

from adapters.dots_client import run_grounding, run_layout
from integrate_children import attach_picture_children


def collect_pictures(layout_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return all blocks with category == 'Picture'."""
    return [b for b in layout_json.get("blocks", []) if b.get("category") == "Picture"]


def process_page(
    image_path: str,
    layout_prompt: str = "prompt_layout_all_en",
    grounding_prompt: str = "prompt_grounding_ocr",
    max_pictures_per_page: int = 12,
) -> Dict[str, Any]:
    """Run layout OCR and per-picture grounding OCR for an image."""
    layout = run_layout(image_path, layout_prompt)

    pictures = collect_pictures(layout)
    if not pictures:
        meta = layout.get("meta", {})
        meta.setdefault("source", "dots")
        layout["meta"] = meta
        return layout

    per_picture_grounding: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    count = 0
    for pic in pictures:
        if count >= max_pictures_per_page:
            break
        bbox = pic.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        gjson = run_grounding(image_path, bbox, grounding_prompt)
        per_picture_grounding.append((pic, gjson))
        count += 1

    merged = attach_picture_children(layout, per_picture_grounding)
    return merged


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="Input image or rasterized PDF page")
    ap.add_argument("--layout-prompt", default="prompt_layout_all_en")
    ap.add_argument("--grounding-prompt", default="prompt_grounding_ocr")
    ap.add_argument("--max-pictures-per-page", type=int, default=12)
    args = ap.parse_args()

    out = process_page(
        image_path=args.image,
        layout_prompt=args.layout_prompt,
        grounding_prompt=args.grounding_prompt,
        max_pictures_per_page=args.max_pictures_per_page,
    )
    json.dump(out, sys.stdout, ensure_ascii=False)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
