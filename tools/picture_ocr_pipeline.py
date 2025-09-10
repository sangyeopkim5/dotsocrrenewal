import os
import json
import argparse
from typing import List, Dict, Tuple

from dots_ocr.parser import DotsOCRParser
from dots_ocr.utils.image_utils import fetch_image
from dots_ocr.utils.doc_utils import load_images_from_pdf
from dots_ocr.utils.layout_utils import post_process_output
from dots_ocr.utils.consts import MIN_PIXELS, MAX_PIXELS


def bbox_contains(parent: List[int], child: List[int]) -> bool:
    """Return True if `child` bbox is fully inside `parent` bbox."""
    return (
        child[0] >= parent[0]
        and child[1] >= parent[1]
        and child[2] <= parent[2]
        and child[3] <= parent[3]
    )


def extract_picture_text(parser: DotsOCRParser, origin_image, bbox: List[int]):
    """Run dots.ocr on a bounding box to extract texts inside it."""
    min_pixels = parser.min_pixels or MIN_PIXELS
    max_pixels = parser.max_pixels or MAX_PIXELS
    image = fetch_image(origin_image, min_pixels=min_pixels, max_pixels=max_pixels)
    prompt = parser.get_prompt(
        "prompt_grounding_ocr",
        bbox,
        origin_image,
        image,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    if parser.use_hf:
        response = parser._inference_with_hf(image, prompt)
    else:
        response = parser._inference_with_vllm(image, prompt)
    cells, filtered = post_process_output(
        response,
        "prompt_grounding_ocr",
        origin_image,
        image,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    if filtered:
        return []
    return cells


def attach_picture_children(block: Dict, cells: List[Dict]):
    seen: set[Tuple[Tuple[int, int, int, int], str]] = set()
    children = []
    for cell in cells:
        tbbox = cell.get("bbox")
        text = cell.get("text", "").strip()
        if not tbbox or not text:
            continue
        if not bbox_contains(block["bbox"], tbbox):
            continue
        key = (tuple(tbbox), text)
        if key in seen:
            continue
        seen.add(key)
        child = {
            "bbox": tbbox,
            "category": "PictureText",
            "text": text,
            "source": "picture-ocr",
        }
        if "conf" in cell:
            child["conf"] = cell["conf"]
        children.append(child)
    if children:
        block["picture-children"] = children


def run_two_pass(parser: DotsOCRParser, input_path: str, layout_prompt: str):
    results = parser.parse_file(input_path, prompt_mode=layout_prompt)
    is_pdf = os.path.splitext(input_path)[1].lower() == ".pdf"
    images_cache = load_images_from_pdf(input_path, dpi=parser.dpi) if is_pdf else None
    for result in results:
        layout_path = result.get("layout_info_path")
        if not layout_path or not os.path.exists(layout_path):
            continue
        with open(layout_path, "r", encoding="utf-8") as f:
            blocks = json.load(f)
        if images_cache is not None:
            origin_image = images_cache[result["page_no"]]
        else:
            origin_image = fetch_image(result["file_path"])
        for block in blocks:
            if block.get("category") == "Picture" and "bbox" in block:
                cells = extract_picture_text(parser, origin_image, block["bbox"])
                attach_picture_children(block, cells)
        with open(layout_path, "w", encoding="utf-8") as f:
            json.dump(blocks, f, ensure_ascii=False)
    return results


def main():
    argp = argparse.ArgumentParser(
        description="Run dots.ocr twice: layout parse then picture text extraction.")
    argp.add_argument("input_path", type=str, help="Input PDF/image file path")
    argp.add_argument("--output", type=str, default="./output", help="Output directory")
    argp.add_argument(
        "--layout_prompt",
        choices=["prompt_layout_all_en", "prompt_layout_only_en"],
        default="prompt_layout_all_en",
        help="Prompt used for the first pass layout parsing",
    )
    argp.add_argument("--ip", type=str, default="localhost")
    argp.add_argument("--port", type=int, default=8000)
    argp.add_argument("--model_name", type=str, default="model")
    argp.add_argument("--temperature", type=float, default=0.1)
    argp.add_argument("--top_p", type=float, default=1.0)
    argp.add_argument("--max_completion_tokens", type=int, default=16384)
    argp.add_argument("--num_thread", type=int, default=16)
    argp.add_argument("--dpi", type=int, default=200)
    argp.add_argument("--min_pixels", type=int, default=None)
    argp.add_argument("--max_pixels", type=int, default=None)
    argp.add_argument("--use_hf", action="store_true")
    args = argp.parse_args()

    parser = DotsOCRParser(
        ip=args.ip,
        port=args.port,
        model_name=args.model_name,
        temperature=args.temperature,
        top_p=args.top_p,
        max_completion_tokens=args.max_completion_tokens,
        num_thread=args.num_thread,
        dpi=args.dpi,
        output_dir=args.output,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        use_hf=args.use_hf,
    )
    run_two_pass(parser, args.input_path, args.layout_prompt)


if __name__ == "__main__":
    main()
