"""Utilities for merging picture-grounding OCR results into layout JSON."""
from typing import Any, Dict, List, Tuple


def _bbox_key(bbox: List[float]) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    return (
        int(round(x1)),
        int(round(y1)),
        int(round(x2)),
        int(round(y2)),
    )


def _contains(outer: List[float], inner: List[float], tol: float = 0.5) -> bool:
    """Return True if ``inner`` bbox lies within ``outer`` bbox with tolerance."""
    ox1, oy1, ox2, oy2 = outer
    ix1, iy1, ix2, iy2 = inner
    return (
        ix1 >= ox1 - tol
        and iy1 >= oy1 - tol
        and ix2 <= ox2 + tol
        and iy2 <= oy2 + tol
    )


def _to_picturetext_block(txt_blk: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a second-pass Text block into a PictureText block."""
    out: Dict[str, Any] = {
        "category": "PictureText",
        "bbox": [
            float(txt_blk["bbox"][0]),
            float(txt_blk["bbox"][1]),
            float(txt_blk["bbox"][2]),
            float(txt_blk["bbox"][3]),
        ],
        "text": txt_blk.get("text", ""),
        "source": "picture-ocr",
    }
    if "conf" in txt_blk and txt_blk["conf"] is not None:
        try:
            out["conf"] = float(txt_blk["conf"])
        except Exception:
            pass
    return out


def attach_picture_children(
    dots_layout: Dict[str, Any],
    per_picture_grounding: List[Tuple[Dict[str, Any], Dict[str, Any]]],
    *,
    dedup_exact: bool = True,
) -> Dict[str, Any]:
    """Attach PictureText children to each Picture block in ``dots_layout``.

    ``per_picture_grounding`` is a list of pairs where the first element is the
    Picture block reference from ``dots_layout['blocks']`` and the second is the
    grounding OCR JSON result for that picture.
    """
    blocks = dots_layout.get("blocks", [])
    for pic, gjson in per_picture_grounding:
        if pic.get("category") != "Picture":
            continue
        pic.setdefault("children", [])

        seen = set()
        if dedup_exact:
            for ch in pic["children"]:
                if ch.get("category") == "PictureText" and "text" in ch and "bbox" in ch:
                    seen.add((_bbox_key(ch["bbox"]), ch["text"]))

        pb = pic.get("bbox")
        if not pb or len(pb) != 4:
            continue

        for blk in gjson.get("blocks", []):
            if blk.get("category") != "Text":
                continue
            tb = blk.get("bbox")
            text = blk.get("text", "")
            if not tb or len(tb) != 4:
                continue
            if not _contains(pb, tb):
                continue
            if dedup_exact:
                key = (_bbox_key(tb), text)
                if key in seen:
                    continue
            pic["children"].append(_to_picturetext_block(blk))
            if dedup_exact:
                seen.add((_bbox_key(tb), text))

    meta = dots_layout.get("meta", {})
    meta["merge_version"] = "hier-v1"
    meta["source"] = "dots+picture2pass"
    dots_layout["meta"] = meta
    return dots_layout
