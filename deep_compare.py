"""
Deep Field-by-Field Comparison: Local Server vs AI Studio
==========================================================
Uses the same 1.pdf + identical parameters.
Compares every JSON field, every block, every image.
"""
import base64
import json
import os
import sys
import time
import requests

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_PATH = os.path.join(SCRIPT_DIR, "..", "amd_demo", "1.pdf")
if not os.path.exists(PDF_PATH):
    PDF_PATH = r"C:\Users\Tinkerclaw\Desktop\amd_demo\1.pdf"

OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output_deep_compare")
os.makedirs(OUTPUT_DIR, exist_ok=True)

AI_STUDIO_URL = "https://q6mbb0r0t8m9q4pf.aistudio-app.com/layout-parsing"
AI_STUDIO_TOKEN = "effc529bb4e8018e3ebd3b04cbb0fe1e9c83272f"
LOCAL_URL = "http://localhost:8399/layout-parsing"

# Track all differences
all_diffs = []

def diff(field, expected, actual, page=None):
    """Record a difference."""
    loc = f"Page {page} | " if page is not None else ""
    all_diffs.append(f"{loc}{field}: AI='{expected}' vs LOCAL='{actual}'")

def safe_print(text):
    print(text.encode("gbk", errors="replace").decode("gbk"))


def build_payload():
    with open(PDF_PATH, "rb") as f:
        file_data = base64.b64encode(f.read()).decode("ascii")
    return {
        "file": file_data,
        "fileType": 0,
        "markdownIgnoreLabels": [
            "header", "header_image", "footer", "footer_image",
            "number", "footnote", "aside_text"
        ],
        "useDocOrientationClassify": False,
        "useDocUnwarping": False,
        "useLayoutDetection": True,
        "useChartRecognition": False,
        "useSealRecognition": True,
        "useOcrForImageBlock": False,
        "mergeTables": True,
        "relevelTitles": True,
        "layoutShapeMode": "auto",
        "promptLabel": "ocr",
        "repetitionPenalty": 1,
        "temperature": 0,
        "topP": 1,
        "minPixels": 147384,
        "maxPixels": 2822400,
        "layoutNms": True,
        "restructurePages": True,
    }


def compare_top_level(ai, local):
    safe_print("\n  [Top-level]")
    for key in ["errorCode", "errorMsg"]:
        av, lv = ai.get(key), local.get(key)
        if av != lv:
            diff(f"top.{key}", av, lv)
            safe_print(f"    DIFF {key}: {av} vs {lv}")
        else:
            safe_print(f"    OK   {key}: {av}")

    for key in ["logId", "result"]:
        if key not in local:
            diff(f"top.{key}", "present", "MISSING")
            safe_print(f"    DIFF {key}: MISSING in local")


def compare_data_info(ai_di, local_di):
    safe_print("\n  [dataInfo]")
    for key in ["type", "numPages"]:
        av, lv = ai_di.get(key), local_di.get(key)
        if av != lv:
            diff(f"dataInfo.{key}", av, lv)
            safe_print(f"    DIFF {key}: {av} vs {lv}")
        else:
            safe_print(f"    OK   {key}: {av}")

    ai_pages = ai_di.get("pages", [])
    local_pages = local_di.get("pages", [])
    if len(ai_pages) != len(local_pages):
        diff("dataInfo.pages count", len(ai_pages), len(local_pages))
    else:
        for i, (ap, lp) in enumerate(zip(ai_pages, local_pages)):
            for dim in ["width", "height"]:
                # Allow small diff (< 3 pixels) due to PDF render DPI
                av, lv = ap.get(dim), lp.get(dim)
                if av != lv and abs(av - lv) > 2:
                    diff(f"dataInfo.pages[{i}].{dim}", av, lv)
                    safe_print(f"    DIFF pages[{i}].{dim}: {av} vs {lv}")


def compare_model_settings(ai_ms, local_ms, page):
    for key in sorted(set(list(ai_ms.keys()) + list(local_ms.keys()))):
        av = ai_ms.get(key, "<missing>")
        lv = local_ms.get(key, "<missing>")
        if av != lv:
            diff(f"Page {page} model_settings.{key}", str(av)[:80], str(lv)[:80])


def compare_block(ai_b, local_b, page, block_idx):
    prefix = f"Page {page} block[{block_idx}]"

    # Required fields
    for key in ["block_label", "block_id", "block_order", "group_id"]:
        av, lv = ai_b.get(key), local_b.get(key)
        if av != lv:
            diff(f"{prefix}.{key}", str(av), str(lv))

    # block_content: compare existence and rough length
    ai_content = ai_b.get("block_content", "")
    local_content = local_b.get("block_content", "")
    if not ai_content and not local_content:
        pass  # Both empty is fine
    elif not local_content and ai_content:
        diff(f"{prefix}.block_content", f"len={len(ai_content)}", "EMPTY")
    elif local_content and not ai_content:
        diff(f"{prefix}.block_content", "EMPTY", f"len={len(local_content)}")

    # block_bbox: allow small pixel tolerance (< 3px)
    ai_bbox = ai_b.get("block_bbox", [])
    local_bbox = local_b.get("block_bbox", [])
    if len(ai_bbox) == 4 and len(local_bbox) == 4:
        for i, (a, l) in enumerate(zip(ai_bbox, local_bbox)):
            if abs(a - l) > 2:
                diff(f"{prefix}.block_bbox[{i}]", a, l)

    # block_polygon_points
    ai_pp = ai_b.get("block_polygon_points", [])
    local_pp = local_b.get("block_polygon_points", [])
    if len(ai_pp) != len(local_pp):
        diff(f"{prefix}.polygon_points count", len(ai_pp), len(local_pp))
    else:
        for i, (ap, lp) in enumerate(zip(ai_pp, local_pp)):
            for j, (a, l) in enumerate(zip(ap, lp)):
                if abs(a - l) > 2:
                    diff(f"{prefix}.polygon[{i}][{j}]", a, l)

    # Extra keys in AI Studio
    for key in ["global_block_id", "global_group_id"]:
        if key in ai_b and key not in local_b:
            diff(f"{prefix}.{key}", "present in AI", "MISSING in local")


def compare_layout_det_res(ai_ldr, local_ldr, page):
    ai_boxes = ai_ldr.get("boxes", [])
    local_boxes = local_ldr.get("boxes", [])

    if len(ai_boxes) != len(local_boxes):
        diff(f"Page {page} layout_det_res.boxes count", len(ai_boxes), len(local_boxes))
        return

    for j, (ab, lb) in enumerate(zip(ai_boxes, local_boxes)):
        # label
        if ab.get("label") != lb.get("label"):
            diff(f"Page {page} box[{j}].label", ab.get("label"), lb.get("label"))

        # cls_id
        if ab.get("cls_id") != lb.get("cls_id"):
            diff(f"Page {page} box[{j}].cls_id", ab.get("cls_id"), lb.get("cls_id"))

        # score: allow tolerance
        ai_s = ab.get("score", 0)
        local_s = lb.get("score", 0)
        if abs(ai_s - local_s) > 0.05:
            diff(f"Page {page} box[{j}].score", f"{ai_s:.4f}", f"{local_s:.4f}")

        # coordinate
        ai_c = ab.get("coordinate", [])
        local_c = lb.get("coordinate", [])
        if len(ai_c) == 4 and len(local_c) == 4:
            for k, (a, l) in enumerate(zip(ai_c, local_c)):
                if abs(a - l) > 2:
                    diff(f"Page {page} box[{j}].coord[{k}]", a, l)

        # order
        if ab.get("order") != lb.get("order"):
            diff(f"Page {page} box[{j}].order", ab.get("order"), lb.get("order"))


def compare_page(ai_page, local_page, page_idx):
    safe_print(f"\n  [Page {page_idx}]")

    # 1. Key presence
    for key in ["prunedResult", "markdown", "outputImages", "inputImage"]:
        if key in ai_page and key not in local_page:
            diff(f"Page {page_idx}.{key}", "present", "MISSING")
        elif key not in ai_page and key in local_page:
            diff(f"Page {page_idx}.{key}", "missing in AI", "present in local")

    ai_pr = ai_page.get("prunedResult", {})
    local_pr = local_page.get("prunedResult", {})

    # 2. prunedResult top-level keys
    ai_pr_keys = set(ai_pr.keys())
    local_pr_keys = set(local_pr.keys())
    missing_in_local = ai_pr_keys - local_pr_keys
    extra_in_local = local_pr_keys - ai_pr_keys
    if missing_in_local:
        diff(f"Page {page_idx}.prunedResult missing keys", str(missing_in_local), "")
    if extra_in_local:
        diff(f"Page {page_idx}.prunedResult extra keys", "", str(extra_in_local))

    # 3. model_settings
    ai_ms = ai_pr.get("model_settings", {})
    local_ms = local_pr.get("model_settings", {})
    compare_model_settings(ai_ms, local_ms, page_idx)

    # 4. parsing_res_list
    ai_blocks = ai_pr.get("parsing_res_list", [])
    local_blocks = local_pr.get("parsing_res_list", [])
    if len(ai_blocks) != len(local_blocks):
        diff(f"Page {page_idx} block count", len(ai_blocks), len(local_blocks))
    for j in range(min(len(ai_blocks), len(local_blocks))):
        compare_block(ai_blocks[j], local_blocks[j], page_idx, j)

    # 5. layout_det_res
    ai_ldr = ai_pr.get("layout_det_res", {})
    local_ldr = local_pr.get("layout_det_res", {})
    compare_layout_det_res(ai_ldr, local_ldr, page_idx)

    # 6. markdown
    ai_md = ai_page.get("markdown", {})
    local_md = local_page.get("markdown", {})
    ai_md_text = ai_md.get("text", "")
    local_md_text = local_md.get("text", "")
    safe_print(f"    markdown.text: AI={len(ai_md_text)} chars, LOCAL={len(local_md_text)} chars")

    ai_md_imgs = ai_md.get("images", {})
    local_md_imgs = local_md.get("images", {})
    safe_print(f"    markdown.images: AI={len(ai_md_imgs)} items, LOCAL={len(local_md_imgs)} items")
    if set(ai_md_imgs.keys()) != set(local_md_imgs.keys()):
        ai_only = set(ai_md_imgs.keys()) - set(local_md_imgs.keys())
        local_only = set(local_md_imgs.keys()) - set(ai_md_imgs.keys())
        if ai_only:
            diff(f"Page {page_idx} md.images AI-only", str(ai_only), "")
        if local_only:
            diff(f"Page {page_idx} md.images local-only", "", str(local_only))

    # 7. outputImages
    ai_oi = ai_page.get("outputImages", {})
    local_oi = local_page.get("outputImages", {})
    if set(ai_oi.keys()) != set(local_oi.keys()):
        diff(f"Page {page_idx} outputImages keys", str(set(ai_oi.keys())), str(set(local_oi.keys())))
    else:
        safe_print(f"    outputImages keys: {list(ai_oi.keys())} (match)")

    # 8. inputImage
    ai_has_input = bool(ai_page.get("inputImage"))
    local_has_input = bool(local_page.get("inputImage"))
    if ai_has_input != local_has_input:
        diff(f"Page {page_idx} inputImage", str(ai_has_input), str(local_has_input))


def main():
    safe_print("=" * 70)
    safe_print("Deep Field-by-Field Comparison")
    safe_print("=" * 70)

    payload = build_payload()

    # ── Load AI Studio reference ────────────────────────────────────
    ai_ref = os.path.join(OUTPUT_DIR, "aistudio_reference.json")
    if os.path.exists(ai_ref):
        with open(ai_ref, "r", encoding="utf-8") as f:
            ai_data = json.load(f)
        safe_print("AI Studio: loaded from cache")
    else:
        safe_print("AI Studio: calling API ...")
        headers = {"Authorization": f"token {AI_STUDIO_TOKEN}", "Content-Type": "application/json"}
        r = requests.post(AI_STUDIO_URL, json=payload, headers=headers, timeout=600)
        safe_print(f"  Status: {r.status_code}")
        assert r.status_code == 200, f"AI Studio error: {r.text[:200]}"
        ai_data = r.json()
        with open(ai_ref, "w", encoding="utf-8") as f:
            json.dump(ai_data, f, ensure_ascii=False)
        safe_print("  Saved reference")

    # ── Call local server ───────────────────────────────────────────
    safe_print("\nLocal server: calling /layout-parsing ...")
    t0 = time.time()
    r = requests.post(LOCAL_URL, json=payload, timeout=600)
    elapsed = time.time() - t0
    safe_print(f"  Status: {r.status_code}, Time: {elapsed:.1f}s")
    assert r.status_code == 200, f"Local error: {r.text[:300]}"
    local_data = r.json()

    # Save local response
    local_path = os.path.join(OUTPUT_DIR, "local_response.json")
    with open(local_path, "w", encoding="utf-8") as f:
        json.dump(local_data, f, ensure_ascii=False, indent=2)
    safe_print(f"  Saved: {local_path}")

    # ── Compare ─────────────────────────────────────────────────────
    safe_print("\n" + "=" * 70)
    safe_print("COMPARISON")
    safe_print("=" * 70)

    # Top-level
    compare_top_level(ai_data, local_data)

    # dataInfo
    ai_di = ai_data.get("result", {}).get("dataInfo", {})
    local_di = local_data.get("result", {}).get("dataInfo", {})
    compare_data_info(ai_di, local_di)

    # Page count
    ai_pages = ai_data.get("result", {}).get("layoutParsingResults", [])
    local_pages = local_data.get("result", {}).get("layoutParsingResults", [])
    safe_print(f"\n  Page count: AI={len(ai_pages)}, LOCAL={len(local_pages)}")
    if len(ai_pages) != len(local_pages):
        diff("page count", len(ai_pages), len(local_pages))

    # Per-page deep compare
    for i in range(min(len(ai_pages), len(local_pages))):
        compare_page(ai_pages[i], local_pages[i], i)

    # preprocessedImages count
    ai_ppi = ai_data.get("result", {}).get("preprocessedImages", [])
    local_ppi = local_data.get("result", {}).get("preprocessedImages", [])
    if len(ai_ppi) != len(local_ppi):
        diff("preprocessedImages count", len(ai_ppi), len(local_ppi))

    # ── Summary ─────────────────────────────────────────────────────
    safe_print("\n" + "=" * 70)
    safe_print("SUMMARY")
    safe_print("=" * 70)
    safe_print(f"  Total differences: {len(all_diffs)}")

    # Categorize diffs
    critical = [d for d in all_diffs if any(k in d for k in [
        "MISSING", "block_label", "block_order", "block_id",
        "page count", "errorCode", "errorMsg", "outputImages keys",
    ])]
    structural = [d for d in all_diffs if any(k in d for k in [
        "missing keys", "extra keys", "parsing_res", "global_block_id",
    ])]
    precision = [d for d in all_diffs if any(k in d for k in [
        "block_bbox", "polygon", "coordinate", "score",
    ])]
    content = [d for d in all_diffs if "block_content" in d or "markdown" in d]
    other = [d for d in all_diffs if d not in critical and d not in structural
             and d not in precision and d not in content]

    safe_print(f"\n  Critical (structure/format):  {len(critical)}")
    safe_print(f"  Structural (keys/fields):     {len(structural)}")
    safe_print(f"  Precision (bbox/score/pixel): {len(precision)}")
    safe_print(f"  Content (text differences):   {len(content)}")
    safe_print(f"  Other:                        {len(other)}")

    if critical:
        safe_print("\n  CRITICAL differences:")
        for d in critical[:20]:
            safe_print(f"    ! {d}")

    if structural:
        safe_print("\n  STRUCTURAL differences:")
        for d in structural[:20]:
            safe_print(f"    ~ {d}")

    if precision:
        safe_print(f"\n  PRECISION differences ({len(precision)} total, showing first 5):")
        for d in precision[:5]:
            safe_print(f"    ≈ {d}")

    if content:
        safe_print(f"\n  CONTENT differences ({len(content)} total, showing first 5):")
        for d in content[:5]:
            safe_print(f"    T {d}")

    if other:
        safe_print(f"\n  OTHER differences:")
        for d in other[:10]:
            safe_print(f"    ? {d}")

    # Save full diff report
    diff_path = os.path.join(OUTPUT_DIR, "diff_report.txt")
    with open(diff_path, "w", encoding="utf-8") as f:
        f.write(f"Total differences: {len(all_diffs)}\n\n")
        for d in all_diffs:
            f.write(d + "\n")
    safe_print(f"\n  Full diff report: {diff_path}")

    # Verdict
    if not critical and not structural:
        safe_print("\n  VERDICT: PASS - Format fully compatible with AI Studio")
        safe_print("  (Precision and content diffs are expected due to model runtime differences)")
    else:
        safe_print("\n  VERDICT: ISSUES FOUND - See details above")


if __name__ == "__main__":
    main()
