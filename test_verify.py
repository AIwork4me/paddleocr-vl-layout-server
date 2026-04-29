"""
Verification Script: Local /layout-parsing Server vs AI Studio
==============================================================
Compares the local server output against AI Studio reference output
using identical input and parameters.
"""

import base64
import json
import os
import sys
import time
import requests


AI_STUDIO_URL = "https://q6mbb0r0t8m9q4pf.aistudio-app.com/layout-parsing"
AI_STUDIO_TOKEN = os.environ.get("AISTUDIO_TOKEN", "")
LOCAL_URL = os.environ.get("LOCAL_URL", "http://localhost:8399/layout-parsing")


def build_payload(file_path, file_type=0):
    with open(file_path, "rb") as f:
        file_data = base64.b64encode(f.read()).decode("ascii")
    return {
        "file": file_data,
        "fileType": file_type,
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


def compare_structure(ai_data, local_data):
    issues = []

    for key in ["logId", "errorCode", "errorMsg", "result"]:
        if key not in local_data:
            issues.append(f"Missing top-level key: {key}")
            return issues

    ai_r = ai_data["result"]
    loc_r = local_data["result"]

    for key in ["layoutParsingResults", "preprocessedImages", "dataInfo"]:
        if key not in loc_r:
            issues.append(f"Missing result key: {key}")

    ai_pages = len(ai_r.get("layoutParsingResults", []))
    loc_pages = len(loc_r.get("layoutParsingResults", []))
    if ai_pages != loc_pages:
        issues.append(f"Page count: AI={ai_pages}, Local={loc_pages}")
    print(f"  Page count: {ai_pages} {'(match)' if ai_pages == loc_pages else '(MISMATCH)'}")

    for i in range(min(ai_pages, loc_pages)):
        ai_p = ai_r["layoutParsingResults"][i]
        loc_p = loc_r["layoutParsingResults"][i]

        for key in ["prunedResult", "markdown", "outputImages", "inputImage"]:
            if key not in loc_p:
                issues.append(f"Page {i}: missing '{key}'")

        if "prunedResult" not in loc_p:
            continue

        ai_pr = ai_p["prunedResult"]
        loc_pr = loc_p["prunedResult"]

        # Block count
        ai_bl = ai_pr.get("parsing_res_list", [])
        loc_bl = loc_pr.get("parsing_res_list", [])
        if len(ai_bl) != len(loc_bl):
            issues.append(f"Page {i}: blocks {len(ai_bl)} vs {len(loc_bl)}")

        # Block labels & IDs
        for j in range(min(len(ai_bl), len(loc_bl))):
            for key in ["block_label", "block_id", "block_order", "group_id"]:
                if ai_bl[j].get(key) != loc_bl[j].get(key):
                    issues.append(f"Page {i} block {j}: {key} mismatch")

        # Markdown present
        if not loc_p.get("markdown", {}).get("text", ""):
            issues.append(f"Page {i}: empty markdown text")

    return issues


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_verify.py <file.pdf|file.png> [--aistudio]")
        sys.exit(1)

    file_path = sys.argv[1]
    file_type = 0 if file_path.lower().endswith(".pdf") else 1
    run_aistudio = "--aistudio" in sys.argv

    print("=" * 60)
    print("Layout Parsing Verification")
    print("=" * 60)

    payload = build_payload(file_path, file_type)

    # Load or fetch AI Studio reference
    ref_path = "reference_aistudio.json"
    if os.path.exists(ref_path):
        with open(ref_path, "r", encoding="utf-8") as f:
            ai_data = json.load(f)
        print(f"AI Studio reference: loaded from cache")
    elif run_aistudio and AI_STUDIO_TOKEN:
        print("Calling AI Studio ...")
        headers = {"Authorization": f"token {AI_STUDIO_TOKEN}", "Content-Type": "application/json"}
        r = requests.post(AI_STUDIO_URL, json=payload, headers=headers, timeout=600)
        assert r.status_code == 200, f"AI Studio error: {r.status_code}"
        ai_data = r.json()
        with open(ref_path, "w", encoding="utf-8") as f:
            json.dump(ai_data, f, ensure_ascii=False)
        print("AI Studio reference saved")
    else:
        print("ERROR: No AI Studio reference. Run with --aistudio and set AISTUDIO_TOKEN")
        sys.exit(1)

    # Call local server
    print("Calling local server ...")
    t0 = time.time()
    r = requests.post(LOCAL_URL, json=payload, timeout=600)
    elapsed = time.time() - t0
    print(f"Status: {r.status_code}, Time: {elapsed:.1f}s")

    if r.status_code != 200:
        print(f"Error: {r.text[:300]}")
        sys.exit(1)

    local_data = r.json()

    # Compare
    print("\nComparing ...")
    issues = compare_structure(ai_data, local_data)

    if issues:
        print(f"\n{len(issues)} issues found:")
        for i in issues:
            print(f"  - {i}")
    else:
        print("\nAll checks passed!")

    # Save local output
    with open("local_response.json", "w", encoding="utf-8") as f:
        json.dump(local_data, f, ensure_ascii=False, indent=2)

    print(f"\nerrorCode: {local_data.get('errorCode')} | errorMsg: {local_data.get('errorMsg')}")
    pages = local_data.get("result", {}).get("layoutParsingResults", [])
    print(f"Pages: {len(pages)}")
    for i, p in enumerate(pages[:3]):
        md_len = len(p.get("markdown", {}).get("text", ""))
        imgs = len(p.get("markdown", {}).get("images", {}))
        print(f"  Page {i}: {md_len} chars, {imgs} images")


if __name__ == "__main__":
    main()
