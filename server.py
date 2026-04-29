"""
PaddleOCR-VL Layout Parsing Server
====================================
A FastAPI server that wraps PaddleOCRVL + vLLM backend to expose
a /layout-parsing endpoint compatible with PaddlePaddle AI Studio.

Architecture:
  Client -> /layout-parsing -> FastAPI -> PaddleOCRVL (local layout + remote vLLM)

Usage:
  # Set vLLM server URL (default: http://localhost:8000/v1)
  export VLLM_SERVER_URL="http://your-vllm-server:8000/v1"
  python server.py
"""

import base64
import io
import json
import os
import shutil
import tempfile
import time
import traceback
import uuid
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# ─── Configuration ───────────────────────────────────────────────────
VLLM_SERVER_URL = os.environ.get("VLLM_SERVER_URL", "http://localhost:8000/v1")
SERVER_PORT = int(os.environ.get("PORT", "8399"))

# ─── PaddleOCRVL lazy init ──────────────────────────────────────────
_pipeline = None


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        from paddleocr import PaddleOCRVL
        _pipeline = PaddleOCRVL(
            vl_rec_backend="vllm-server",
            vl_rec_server_url=VLLM_SERVER_URL,
        )
    return _pipeline


# ─── Request / Response Models ──────────────────────────────────────

class LayoutParsingRequest(BaseModel):
    file: str = Field(..., description="Base64-encoded file content")
    fileType: int = Field(..., description="0=PDF, 1=image")
    markdownIgnoreLabels: Optional[list[str]] = None
    useDocOrientationClassify: Optional[bool] = False
    useDocUnwarping: Optional[bool] = False
    useLayoutDetection: Optional[bool] = True
    useChartRecognition: Optional[bool] = False
    useSealRecognition: Optional[bool] = True
    useOcrForImageBlock: Optional[bool] = False
    mergeTables: Optional[bool] = True
    relevelTitles: Optional[bool] = True
    layoutShapeMode: Optional[str] = "auto"
    promptLabel: Optional[str] = "ocr"
    repetitionPenalty: Optional[float] = 1
    temperature: Optional[float] = 0
    topP: Optional[float] = 1
    minPixels: Optional[int] = 147384
    maxPixels: Optional[int] = 2822400
    layoutNms: Optional[bool] = True
    restructurePages: Optional[bool] = True


# ─── Parameter Mapping ──────────────────────────────────────────────

PARAM_MAPPING = {
    "markdownIgnoreLabels": "markdown_ignore_labels",
    "useDocOrientationClassify": "use_doc_orientation_classify",
    "useDocUnwarping": "use_doc_unwarping",
    "useLayoutDetection": "use_layout_detection",
    "useChartRecognition": "use_chart_recognition",
    "useSealRecognition": "use_seal_recognition",
    "useOcrForImageBlock": "use_ocr_for_image_block",
    "layoutShapeMode": "layout_shape_mode",
    "promptLabel": "prompt_label",
    "repetitionPenalty": "repetition_penalty",
    "temperature": "temperature",
    "topP": "top_p",
    "minPixels": "min_pixels",
    "maxPixels": "max_pixels",
    "layoutNms": "layout_nms",
}

EXCLUDED_PARAMS = {"file", "fileType", "mergeTables", "relevelTitles", "restructurePages"}


def build_predict_kwargs(req: LayoutParsingRequest) -> dict:
    """Map AI Studio camelCase params to PaddleOCRVL snake_case kwargs."""
    req_dict = req.model_dump(exclude_none=True, exclude=EXCLUDED_PARAMS)
    return {
        pdx_name: req_dict[ai_name]
        for ai_name, pdx_name in PARAM_MAPPING.items()
        if ai_name in req_dict
    }


# ─── Image Utilities ────────────────────────────────────────────────

def image_to_base64_url(img_bytes: bytes, fmt: str = ".jpg") -> str:
    b64 = base64.b64encode(img_bytes).decode("ascii")
    mime = "image/jpeg" if fmt == ".jpg" else "image/png"
    return f"data:{mime};base64,{b64}"


LAYOUT_COLORS = {
    "doc_title": (255, 0, 0), "paragraph_title": (0, 0, 255),
    "text": (0, 255, 0), "table": (255, 255, 0), "figure": (255, 0, 255),
    "figure_caption": (0, 255, 255), "image": (128, 0, 255),
    "image_box": (128, 0, 255), "chart": (0, 128, 255),
    "chart_box": (0, 128, 255), "formula": (255, 128, 0),
    "header": (128, 128, 128), "footer": (128, 128, 128),
    "aside_text": (128, 128, 128), "footnote": (128, 128, 128),
    "number": (128, 128, 128), "seal": (0, 200, 0),
}


def draw_layout_visualization(img_array: np.ndarray, boxes: list) -> bytes:
    vis = img_array.copy()
    for box in boxes:
        coord = box.get("coordinate", [])
        label = box.get("label", "text")
        if len(coord) == 4:
            x1, y1, x2, y2 = [int(c) for c in coord]
            color = LAYOUT_COLORS.get(label, (0, 255, 0))
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            cv2.putText(vis, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    _, buf = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return buf.tobytes()


# ─── Result Conversion ──────────────────────────────────────────────

def convert_result(res, page_idx: int, img_array: np.ndarray,
                   output_dir: str) -> dict:
    """Convert PaddleOCRVLResult to AI Studio layoutParsingResults format."""

    # 1. Structured JSON via save_to_json
    json_path = os.path.join(output_dir, f"page_{page_idx}.json")
    res.save_to_json(save_path=json_path)
    with open(json_path, "r", encoding="utf-8") as f:
        res_json = json.load(f)

    # Build prunedResult (strip local-only keys)
    pruned_result = {k: v for k, v in res_json.items()
                     if k not in ("input_path", "page_index")}
    if "layout_det_res" in pruned_result and isinstance(pruned_result["layout_det_res"], dict):
        pruned_result["layout_det_res"] = {
            k: v for k, v in pruned_result["layout_det_res"].items()
            if k not in ("input_path", "page_index")
        }

    # 2. Markdown via save_to_markdown
    md_path = os.path.join(output_dir, f"page_{page_idx}.md")
    res.save_to_markdown(save_path=md_path)
    with open(md_path, "r", encoding="utf-8") as f:
        markdown_text = f.read()

    markdown_images = {}
    imgs_dir = os.path.join(output_dir, "imgs")
    if os.path.isdir(imgs_dir):
        for img_name in os.listdir(imgs_dir):
            img_full = os.path.join(imgs_dir, img_name)
            with open(img_full, "rb") as f:
                markdown_images[f"imgs/{img_name}"] = image_to_base64_url(f.read())

    # 3. Output images (layout detection visualization)
    output_images = {}
    boxes = pruned_result.get("layout_det_res", {}).get("boxes", [])
    if boxes and img_array is not None:
        output_images["layout_det_res"] = image_to_base64_url(
            draw_layout_visualization(img_array, boxes)
        )

    # 4. Input image
    _, buf = cv2.imencode(".jpg", img_array, [cv2.IMWRITE_JPEG_QUALITY, 90])
    input_image = image_to_base64_url(buf.tobytes())

    return {
        "prunedResult": pruned_result,
        "markdown": {"text": markdown_text, "images": markdown_images},
        "outputImages": output_images,
        "inputImage": input_image,
    }


def load_pages_from_file(tmp_path: str, file_type: int, file_bytes: bytes) -> dict:
    """Load page images from PDF or image file. Returns {page_idx: np.ndarray}."""
    pages = {}
    if file_type == 0:
        import fitz
        doc = fitz.open(tmp_path)
        for i in range(len(doc)):
            pix = doc[i].get_pixmap(dpi=150)
            arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height, pix.width, pix.n)
            if pix.n == 4:
                arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
            elif pix.n == 1:
                arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
            pages[i] = arr
        doc.close()
    else:
        img = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
        if img is not None:
            pages[0] = img
    return pages


# ─── FastAPI App ────────────────────────────────────────────────────

app = FastAPI(title="PaddleOCR-VL Layout Parsing Server")


@app.post("/layout-parsing")
async def layout_parsing(req: LayoutParsingRequest):
    start_time = time.time()
    log_id = str(uuid.uuid4())

    try:
        pipeline = get_pipeline()

        # Decode and save to temp file
        file_bytes = base64.b64decode(req.file)
        suffix = ".pdf" if req.fileType == 0 else ".png"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        try:
            # Run prediction
            kwargs = build_predict_kwargs(req)
            results = list(pipeline.predict(tmp_path, **kwargs))

            # Post-processing
            if req.restructurePages:
                try:
                    pipeline.restructure_pages(
                        results,
                        merge_tables=req.mergeTables,
                        relevel_titles=req.relevelTitles,
                    )
                except Exception:
                    pass

            # Load original pages for visualization
            original_pages = load_pages_from_file(tmp_path, req.fileType, file_bytes)

            # Convert results
            result_tmp_dir = tempfile.mkdtemp(prefix="layout_result_")
            try:
                layout_results = []
                for i, res in enumerate(results):
                    page_img = original_pages.get(i)
                    layout_results.append(convert_result(res, i, page_img, result_tmp_dir))
            finally:
                shutil.rmtree(result_tmp_dir, ignore_errors=True)

            # Build response
            data_info = {"type": "pdf" if req.fileType == 0 else "image"}
            if req.fileType == 0:
                data_info["numPages"] = len(original_pages)
                data_info["pages"] = [
                    {"width": int(a.shape[1]), "height": int(a.shape[0])}
                    for a in original_pages.values()
                ]
            else:
                for a in original_pages.values():
                    data_info["width"] = int(a.shape[1])
                    data_info["height"] = int(a.shape[0])
                    break

            elapsed = time.time() - start_time
            print(f"[{log_id}] {elapsed:.1f}s, {len(layout_results)} pages")

            return {
                "logId": log_id,
                "errorCode": 0,
                "errorMsg": "Success",
                "result": {
                    "layoutParsingResults": layout_results,
                    "preprocessedImages": [r["inputImage"] for r in layout_results],
                    "dataInfo": data_info,
                },
            }
        finally:
            os.unlink(tmp_path)

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={
            "logId": log_id, "errorCode": -1, "errorMsg": str(e), "result": None,
        })


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=SERVER_PORT)
