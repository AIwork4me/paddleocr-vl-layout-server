# PaddleOCR-VL Layout Parsing Server

A FastAPI server that adds the `/layout-parsing` endpoint to any vLLM instance running PaddleOCR-VL, making it fully compatible with the [PaddlePaddle AI Studio](https://aistudio.baidu.com/) API format.

## Why This Exists

AMD GPU servers running PaddleOCR-VL via vLLM only expose the OpenAI-compatible `/v1/chat/completions` endpoint. However, the standard PaddleOCR client code uses the `/layout-parsing` endpoint with a specific request/response format (PDF support, layout detection, structured Markdown output, etc.).

This server bridges that gap — deploy it alongside vLLM to get full AI Studio compatibility.

## Architecture

```
                    AMD GPU Server
                 ┌─────────────────────────┐
                 │                         │
  Client ──────►│  This Server (:8399)    │
  (AI Studio    │     └──► PaddleOCRVL   │
   format)      │           ├─ PP-DocLayoutV2/V3 (local, CPU)  │
                │           └─ vLLM /v1/chat/completions (GPU)  │
                │                         │
                └─────────────────────────┘
```

- **Layout Detection** (PP-DocLayoutV2): Runs locally on CPU, identifies document elements (titles, paragraphs, tables, formulas, images, etc.)
- **Element Recognition** (PaddleOCR-VL-0.9B): Runs on the remote vLLM server (AMD GPU), performs OCR/table/formula/chart recognition

## Quick Start

### Prerequisites

- Python 3.10+
- A running vLLM server with PaddleOCR-VL model loaded (see [vLLM docs](https://docs.vllm.ai/projects/recipes/en/latest/PaddlePaddle/PaddleOCR-VL.html))

### Install

```bash
# Using uv (recommended)
uv venv && source .venv/bin/activate
uv pip install -e .

# Or using pip
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

### Run

```bash
# Set the vLLM server URL
export VLLM_SERVER_URL="http://your-vllm-server:8000/v1"

# Start the server (default port: 8399)
python server.py

# Or with custom port
PORT=9000 python server.py
```

### Test

```bash
# Health check
curl http://localhost:8399/health

# Process a PDF
python -c "
import base64, requests, json

with open('test.pdf', 'rb') as f:
    b64 = base64.b64encode(f.read()).decode()

resp = requests.post('http://localhost:8399/layout-parsing', json={
    'file': b64,
    'fileType': 0,
    'useLayoutDetection': True,
    'temperature': 0,
})

data = resp.json()
for i, page in enumerate(data['result']['layoutParsingResults']):
    print(f'Page {i}: {len(page[\"markdown\"][\"text\"])} chars')
"
```

## API Reference

### `POST /layout-parsing`

Process a document (PDF or image) and return structured layout parsing results.

**Request Body:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `file` | string | **required** | Base64-encoded file content |
| `fileType` | int | **required** | `0` = PDF, `1` = image |
| `markdownIgnoreLabels` | string[] | `null` | Layout labels to exclude from markdown (e.g., `"header"`, `"footer"`) |
| `useLayoutDetection` | bool | `true` | Enable layout detection |
| `useSealRecognition` | bool | `true` | Enable seal/stamp recognition |
| `useChartRecognition` | bool | `false` | Enable chart recognition |
| `useOcrForImageBlock` | bool | `false` | Use OCR for image blocks |
| `mergeTables` | bool | `true` | Merge multi-page tables |
| `relevelTitles` | bool | `true` | Re-level heading hierarchy |
| `restructurePages` | bool | `true` | Restructure page content |
| `layoutNms` | bool | `true` | Apply NMS to layout detection |
| `promptLabel` | string | `"ocr"` | Recognition prompt (`"ocr"`, `"table"`, `"formula"`, `"chart"`) |
| `temperature` | float | `0` | Generation temperature |
| `topP` | float | `1` | Top-p sampling |
| `repetitionPenalty` | float | `1` | Repetition penalty |
| `minPixels` | int | `147384` | Min pixels for image processing |
| `maxPixels` | int | `2822400` | Max pixels for image processing |
| `layoutShapeMode` | string | `"auto"` | Layout polygon mode |

**Response:**

```json
{
  "logId": "uuid",
  "errorCode": 0,
  "errorMsg": "Success",
  "result": {
    "layoutParsingResults": [
      {
        "prunedResult": {
          "page_count": 1,
          "width": 1224,
          "height": 1584,
          "model_settings": { ... },
          "parsing_res_list": [
            {
              "block_label": "text",
              "block_content": "...",
              "block_bbox": [x1, y1, x2, y2],
              "block_id": 0,
              "block_order": 0,
              "group_id": 0,
              "block_polygon_points": [[x, y], ...]
            }
          ],
          "layout_det_res": {
            "boxes": [
              {
                "cls_id": 0,
                "label": "text",
                "score": 0.98,
                "coordinate": [x1, y1, x2, y2],
                "order": 0
              }
            ]
          }
        },
        "markdown": {
          "text": "# Title\n\nParagraph content...",
          "images": {
            "imgs/img_in_image_box_xxx.jpg": "data:image/jpeg;base64,..."
          }
        },
        "outputImages": {
          "layout_det_res": "data:image/jpeg;base64,..."
        },
        "inputImage": "data:image/jpeg;base64,..."
      }
    ],
    "preprocessedImages": ["data:image/jpeg;base64,..."],
    "dataInfo": {
      "type": "pdf",
      "numPages": 15,
      "pages": [{"width": 1224, "height": 1584}, ...]
    }
  }
}
```

### `GET /health`

Returns `{"status": "ok"}`.

## How It Works

1. **Receive** base64-encoded file via `/layout-parsing`
2. **Decode** and save to temp file (PaddleOCRVL needs a file path)
3. **Layout Detection** — PP-DocLayoutV2/V3 runs locally to identify document structure
4. **Element Recognition** — Each detected block is sent to the vLLM server for OCR/table/formula recognition
5. **Post-processing** — `restructure_pages()` merges tables, re-levels headings across pages
6. **Format** — Convert results to AI Studio response format with Markdown, images, and structured JSON

## Deployment for AMD

### Recommended Setup

```bash
# On the AMD GPU server, start vLLM first:
vllm serve PaddlePaddle/PaddleOCR-VL \
    --trust-remote-code \
    --max-num-batched-tokens 16384 \
    --no-enable-prefix-caching

# Then start this server (can be on same machine or different):
VLLM_SERVER_URL="http://localhost:8000/v1" python server.py
```

### Docker (Optional)

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY . .
RUN pip install -e .
ENV VLLM_SERVER_URL=http://vllm:8000/v1
EXPOSE 8399
CMD ["python", "server.py"]
```

## Verified Compatibility

This server has been verified against the PaddlePaddle AI Studio `/layout-parsing` endpoint using the same PDF input. The output structure is identical:

| Aspect | Status |
|---|---|
| Response JSON structure | Identical |
| Page count | Identical |
| `prunedResult` format | Identical |
| `model_settings` keys | Identical |
| `parsing_res_list` block labels & IDs | Identical |
| `markdown.text` | Matching (minor model precision differences) |
| `markdown.images` | Matching (base64 data URLs) |
| `outputImages` | Matching (layout visualization) |
| `dataInfo` | Identical |

## License

MIT
