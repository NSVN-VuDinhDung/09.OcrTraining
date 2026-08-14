import io
import os
import re
import sys
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor

import time
import pdfplumber
import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse
from PIL import Image

try:
    import pdf_inspector as _pdf_inspector
except ImportError:
    _pdf_inspector = None

try:
    import torch as _torch
except ImportError:
    _torch = None

try:
    import uvicorn
except ImportError:
    uvicorn = None

# Tự động detect GPU – nếu không có thì ép CPU-only
def _detect_device() -> str:
    if _torch is not None and _torch.cuda.is_available():
        name = _torch.cuda.get_device_name(0)
        logging.getLogger(__name__).info(f"GPU detected: {name} — using CUDA")
        return "cuda"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # ép CPU nếu không có GPU
    return "cpu"

_DEVICE = _detect_device()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Thread pool
#   - GPU mode : ít worker CPU hơn (GPU xử lý chính), PyTorch threads = 1
#   - CPU mode : workers = physical cores, mỗi worker dùng 2 PyTorch threads
#     để tránh over-subscription (N workers × 2 threads ≤ tổng logical cores)
# ---------------------------------------------------------------------------
if _DEVICE == "cuda":
    _CPU_PHYSICAL = max(1, (os.cpu_count() or 2) // 4)  # 2-3 workers, GPU làm chính
    _TORCH_THREADS = 1
else:
    _CPU_PHYSICAL = max(1, (os.cpu_count() or 2) // 2)  # 6 on i5-12500
    _TORCH_THREADS = max(1, (os.cpu_count() or 2) // _CPU_PHYSICAL)  # 2
_executor = ThreadPoolExecutor(max_workers=_CPU_PHYSICAL)

# Set ngay khi import để áp dụng cho tất cả thread
if _torch is not None:
    _torch.set_num_threads(_TORCH_THREADS)
    try:
        _torch.set_num_interop_threads(1)
    except RuntimeError:
        pass  # đã bị gọi trước đó (do cuda.is_available()), bỏ qua

# ---------------------------------------------------------------------------
# Model cache – load một lần, dùng mãi
# ---------------------------------------------------------------------------
_models: dict = {}


def get_ocr():
    if "ocr" not in _models:
        logger.info("Loading OCR model...")
        from module.ocr import OCR
        _models["ocr"] = OCR()
        logger.info("OCR model ready.")
    return _models["ocr"]


def get_tsr():
    if "tsr" not in _models:
        logger.info("Loading TSR model...")
        from module import TableStructureRecognizer, LayoutRecognizer
        _models["tsr"] = TableStructureRecognizer()
        _models["layout_cls"] = LayoutRecognizer
        logger.info("TSR model ready.")
    return _models["tsr"], _models["layout_cls"]


def get_layout():
    if "layout" not in _models:
        logger.info("Loading Layout model...")
        from module import LayoutRecognizer
        _models["layout"] = LayoutRecognizer("layout")
        logger.info("Layout model ready.")
    return _models["layout"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PDF_RENDER_DPI = 300          # 300 dpi – khớp với finetuned version
_PT_TO_PX = _PDF_RENDER_DPI / 72.0   # pdfplumber trả toạ độ theo point (72 dpi)
_CID_PATTERN = re.compile(r"\(cid *: *[0-9]+ *\)")


def _file_to_images(data: bytes, filename: str) -> list:
    """Chuyển bytes (ảnh hoặc PDF) thành danh sách PIL.Image."""
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext == "pdf":
        images = []
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            for page in pdf.pages:
                img = page.to_image(resolution=_PDF_RENDER_DPI).annotated
                images.append(img.convert("RGB"))
        return images
    else:
        return [Image.open(io.BytesIO(data)).convert("RGB")]


def _classify_pdf_pages(data: bytes) -> dict | None:
    """Phân loại PDF theo TỪNG TRANG bằng pdf-inspector.

    Trả về ``{page_0indexed: markdown}`` cho các trang có text đáng tin,
    hoặc ``None`` nếu không route được (thiếu lib / lỗi / text bị lỗi encoding)
    — khi đó caller OCR toàn bộ như cũ.

    Chỉ dùng cờ ``needs_ocr`` của từng ``PageMarkdown``. KHÔNG dùng
    ``pdf_type`` (phân loại cả file, quá thô: file 2/3 trang text vẫn có thể
    bị gán ``image_based``) và KHÔNG dùng ``pages_needing_ocr`` (1-indexed,
    lệch với ``PageMarkdown.page`` 0-indexed ngay trong cùng object).
    """
    if _pdf_inspector is None:
        return None
    try:
        res = _pdf_inspector.extract_pages_markdown_bytes(data)
    except Exception as exc:
        logger.warning(f"pdf-inspector không phân loại được, OCR toàn bộ: {exc}")
        return None

    text_md = {}
    for p in res.pages:
        if p.needs_ocr:
            continue
        md = p.markdown or ""
        if _CID_PATTERN.search(md):
            # Font không nhúng bảng mã chuẩn -> text gốc vô dụng, phải OCR
            logger.info(f"Trang {p.page} có lỗi encoding (cid:) -> chuyển sang OCR")
            continue
        text_md[p.page] = md
    return text_md


def _pdf_split_pages(data: bytes, text_pages: set) -> tuple[dict, list]:
    """Mở pdfplumber một lần, xử lý mỗi trang theo đúng đường của nó.

    - Trang thuộc ``text_pages``: ``extract_text_lines()`` -> blocks + bbox
      (đổi point -> pixel @300dpi để cùng hệ toạ độ với bbox của OCR).
    - Trang còn lại: render 300 dpi để đưa vào luồng OCR.

    Trang được xếp vào ``text_pages`` nhưng không bóc ra dòng nào cũng bị
    đẩy sang OCR — không tin cờ tuyệt đối, luôn kiểm tra kết quả thật.

    Trả về ``({page: blocks}, [(page, PIL.Image)])``.
    """
    text_blocks: dict = {}
    ocr_images: list = []

    with pdfplumber.open(io.BytesIO(data)) as pdf:
        for i, page in enumerate(pdf.pages):
            if i in text_pages:
                blocks = [
                    {
                        "text": txt,
                        "score": 1.0,  # text gốc từ file, không phải model đoán
                        "bbox": [
                            int(ln["x0"] * _PT_TO_PX), int(ln["top"] * _PT_TO_PX),
                            int(ln["x1"] * _PT_TO_PX), int(ln["bottom"] * _PT_TO_PX),
                        ],
                    }
                    for ln in page.extract_text_lines()
                    if (txt := (ln.get("text") or "").strip())
                ]
                if blocks:
                    text_blocks[i] = blocks
                    continue
            ocr_images.append(
                (i, page.to_image(resolution=_PDF_RENDER_DPI).annotated.convert("RGB"))
            )

    return text_blocks, ocr_images


def _build_tsr_page(img: Image.Image, lyt: list, ocr) -> str:
    """Tái sử dụng logic get_table_markdown từ t_recognizer.py."""
    from module import TableStructureRecognizer, LayoutRecognizer

    raw_boxes = ocr(np.array(img))
    if not raw_boxes or raw_boxes[0] is None:
        return ""

    avg_height = np.mean([b[-1][1] - b[0][1] for b, _ in raw_boxes]) / 3
    boxes = LayoutRecognizer.sort_Y_firstly(
        [
            {
                "x0": b[0][0], "x1": b[1][0],
                "top": b[0][1], "bottom": b[-1][1],
                "text": t[0],
                "layout_type": "table",
                "page_number": 0,
            }
            for b, t in raw_boxes
            if b[0][0] <= b[1][0] and b[0][1] <= b[-1][1]
        ],
        avg_height,
    )

    def gather(kwd, fzy=10, ption=0.6):
        eles = LayoutRecognizer.sort_Y_firstly(
            [r for r in lyt if re.match(kwd, r["label"])], fzy
        )
        eles = LayoutRecognizer.layouts_cleanup(boxes, eles, 5, ption)
        return LayoutRecognizer.sort_Y_firstly(eles, 0)

    rows = gather(r".* (row|header)")
    headers = gather(r".*header$")
    spans = gather(r".*spanning")
    clmns = sorted(
        [r for r in lyt if re.match(r"table column$", r["label"])],
        key=lambda x: x["x0"],
    )
    clmns = LayoutRecognizer.layouts_cleanup(boxes, clmns, 5, 0.5)

    for b in boxes:
        ii = LayoutRecognizer.find_overlapped_with_threashold(b, rows, thr=0.3)
        if ii is not None:
            b["R"] = ii
            b["R_top"] = rows[ii]["top"]
            b["R_bott"] = rows[ii]["bottom"]

        ii = LayoutRecognizer.find_overlapped_with_threashold(b, headers, thr=0.3)
        if ii is not None:
            b["H"] = ii
            b["H_top"] = headers[ii]["top"]
            b["H_bott"] = headers[ii]["bottom"]
            b["H_left"] = headers[ii]["x0"]
            b["H_right"] = headers[ii]["x1"]

        ii = LayoutRecognizer.find_horizontally_tightest_fit(b, clmns)
        if ii is not None:
            b["C"] = ii
            b["C_left"] = clmns[ii]["x0"]
            b["C_right"] = clmns[ii]["x1"]

        ii = LayoutRecognizer.find_overlapped_with_threashold(b, spans, thr=0.3)
        if ii is not None:
            b["H_top"] = spans[ii]["top"]
            b["H_bott"] = spans[ii]["bottom"]
            b["H_left"] = spans[ii]["x0"]
            b["H_right"] = spans[ii]["x1"]
            b["SP"] = ii

    return TableStructureRecognizer.construct_table(boxes, markdown=True)


# ---------------------------------------------------------------------------
# App + lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    get_ocr()  # preload OCR khi server khởi động
    logger.info(f"Device: {_DEVICE} | Thread pool: {_CPU_PHYSICAL} workers, {_TORCH_THREADS} PyTorch threads/worker")
    yield


app = FastAPI(
    title="DeepDoc VietOCR API",
    description="OCR service cho tài liệu tiếng Việt – hỗ trợ ảnh và PDF",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health", tags=["System"])
def health_check():
    """Kiểm tra trạng thái service."""
    return {"status": "ok"}


@app.post("/v1/ocr", tags=["OCR"])
async def run_ocr(
    file: UploadFile = File(...),
    score_threshold: float = Form(default=0.5, description="Lọc bỏ block có confidence < threshold (0.0–1.0)"),
    output: str = Form(default="blocks", description="blocks: trả về tọa độ + score từng block (mặc định) | text: chỉ trả về text ghép lại"),
):
    """
    Nhận diện văn bản (OCR) từ ảnh hoặc PDF.

    **output=blocks** (mặc định):
    ```json
    { "filename": "...", "pages": [{ "page": 0, "text": "...", "blocks": [{ "text": "...", "score": 0.93, "bbox": [x0,y0,x1,y1] }] }] }
    ```

    **output=text** — chỉ lấy text, không có tọa độ:
    ```json
    { "filename": "...", "text": "toàn bộ nội dung ghép lại", "source": "pdfplumber | ocr | mixed" }
    ```

    **Routing theo TỪNG TRANG** (chỉ với PDF): trang có text gốc đáng tin được
    bóc trực tiếp (~10–50× nhanh hơn, chính xác 100%), trang scan/ảnh mới đưa
    vào OCR. Field `source` ở mỗi trang cho biết trang đó đi đường nào.
    Ảnh và các trang cần OCR đi đúng luồng OCR cũ, không thay đổi.
    """
    data = await file.read()
    filename = file.filename or "upload"
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    # --- Chia trang: text gốc vs cần OCR ---------------------------------
    # text_blocks: {page -> blocks} lấy từ pdfplumber (không OCR)
    # text_md    : {page -> markdown} từ pdf-inspector, dùng cho output=text
    # ocr_targets: [(page, PIL.Image)] đưa vào luồng OCR
    text_blocks: dict = {}
    text_md: dict = {}
    try:
        if ext == "pdf" and (routed := _classify_pdf_pages(data)) is not None:
            text_md = routed
            text_blocks, ocr_targets = _pdf_split_pages(data, set(routed))
            # Trang bị _pdf_split_pages đẩy sang OCR thì bỏ markdown tương ứng
            text_md = {p: md for p, md in text_md.items() if p in text_blocks}
        else:
            # Ảnh, hoặc PDF không route được → OCR toàn bộ như cũ
            ocr_targets = list(enumerate(_file_to_images(data, filename)))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Không đọc được file: {exc}")

    if text_blocks:
        logger.info(
            f"{filename}: {len(text_blocks)} trang text gốc, {len(ocr_targets)} trang OCR"
        )

    # --- Luồng OCR (không đổi) ------------------------------------------
    ocr = get_ocr()
    loop = asyncio.get_event_loop()

    def _ocr_page(page_num: int, img):
        if _torch is not None:
            _torch.set_num_threads(_TORCH_THREADS)
        raw = ocr(np.array(img), 0)
        if not raw or raw[0] is None:
            return {"page": page_num, "source": "ocr", "text": "", "blocks": []}
        blocks = [
            {
                "text": text,
                "score": round(float(score), 4),
                "bbox": [int(b[0][0]), int(b[0][1]), int(b[1][0]), int(b[-1][1])],
            }
            for b, (text, score) in raw
            if b[0][0] <= b[1][0] and b[0][1] <= b[-1][1] and score >= score_threshold
        ]
        return {
            "page": page_num,
            "source": "ocr",
            "text": "\n".join(bl["text"] for bl in blocks),
            "blocks": blocks,
        }

    tasks = [
        loop.run_in_executor(_executor, _ocr_page, page_num, img)
        for page_num, img in ocr_targets
    ]
    pages = list(await asyncio.gather(*tasks))

    # --- Ghép trang text gốc vào cùng schema ----------------------------
    for page_num, blocks in text_blocks.items():
        pages.append({
            "page": page_num,
            "source": "pdfplumber",
            "text": "\n".join(bl["text"] for bl in blocks),
            "blocks": blocks,
        })
    pages.sort(key=lambda x: x["page"])

    sources = {p["source"] for p in pages}
    file_source = sources.pop() if len(sources) == 1 else "mixed"

    if output == "text":
        # Trang text gốc dùng markdown của pdf-inspector (giữ định dạng tiêu đề,
        # bảng...), trang OCR dùng text ghép từ blocks
        parts = [text_md.get(p["page"]) or p["text"] for p in pages]
        return {
            "filename": filename,
            "text": "\n\n".join(t for t in parts if t),
            "source": file_source,
        }

    return {"filename": filename, "source": file_source, "pages": pages}


@app.post("/v1/tsr", tags=["TSR"])
async def run_tsr(
    file: UploadFile = File(...),
    threshold: float = Form(default=0.2),
):
    """
    Nhận diện cấu trúc bảng (Table Structure Recognition).

    **Response** (mỗi phần tử = 1 trang):
    ```json
    {
      "filename": "...",
      "pages": [{ "page": 0, "markdown": "| col1 | col2 |\\n|---|---|\\n..." }]
    }
    ```
    """
    data = await file.read()
    try:
        images = _file_to_images(data, file.filename or "upload")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Không đọc được file: {exc}")

    tsr, _ = get_tsr()
    ocr = get_ocr()
    layouts = tsr(images, thr=float(threshold))
    pages = []

    for page_num, (img, lyt) in enumerate(zip(images, layouts)):
        try:
            markdown = _build_tsr_page(img, lyt, ocr)
        except Exception as exc:
            logger.warning(f"TSR trang {page_num} lỗi: {exc}")
            markdown = ""
        pages.append({"page": page_num, "markdown": markdown})

    return {"filename": file.filename, "pages": pages}


@app.post("/v1/layout", tags=["Layout"])
async def run_layout(
    file: UploadFile = File(...),
    threshold: float = Form(default=0.5),
):
    """
    Nhận diện bố cục tài liệu (Layout Recognition).

    **Response** (mỗi phần tử = 1 trang):
    ```json
    {
      "filename": "...",
      "pages": [
        {
          "page": 0,
          "regions": [{ "type": "text", "bbox": [x0, y0, x1, y1], "score": 0.9 }]
        }
      ]
    }
    ```
    """
    data = await file.read()
    try:
        images = _file_to_images(data, file.filename or "upload")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Không đọc được file: {exc}")

    detr = get_layout()
    layouts = detr.forward(images, thr=float(threshold))
    pages = []

    for page_num, lyt in enumerate(layouts):
        regions = [
            {
                "type": r.get("type", ""),
                "bbox": [int(v) for v in r.get("bbox", [])],
                "score": round(float(r.get("score", 0)), 4),
            }
            for r in lyt
        ]
        pages.append({"page": page_num, "regions": regions})

    return {"filename": file.filename, "pages": pages}


# ---------------------------------------------------------------------------
# pdf-inspector test endpoint
# ---------------------------------------------------------------------------

@app.post("/v1/pdf-inspect", tags=["Experiment"])
async def pdf_inspect(
    file: UploadFile = File(...),
    format: str = Form(default="json", description="json | text — 'text' trả về markdown thuần, dễ đọc hơn"),
):
    """
    **[Thử nghiệm]** Phân tích PDF bằng pdf-inspector (Rust, không dùng OCR).

    - `format=json` (mặc định): trả về JSON với đầy đủ metadata
    - `format=text`: trả về markdown thuần (`text/plain`), dễ copy và render
    """
    if _pdf_inspector is None:
        raise HTTPException(status_code=501, detail="pdf-inspector chưa được cài: pip install pdf-inspector")

    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Chỉ hỗ trợ file .pdf")

    data = await file.read()
    t0 = time.perf_counter()
    try:
        result = _pdf_inspector.process_pdf_bytes(data)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"pdf-inspector lỗi: {exc}")
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)

    markdown = result.markdown or ""

    if format == "text":
        header = (
            f"pdf_type   : {result.pdf_type}\n"
            f"confidence : {round(float(result.confidence), 4)}\n"
            f"time_ms    : {elapsed_ms}\n"
            f"{'─' * 60}\n\n"
        )
        return PlainTextResponse(content=header + markdown)

    return {
        "filename": file.filename,
        "pdf_type": result.pdf_type,
        "confidence": round(float(result.confidence), 4),
        # PdfResult.pages_needing_ocr là 1-indexed (stub không ghi chú) —
        # chuyển về 0-indexed cho khớp với `pages[].page` ở các endpoint khác
        "pages_needing_ocr": [max(0, p - 1) for p in (result.pages_needing_ocr or [])],
        "markdown": markdown,
        "extraction_ms": elapsed_ms,
    }


# ---------------------------------------------------------------------------
# Entry point (dev)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if uvicorn is None:
        raise RuntimeError("uvicorn chưa được cài: pip install uvicorn[standard]")
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
