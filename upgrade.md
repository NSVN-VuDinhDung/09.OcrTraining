# Nhật ký cải tiến (Upgrade Log)

> Tài liệu này ghi lại toàn bộ thay đổi quan trọng so với phiên bản gốc (`first commit: 88f18bd`).

---

## 1. `module/ocr.py` — Trả về confidence score thực thay vì hardcode 1.0

### Thay đổi
```python
# TRƯỚC
text = self.detector.predict(img)
results.append((text, 1.0))          # score luôn = 1.0, không có ý nghĩa

# SAU
text, prob = self.detector.predict(img, return_prob=True)
results.append((text, float(prob)))  # score thực từ model
```

### Lý do
VietOCR hỗ trợ tham số `return_prob=True`, trả về xác suất dự đoán thực tế của model.  
Khi score bị hardcode là `1.0`, toàn bộ logic lọc theo `score_threshold` ở API và CLI đều vô nghĩa — mọi block luôn pass qua bộ lọc.  
Sau khi sửa, `score_threshold` hoạt động đúng: các block OCR có độ tin cậy thấp sẽ bị loại.

---

## 2. `module/ocr_onnx.py` — Chuẩn bị biến `cpu_count` cho cấu hình ONNX Runtime

### Thay đổi
```python
# Thêm ngay trước khi tạo SessionOptions
cpu_count = os.cpu_count() or 4
```

### Lý do
Thêm biến `cpu_count` để chuẩn bị cho việc cấu hình động `intra_op_num_threads` / `inter_op_num_threads` của ONNX Runtime dựa theo số core thực tế của máy thay vì hardcode.  
Hiện tại hai giá trị này đang là `2` — có thể dùng `cpu_count` để điều chỉnh tối ưu hơn trong tương lai.

---

## 3. `requirements.txt` — Tách torch ra khỏi pip, thêm stack FastAPI + VietOCR

### Thay đổi

| Hành động | Package |
|-----------|---------|
| **Xóa** | `torch` (cài qua pip sẽ kéo version CUDA, nặng và sai) |
| **Thêm** | `fastapi`, `uvicorn[standard]`, `python-multipart` |
| **Thêm** | `vietocr`, `torchvision`, `einops`, `gdown`, `lmdb`, `trio` |
| **Ghi chú** | Hướng dẫn cài torch CPU đúng cách qua `pytorch.org` |

### Lý do
- `torch` cần cài với index URL chỉ định (`https://download.pytorch.org/whl/cpu`) để lấy đúng bản CPU-only, tránh tải bản CUDA ~2GB không cần thiết.
- `fastapi` + `uvicorn` + `python-multipart` là bộ ba bắt buộc để chạy REST API.
- `vietocr` và các phụ thuộc của nó (`einops`, `gdown`, `lmdb`) cần được liệt kê tường minh vì pip không tự resolve khi dùng `--no-deps`.

---

## 4. `api.py` — File mới: REST API hoàn chỉnh cho OCR service

### Tổng quan
File mới, xây dựng REST API bằng **FastAPI** để expose toàn bộ pipeline qua HTTP.

### Các endpoint

| Method | Endpoint | Chức năng |
|--------|----------|-----------|
| `GET` | `/health` | Health check — kiểm tra service còn sống |
| `POST` | `/v1/ocr` | OCR ảnh hoặc PDF, trả về text + bounding box + score |
| `POST` | `/v1/tsr` | Table Structure Recognition — trả về Markdown table |
| `POST` | `/v1/layout` | Layout Detection — trả về vùng bố cục + loại + score |

### Thiết kế quan trọng

#### a) Model cache — load một lần, dùng mãi
```python
_models: dict = {}

def get_ocr():
    if "ocr" not in _models:
        _models["ocr"] = OCR()
    return _models["ocr"]
```
**Lý do:** Model OCR/TSR/Layout nặng (~vài trăm MB). Nếu load mỗi request sẽ cực kỳ chậm. Cache vào dict global, load đúng 1 lần khi server khởi động.

#### b) Thread pool – tránh over-subscription
```python
_CPU_PHYSICAL = max(1, (os.cpu_count() or 2) // 2)   # số physical core
_TORCH_THREADS = max(1, (os.cpu_count() or 2) // _CPU_PHYSICAL)  # thread/worker
_executor = ThreadPoolExecutor(max_workers=_CPU_PHYSICAL)
```
**Lý do:** FastAPI chạy trên async event loop (single thread). OCR là CPU-bound, cần chạy trong `ThreadPoolExecutor` để không block event loop.  
Công thức đảm bảo: `workers × PyTorch_threads = logical_cores` — dùng 100% CPU mà không over-subscribe.  
Với i5-12500 (12 logical cores): 6 workers × 2 threads = 12 = đúng bằng tổng logical cores.

#### c) Async + run_in_executor cho xử lý song song
```python
tasks = [loop.run_in_executor(_executor, _ocr_page, i, img) for i, img in enumerate(images)]
pages = list(await asyncio.gather(*tasks))
```
**Lý do:** PDF nhiều trang được OCR song song trên nhiều core thay vì tuần tự từng trang.

#### d) Preload model khi server khởi động (lifespan)
```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    get_ocr()  # warm-up
    yield
```
**Lý do:** Request đầu tiên sau khi server khởi động sẽ không bị chậm vì model đã được load sẵn.

#### e) Auto-detect GPU, fallback CPU
```python
def _detect_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # ép CPU nếu không có GPU
    return "cpu"

_DEVICE = _detect_device()
```
**Lý do:** Phiên bản cũ hardcode `CUDA_VISIBLE_DEVICES = ""` khiến GPU bị bỏ qua hoàn toàn, lãng phí tài nguyên trên server có GPU.  
Phiên bản mới tự phát hiện GPU khi khởi động: nếu có thì dùng CUDA, nếu không thì mới ép CPU-only.

**Thread pool cũng điều chỉnh theo:**
- **GPU mode:** ít CPU workers hơn (2-3), GPU xử lý chính, `PyTorch_threads = 1`
- **CPU mode:** giữ nguyên `workers × 2 = logical_cores`

---

## 5. `run_command.md` — File mới: tài liệu lệnh chạy

### Nội dung
Tài liệu hướng dẫn toàn bộ vòng đời dự án:
1. Tạo virtual environment
2. Cài PyTorch CPU đúng cách
3. Cài dependencies
4. Chạy API server (lệnh chính + lệnh uvicorn với reload)
5. Chạy CLI OCR (ảnh đơn / thư mục)
6. Chạy Layout Recognizer
7. Chạy Table Structure Recognizer

**Lý do:** Tập trung toàn bộ lệnh cần thiết vào một file để onboard nhanh, không cần đọc README dài.

---

## Tóm tắt tác động — v1 (khởi tạo)

| File | Loại thay đổi | Tác động |
|------|---------------|----------|
| `module/ocr.py` | Sửa | Score OCR chính xác → lọc `score_threshold` hoạt động đúng |
| `module/ocr_onnx.py` | Sửa nhỏ | Chuẩn bị cấu hình ONNX Runtime linh hoạt |
| `requirements.txt` | Sửa | Cài đặt đúng, đủ, không bị lỗi torch/CUDA |
| `api.py` | Mới | REST API production-ready cho toàn bộ pipeline |
| `run_command.md` | Mới | Tài liệu lệnh chạy nhanh |

---

---

# Lịch sử thay đổi — v2 (2026-08-06)

---

## 6. `api.py` — Auto-detect GPU, không hardcode CPU-only

### Thay đổi
```python
# TRƯỚC: luôn tắt GPU
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# SAU: detect tự động
def _detect_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    return "cpu"

_DEVICE = _detect_device()
```
Thread pool cũng điều chỉnh theo `_DEVICE`:
- **CUDA**: ít worker CPU hơn (logical_cores // 4), PyTorch threads = 1
- **CPU**: giữ nguyên công thức cũ (workers × 2 = logical_cores)

### Lý do
Phiên bản cũ lãng phí GPU trên server có card. Phiên bản mới tự phát hiện và dùng CUDA nếu có, chỉ ép CPU-only khi không có GPU.

---

## 7. `api.py` — Endpoint thử nghiệm `/v1/pdf-inspect`

### Thay đổi
Thêm endpoint mới trong tag `Experiment`:

```
POST /v1/pdf-inspect
  params: file (PDF), format (json | text)
  → phân loại PDF + extract text không dùng OCR
```

Response `format=json`:
```json
{
  "pdf_type": "TextBased",
  "confidence": 0.98,
  "pages_needing_ocr": [],
  "markdown": "# Nội dung...",
  "extraction_ms": 142.3
}
```

Response `format=text`: trả về `text/plain` với header tóm tắt + nội dung markdown, dễ copy vào markdown viewer.

### Lý do
Tạo endpoint riêng để test chất lượng pdf-inspector trên file thực tế trước khi quyết định tích hợp chính thức vào pipeline OCR. Khi `format=text` tránh vấn đề `\n` bị encode trong JSON.

---

## 8. `api.py` + `requirements.txt` — Tích hợp pdf-inspector vào `/v1/ocr`

### Thay đổi — `requirements.txt`
Thêm `pdf-inspector` (có wheel sẵn trên PyPI, không cần build Rust).

### Thay đổi — `/v1/ocr`
Thêm tham số `output`:

| Giá trị | Hành vi |
|---------|---------|
| `blocks` (mặc định) | Giữ nguyên response cũ: `{pages: [{page, text, blocks[{text,score,bbox}]}]}` |
| `text` | Chỉ trả text ghép lại, không có tọa độ: `{text: "...", source: "..."}` |

**Smart routing khi `output=text` + file PDF:**
```
PDF → pdf-inspector classify (~20ms)
  TextBased → extract trực tiếp (~150ms) → source: "pdf-inspector"
  Scanned / lỗi → fallback OCR như cũ   → source: "ocr"
```

Field `source` trong response cho biết đường nào đã xử lý.

### Lý do
- `output=blocks`: không thay đổi gì — đảm bảo backward-compatible.
- `output=text`: tối ưu cho use case chỉ cần nội dung text (LLM, search index, tóm tắt) — PDF text gốc nhanh hơn 10–50× so với OCR.
- Hạn chế sửa core: toàn bộ logic routing nằm trong `api.py`, không chạm vào `module/`.

---

## 9. `run_command.md` — Thêm lệnh fix lỗi port

### Thay đổi
Thêm lệnh kill process khi gặp lỗi `[Errno 10048] port 8000 already in use`:
```powershell
Get-NetTCPConnection -LocalPort 8000 -State Listen | ... | Stop-Process -Force
```

### Lý do
Lỗi này xảy ra khi server cũ vẫn còn chạy sau khi terminal bị đóng đột ngột. Lệnh này giải phóng port trước khi restart.

---

## Tóm tắt tác động — v2

| File | Thay đổi | Tác động |
|------|----------|----------|
| `api.py` | Auto-detect GPU | Tận dụng GPU nếu có, không lãng phí |
| `api.py` | `/v1/pdf-inspect` (mới) | Test pdf-inspector độc lập trước khi dùng |
| `api.py` | `/v1/ocr` thêm `output` param | Chọn giữa tọa độ đầy đủ vs text ghép, + routing thông minh |
| `requirements.txt` | Thêm `pdf-inspector` | Cài được trên máy mới bằng `pip install -r requirements.txt` |
| `run_command.md` | Thêm lệnh kill port | Fix lỗi port conflict khi restart |

---

---

# Lịch sử thay đổi — v3 (2026-08-06)

---

## 10. `api.py` — Sửa bug `pdf_type` sai giá trị so sánh

### Thay đổi
```python
# TRƯỚC — sai, không bao giờ match
if pi.pdf_type == "TextBased":

# SAU — đúng, Python binding trả snake_case
if pi.pdf_type == "text_based":
```

### Lý do
Python binding của pdf-inspector trả về `"text_based"` (snake_case), trong khi Node.js binding trả `"TextBased"` (PascalCase). Do so sánh sai nên điều kiện không bao giờ đúng — mọi PDF đều fallthrough xuống OCR, khiến endpoint `/v1/ocr?output=text` chậm như OCR thông thường thay vì ~150ms của pdf-inspector.

---

## 11. `api.py` — Dọn dẹp toàn bộ import vào top-level

### Vấn đề
Nhiều `import` nằm giữa thân hàm, gây khó đọc và dễ nhầm về dependency của file:
- `import pdfplumber` trong `_file_to_images()`
- `import time` trong `pdf_inspect()`
- `import torch` trong `_ocr_page()` (nested function)
- `import torch` trong `_detect_device()` (lazy init)
- `import uvicorn` trong `if __name__ == "__main__":`

### Thay đổi — cấu trúc import mới ở top-level

```python
import time
import pdfplumber

try:
    import torch as _torch        # None nếu không có
except ImportError:
    _torch = None

try:
    import pdf_inspector as _pdf_inspector   # None nếu không cài
except ImportError:
    _pdf_inspector = None

try:
    import uvicorn                 # None nếu không cài
except ImportError:
    uvicorn = None
```

`_detect_device()` được đơn giản hóa dùng `_torch` trực tiếp, không còn lazy import bên trong:

```python
# TRƯỚC
def _detect_device() -> str:
    try:
        import torch
        if torch.cuda.is_available(): ...

# SAU
def _detect_device() -> str:
    if _torch is not None and _torch.cuda.is_available(): ...
```

Khối set thread cũng gọn hơn:

```python
# TRƯỚC
_torch = None
try:
    import torch as _torch
    _torch.set_num_threads(...)
except Exception:
    pass

# SAU
if _torch is not None:
    _torch.set_num_threads(_TORCH_THREADS)
    _torch.set_num_interop_threads(1)
```

### Ngoại lệ có chủ đích — `from module import ...` vẫn ở trong hàm

Các import sau **cố ý giữ** trong hàm vì là **deferred model loading**:

| Hàm | Import | Lý do giữ lazy |
|-----|--------|----------------|
| `get_ocr()` | `from module.ocr import OCR` | Load model chỉ khi request đầu tiên, không làm chậm server startup |
| `get_tsr()` | `from module import TableStructureRecognizer` | Như trên |
| `get_layout()` | `from module import LayoutRecognizer` | Như trên |
| `_build_tsr_page()` | `from module import ...` | Hàm helper, phụ thuộc vào sys.path đã set |

Nếu chuyển lên top-level, tất cả model sẽ được load ngay khi `import api` — không mong muốn.

### Lý do tổng quát
- Import ở top-level làm rõ dependency của toàn file chỉ bằng cách đọc phần đầu
- Dùng `_name = None` pattern để kiểm tra availability tường minh thay vì try/except mỗi lần dùng
- Hiệu năng không đổi (Python cache module trong `sys.modules`, import lần 2 là dict lookup ~ns)

---

## Tóm tắt tác động — v3

| File | Thay đổi | Tác động |
|------|----------|----------|
| `api.py` | Fix `pdf_type == "text_based"` | `/v1/ocr?output=text` với PDF text gốc giờ đúng ~150ms thay vì chạy OCR |
| `api.py` | Toàn bộ import lên top-level | Code sạch, dễ đọc dependency, không ảnh hưởng hiệu năng |

---

---

# Lịch sử thay đổi — v4 (2026-08-07)

---

## 12. `api.py` — Sửa bug `IndexError` khi trang PDF rỗng

### Thay đổi
```python
# TRƯỚC — crash khi OCR trả về list rỗng []
if raw is None or raw[0] is None:

# SAU — xử lý đúng cả 3 trường hợp: None, [], [None, ...]
if not raw or raw[0] is None:
```

### Lý do
Khi trang PDF gần như trắng (không có text), `ocr()` trả về `[]` thay vì `None`.  
`raw[0]` trên list rỗng ném `IndexError` → FastAPI bắt thành 500 Internal Server Error.  
`not raw` xử lý đồng thời `None`, `[]`, và các falsy value khác.

**Phát hiện:** File `20260731_..._Hợp đồng bảo hiểm tòa nhà Việt Hải_compressed.pdf` (38 trang) bị lỗi do trang cuối (trang 37) là trang rỗng.

---

## 13. `module/ocr.py` — Thêm padding bbox trước khi crop cho VietOCR

### Thay đổi
Thêm hàm `expand_box()` và áp dụng trước bước `get_rotate_crop_image`:

```python
def expand_box(box, pad_x=3, pad_y=6, img_shape=None):
    """Mở rộng bbox thêm pad_x/pad_y pixel mỗi phía để VietOCR thấy đầy đủ dấu thanh."""
    box = box.copy().astype(np.float32)
    vec_x = (box[1] - box[0])  # top-left → top-right
    vec_y = (box[3] - box[0])  # top-left → bottom-left
    len_x = np.linalg.norm(vec_x)
    len_y = np.linalg.norm(vec_y)
    if len_x < 1 or len_y < 1:
        return box
    unit_x = vec_x / len_x
    unit_y = vec_y / len_y
    box[0] -= unit_x * pad_x + unit_y * pad_y
    box[1] += unit_x * pad_x - unit_y * pad_y
    box[2] += unit_x * pad_x + unit_y * pad_y
    box[3] -= unit_x * pad_x - unit_y * pad_y
    if img_shape is not None:
        h, w = img_shape[:2]
        box[:, 0] = np.clip(box[:, 0], 0, w - 1)
        box[:, 1] = np.clip(box[:, 1], 0, h - 1)
    return box

# Trong OCR.__call__:
# TRƯỚC
tmp_box = copy.deepcopy(dt_boxes[bno])

# SAU
tmp_box = expand_box(dt_boxes[bno], pad_x=3, pad_y=6, img_shape=ori_im.shape)
```

### Lý do
Tiếng Việt có ký tự 2 tầng dấu chồng nhau (ắ, ổ, ợ...). PaddleOCR detection khớp bbox sát với vùng text — nhưng VietOCR resize crop về height=32px, khiến dấu thanh tầng trên bị cắt → model đọc sai dấu ("BÀO HIẾM" thay vì "BẢO HIỂM").

Padding `pad_y=6px` dọc theo hướng vuông góc với dòng chữ cho VietOCR thấy đầy đủ nét chữ ở rìa trên/dưới.  
Bbox trả về trong response API **không thay đổi** (vẫn dùng `dt_boxes` gốc), chỉ ảnh hưởng vùng crop đưa vào recognition.

**Giá trị mặc định:** `pad_x=3`, `pad_y=6` — điều chỉnh trong `module/ocr.py` nếu cần.

---

## Tóm tắt tác động — v4

| File | Thay đổi | Tác động |
|------|----------|----------|
| `api.py` | Fix `not raw` thay `raw is None` | Hết lỗi 500 với trang PDF rỗng/trắng |
| `module/ocr.py` | Thêm `expand_box()` + padding 3/6px | Cải thiện nhận dạng dấu thanh tiếng Việt |

---

---

# Lịch sử thay đổi — v5 (2026-08-13)

---

## 14. `module/ocr.py` — VietOCR dùng GPU khi có, thay vì hardcode CPU

### Vấn đề

`TextRecognizer.__init__` hardcode `config['device'] = 'cpu'`, nên VietOCR **luôn** chạy CPU
kể cả khi `_detect_device()` trong `api.py` đã phát hiện GPU và báo `"cuda"`.

Hậu quả nghiêm trọng hơn là: ở chế độ GPU, `api.py` chủ động **giảm** tài nguyên CPU
(`_CPU_PHYSICAL = cores // 4`, `_TORCH_THREADS = 1`) vì tưởng GPU đang gánh phần chính.
Nhưng recognition — phần nặng nhất của pipeline — vẫn nằm trên CPU với ít worker và ít thread hơn.
→ **Máy có GPU chạy chậm hơn máy chỉ có CPU**, ngược hoàn toàn với ý định thiết kế.

### Thay đổi

Thêm hàm `resolve_vietocr_device()` (đặt ngay trước class `TextRecognizer`):

```python
def resolve_vietocr_device(device_id: int | None = None) -> str:
    """Chọn device cho VietOCR: cuda:<id> nếu có GPU, ngược lại cpu.

    Có thể ép thủ công bằng biến môi trường VIETOCR_DEVICE (vd: 'cpu', 'cuda:0').
    """
    forced = os.environ.get("VIETOCR_DEVICE")
    if forced:
        return forced

    try:
        if not torch.cuda.is_available():
            return "cpu"
        count = torch.cuda.device_count()
        if count <= 0:
            return "cpu"
        idx = device_id if device_id is not None else 0
        if idx >= count:
            idx = 0
        return f"cuda:{idx}"
    except Exception:
        return "cpu"
```

Và trong `TextRecognizer.__init__`:

```python
# TRƯỚC
config['device'] = 'cpu'

# SAU
config['device'] = resolve_vietocr_device(device_id)
logging.info(f"VietOCR recognizer uses device: {config['device']}")
```

### Vì sao an toàn với các luồng đang chạy

| Luồng | Hành vi sau khi sửa |
|-------|---------------------|
| **Multi-GPU** (`PARALLEL_DEVICES > 0`) | `device_id` đã được `OCR.__init__` truyền sẵn vào `TextRecognizer(model_dir, device_id)` → mỗi recognizer nằm đúng card của nó, **khớp với `TextDetector` ONNX cùng `device_id`** |
| **CLI `t_ocr.py`** | File set `CUDA_VISIBLE_DEVICES=''` ngay đầu, trước khi import torch → `is_available()` = False → vẫn CPU, **không đổi hành vi cũ** |
| **`api.py` máy không GPU** | `_detect_device()` cũng set `CUDA_VISIBLE_DEVICES=''` lúc import, chạy **trước** khi `get_ocr()` lazy-import `module.ocr` → nhất quán, không có chuyện API log `cpu` mà VietOCR lại bám GPU |
| **Guard `idx >= count`** | Nếu `device_id` vượt số card thực tế thì fallback về card 0 thay vì crash |
| **`try/except`** | Mọi lỗi khi hỏi torch (không có torch, driver lỗi...) đều fallback `cpu` — không bao giờ làm chết service |

### Escape hatch — biến môi trường `VIETOCR_DEVICE`

Ép device thủ công mà không cần sửa code:

```powershell
# Ép CPU trên máy có GPU (để test / so sánh tốc độ)
$env:VIETOCR_DEVICE = "cpu"

# Ép dùng card thứ 2
$env:VIETOCR_DEVICE = "cuda:1"
```

---

## 15. `api.py` — Log rõ device đang dùng lúc khởi động

### Thay đổi
```python
# TRƯỚC
logger.info(f"Thread pool: {_CPU_PHYSICAL} workers, {_TORCH_THREADS} PyTorch threads/worker")

# SAU
logger.info(f"Device: {_DEVICE} | Thread pool: {_CPU_PHYSICAL} workers, {_TORCH_THREADS} PyTorch threads/worker")
```

### Lý do
Chính vì bug ở mục 14 mà không ai phát hiện ra sớm: log cũ chỉ in thread pool, không in device.
Giờ chỉ cần nhìn dòng log đầu tiên là biết service đang chạy `cpu` hay `cuda`,
đối chiếu ngay với dòng `VietOCR recognizer uses device: ...` từ `module/ocr.py`.

---

## Đính chính tài liệu

Mục 13 (v4) ghi giá trị mặc định là `pad_x=3, pad_y=6`, nhưng code hiện tại tại
`module/ocr.py` (trong `OCR.__call__`) đang dùng `pad_x=3, pad_y=10` —
giá trị đã được tinh chỉnh thêm sau khi viết tài liệu v4. Signature của `expand_box()`
vẫn giữ default `pad_y=6`, chỉ call-site truyền `10`.

---

## Việc còn tồn đọng (chưa làm)

**Thread pool ở chế độ GPU chưa được tinh chỉnh lại.**
Sau khi sửa mục 14, khi chạy trên GPU thật sẽ có `cores // 4` worker cùng gọi chung
**một** instance `Predictor` VietOCR trên cùng một GPU. Torch không chặn việc này,
nhưng các thread sẽ tranh nhau và VRAM tăng theo số batch đồng thời.

Khi có GPU để đo thực tế, cân nhắc một trong hai:
- Hạ `_CPU_PHYSICAL` xuống 1–2 ở nhánh `_DEVICE == "cuda"`, hoặc
- Dùng `PARALLEL_DEVICES` để tạo recognizer riêng cho mỗi card

Chưa chỉnh vì chưa có GPU để benchmark — không nên đoán con số.

---

## Tóm tắt tác động — v5

| File | Thay đổi | Tác động |
|------|----------|----------|
| `module/ocr.py` | `resolve_vietocr_device()` thay `device = 'cpu'` hardcode | GPU được dùng thật; hết nghịch lý "có GPU chạy chậm hơn không GPU" |
| `module/ocr.py` | Log device của recognizer | Xác nhận được VietOCR nằm ở đâu |
| `api.py` | Log thêm `Device: {_DEVICE}` | Nhìn log là biết ngay cấu hình runtime |

---

---

# Lịch sử thay đổi — v6 (2026-08-14)

> Chủ đề: `/v1/ocr` route theo **từng trang** thay vì cả file, và trả **toạ độ**
> cho cả PDF text gốc. Luồng OCR (`module/ocr.py`) **không bị chạm**.

---

## 16. Khảo sát API thật của `pdf-inspector` — 3 phát hiện

Toàn bộ thiết kế v6 dựa trên việc đọc `pdf_inspector/__init__.pyi` (v0.2.6) và
chạy thử trên PDF 3 trang tự tạo (trang 0, 2 = text thuần; trang 1 = chỉ ảnh raster).

### a) `pdf_type` quá thô để route

PDF test có 2/3 trang là text đọc được sạch sẽ nhưng bị phân loại:

```
pdf_type = 'image_based'   confidence = 0.80
```

Stub cho biết có **4** giá trị: `'text_based'`, `'scanned'`, `'image_based'`, `'mixed'`.
Code v2–v5 chỉ chấp nhận `'text_based'` → mọi file `'mixed'` / `'image_based'`
(chính là PDF xuất từ Word có chèn logo, ảnh chữ ký, bảng chụp màn hình)
đều rơi xuống OCR **toàn bộ**, kể cả những trang text thuần.

### b) `pages_needing_ocr` — 3 hàm, 3 kiểu đánh index khác nhau

Đo trên cùng một file, trang cần OCR thật sự là **trang index 1**:

| Hàm | Trả về | Kết luận |
|-----|--------|----------|
| `process_pdf_bytes()` → `PdfResult` | `[1, 2, 3]` | **1-indexed**, stub *không* ghi chú |
| `classify_pdf_bytes()` → `PdfClassification` | `[0, 1, 2]` | 0-indexed (stub ghi rõ) |
| `extract_pages_markdown_bytes()` | `[2]` | 1-indexed (stub ghi rõ) |

Hai hệ quả:
- **Bug trong code cũ:** `/v1/pdf-inspect` trả thẳng `PdfResult.pages_needing_ocr`
  → 1-indexed, trong khi mọi endpoint khác dùng `pages[].page` 0-indexed. Lệch 1 trang.
- **`PdfResult.pages_needing_ocr` vô dụng để route:** nó trả `[1,2,3]` = "cả 3 trang",
  chỉ phản chiếu `pdf_type` toàn file chứ không đánh giá từng trang.

### c) Tín hiệu đáng tin duy nhất: `PageMarkdown.needs_ocr`

`extract_pages_markdown_bytes()` trả về cờ **riêng cho từng trang**, và khoanh đúng:

```
page=0  needs_ocr=False  md='## TRANG MOT: Hop dong bao hiem\n\nDong giua trang\n'
page=1  needs_ocr=True   md=''                    ← đúng trang ảnh
page=2  needs_ocr=False  md='## TRANG BA: Dieu khoan\n'
```

Cảnh báo: ngay **trong cùng object** này, `pages[].page` là 0-indexed
nhưng `pages_needing_ocr` là 1-indexed. Chỉ dùng `p.needs_ocr`, bỏ qua hẳn list kia.

---

## 17. Chọn thư viện lấy toạ độ — đo thực nghiệm, không đoán

`pdf-inspector` **có** `extract_text_with_positions_bytes()`, nên cả nó lẫn
`pdfplumber` đều lấy được toạ độ. Đo độ chính xác bằng cách render trang 300 dpi,
tìm **hộp bao pixel mực thật** làm chuẩn:

```
GROUND TRUTH (mực thật)        : x 248..1201     y 306..362
pdfplumber extract_text_lines  : x 248.1..1206.4  y 319.4..377.7   → top +13.4  bottom +15.7
pdf-inspector with_positions   : x 248.1..1206.4  y 292.4..350.7   → top -13.6  bottom -11.3
```

**Trục x giống nhau tuyệt đối** — cùng đọc một content stream. Trục y đều lệch
~13px @300dpi (≈0.4% chiều cao trang) nhưng **ngược chiều**, vì cả hai trả hộp
theo metric font chứ không phải hộp bao mực. Về độ chính xác: **hoà**.

Điểm quyết định là công sức và rủi ro:

| | pdf-inspector | pdfplumber |
|---|---|---|
| Gốc y | **baseline** (phải đoán hệ số ascent ≈0.75) | `top`/`bottom` sẵn, gốc từ đỉnh — khớp bbox OCR |
| Index trang | 1-indexed → phải `-1` | 0-indexed — khớp `pages[].page` |
| **Chiều cao trang để lật y** | **không có field nào cung cấp** | `page.height` sẵn cùng object |
| Lọc | phải bỏ `item_type == 'image'` | không cần |
| Trong project | chỉ dùng để phân loại | **đã dùng sẵn ở `_file_to_images()`** |

Dòng thứ 3 là điểm chốt: `pdf-inspector` không trả kích thước trang ở bất kỳ đâu,
mà không có chiều cao trang thì không lật được trục y — tức là **vẫn phải mở
pdfplumber** chỉ để lấy `page.height`. Đã dùng pdfplumber rồi thì gọi thêm
`extract_text_lines()` là miễn phí, lại tránh được hằng số suy đoán `0.75`
(đo trên đúng 1 font, 1 file — không đảm bảo đúng với font khác).

**Kết luận:** pdfplumber lấy toạ độ, pdf-inspector phân loại per-page. Mỗi thư viện làm phần nó mạnh.

---

## 18. `api.py` — `/v1/ocr` route theo từng trang

### Hai helper mới

```python
_PDF_RENDER_DPI = 300
_PT_TO_PX = _PDF_RENDER_DPI / 72.0      # pdfplumber trả point, bbox OCR là pixel
_CID_PATTERN = re.compile(r"\(cid *: *[0-9]+ *\)")

_classify_pdf_pages(data) -> dict | None
    # {page_0indexed: markdown} cho trang text đáng tin; None = OCR toàn bộ

_pdf_split_pages(data, text_pages) -> ({page: blocks}, [(page, PIL.Image)])
    # mở pdfplumber MỘT lần: trang text -> extract_text_lines -> blocks
    #                        trang còn lại -> render 300dpi cho OCR
```

### Luồng mới trong `/v1/ocr`

```
ext == "pdf" và _classify_pdf_pages() != None ?
  ├─ CÓ    → trang needs_ocr=False → pdfplumber → blocks (score 1.0)
  │          trang needs_ocr=True  → render 300dpi → LUỒNG OCR CŨ
  └─ KHÔNG → OCR toàn bộ, y hệt trước (ảnh, hoặc PDF không route được)
```

`_ocr_page()`, `_executor`, `score_threshold`, `ocr(np.array(img), 0)` — **giữ nguyên**.
Thay đổi duy nhất là *trang nào* được đưa vào luồng đó.

### `score: 1.0` cho block từ pdfplumber

Khác hẳn trường hợp hardcode `1.0` đã sửa ở mục 1 (v1) — ở đó `1.0` là giả,
che mất score thật của model. Ở đây text lấy trực tiếp từ file nên **đúng là
chắc chắn 100%**, không có khái niệm confidence.

### `source` chuyển xuống cấp trang

Một file giờ có thể vừa `pdfplumber` vừa `ocr`, nên một giá trị cho cả file
không còn diễn tả đúng:

```json
{
  "filename": "...",
  "source": "mixed",
  "pages": [
    { "page": 0, "source": "pdfplumber", "text": "...", "blocks": [...] },
    { "page": 1, "source": "ocr",        "text": "...", "blocks": [...] }
  ]
}
```

Đây là **thêm field**, không đổi field cũ → client đang đọc `pages[].blocks` vẫn chạy.
`source` cấp file = giá trị chung nếu đồng nhất, ngược lại `"mixed"`.
Có `source` theo trang thì debug được ngay: trang nào text kém, biết nó đi đường nào.

### Ba nhánh nghi ngờ đều rơi về OCR

| Tình huống | Xử lý |
|---|---|
| `_pdf_inspector` chưa cài / `extract_pages_markdown_bytes()` ném lỗi | OCR toàn bộ |
| Trang có `(cid:123)` trong markdown (font không nhúng bảng mã) | OCR trang đó |
| Trang `needs_ocr=False` nhưng `extract_text_lines()` không ra dòng nào | OCR trang đó |

Nguyên tắc: đường nhanh chỉ dùng khi chắc chắn, OCR là mặc định an toàn.
Đặc biệt nhánh 3 — **không tin cờ tuyệt đối, luôn kiểm tra kết quả thật**.

Pattern `(cid:` chính là pattern đã có sẵn trong `LayoutRecognizer.__is_garbage()`.

### Lợi ích phụ: RAM

Trước đây `_file_to_images()` render **toàn bộ** trang ở 300 dpi trước khi làm gì.
File 38 trang → 38 ảnh ~2480×3508 px trong RAM cùng lúc ≈ **1GB**
(đúng file đã gây lỗi ở mục 12, v4). Giờ chỉ render trang cần OCR —
file 38 trang mà 2 trang scan thì render 2 ảnh.

---

## 19. `api.py` — Sửa bug 1-indexed ở `/v1/pdf-inspect`

```python
# TRƯỚC — 1-indexed, lệch với pages[].page của mọi endpoint khác
"pages_needing_ocr": list(result.pages_needing_ocr or []),

# SAU
"pages_needing_ocr": [max(0, p - 1) for p in (result.pages_needing_ocr or [])],
```

`/v1/pdf-inspect` vẫn giữ nguyên vai trò endpoint thử nghiệm, không dùng trong luồng chính.

---

## Kiểm thử đã chạy

Gọi qua `TestClient` (HTTP thật, có lifespan + preload model):

**PDF 3 trang hỗn hợp** — routing đúng, dù `pdf_type` toàn file là `image_based`:
```
file source: mixed
page=0 source=pdfplumber  blocks=2   bbox=[248,319,1206,377] score=1.0 'TRANG MOT: Hop dong bao hiem'
page=1 source=ocr         blocks=0   (trang ảnh nhiễu, không có chữ — đúng)
page=2 source=pdfplumber  blocks=1
output=text → source: mixed, giữ được định dạng markdown '## TRANG MOT: ...'
```

**Ảnh JPG** — luồng OCR cũ nguyên vẹn:
```
status 200 (8.7s)  file source: ocr  page=0 blocks=90
score_threshold 0.5 → 90 blocks | 0.99 → 0 blocks   (bộ lọc vẫn hoạt động)
```

**`/v1/pdf-inspect`** — `pages_needing_ocr` giờ trả `[0, 1, 2]` (0-indexed).

---

## CHƯA kiểm chứng — cần bạn test bằng tài liệu thật

`needs_ocr` mới chỉ được xác minh trên PDF do **matplotlib** sinh ra, cấu trúc font
có thể khác PDF xuất từ **Word**. Cần chạy `/v1/ocr` trên vài file hợp đồng thật
rồi soi `source` của từng trang xem có khoanh đúng trang scan không.

Nếu `needs_ocr` tỏ ra không đáng tin trên file thật, có phương án dự phòng không
phụ thuộc `pdf-inspector`: dùng pdfplumber tự đánh giá — trang nào `extract_text()`
rỗng hoặc quá ngắn so với diện tích trang thì đẩy sang OCR.

---

## Tóm tắt tác động — v6

| File | Thay đổi | Tác động |
|------|----------|----------|
| `api.py` | `_classify_pdf_pages()` + `_pdf_split_pages()` | Route per-page thay cho `pdf_type` toàn file |
| `api.py` | `/v1/ocr` xử lý cả `'mixed'` / `'image_based'` | PDF Word có ảnh chèn: chỉ OCR trang scan, không OCR cả file |
| `api.py` | `output=blocks` có bbox cho PDF text gốc | Hover highlight dùng chung 1 code path cho mọi loại tài liệu |
| `api.py` | `source` theo từng trang | Biết trang nào đi đường nào, debug nhanh |
| `api.py` | Chỉ render trang cần OCR | RAM giảm mạnh với PDF nhiều trang |
| `api.py` | Fix `pages_needing_ocr` 1→0-indexed | `/v1/pdf-inspect` khớp quy ước chung |
| `module/` | **không chạm** | Luồng OCR giữ nguyên hoàn toàn |
