# OcrTraining — Fine-tune VietOCR cho DeepDoc

Môi trường **training** tách riêng cho pipeline OCR tiếng Việt DeepDoc + VietOCR.

Repo này tách từ project OCR đang chạy production, mục đích **chỉ để fine-tune
model recognition** — không thay thế project inference.

| Tài liệu | Nội dung |
|---|---|
| [training_guide.md](training_guide.md) | Hướng dẫn đầy đủ: cơ chế fine-tune, dataset, tham số, Colab step-by-step, đọc chỉ số |
| [train_log.md](train_log.md) | Nhật ký các lần train thật, kèm lỗi gặp phải và cách xử lý |
| [dataset_sample/README.md](dataset_sample/README.md) | Format dataset + hai cái bẫy của vietocr |
| [compat.py](compat.py) | Các bản vá bắt buộc để vietocr 0.3.13 chạy được — đọc trước khi debug |
| [upgrade.md](upgrade.md) | Lịch sử thay đổi của project OCR gốc (v1–v6) |
| [README_deepdoc_goc.md](README_deepdoc_goc.md) | README gốc của DeepDoc + VietOCR |

---

## Bắt đầu nhanh

```powershell
git clone https://github.com/NSVN-VuDinhDung/09.OcrTraining.git
cd 09.OcrTraining

# 1. Môi trường
python -m venv venv_train
.\venv_train\Scripts\python.exe -m pip install --upgrade pip
.\venv_train\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.\venv_train\Scripts\python.exe -m pip install --no-deps vietocr
.\venv_train\Scripts\python.exe -m pip install pyyaml gdown lmdb einops tqdm requests matplotlib opencv-python six scipy scikit-image imageio prefetch_generator huggingface_hub
.\venv_train\Scripts\python.exe -m pip install --no-deps imgaug albumentations
.\venv_train\Scripts\python.exe -m pip install "albucore==0.0.24" pydantic simsimd stringzilla

# 2. Model (~810MB, repo không chứa — xem mục dưới)
.\venv_train\Scripts\python.exe download_models.py

# 3. Dataset mẫu để thử
.\venv_train\Scripts\python.exe dataset_sample\make_sample_dataset.py --n 400

# 4. Train thử
.\venv_train\Scripts\python.exe train.py --name thu --iters 500
```

> **Vì sao `--no-deps` cho vietocr:** nó ghim `pillow==10.2.0`, `einops==0.2.0`,
> `gdown==4.4.0`. Các bản cũ này không có wheel cho Python 3.13+, pip sẽ cố build
> từ source rồi fail. Bản mới hoạt động bình thường.

---

## Model không nằm trong repo

Tổng ~810MB. Không đưa vào Git vì:

- `vgg_transformer.pth` **145MB, vượt giới hạn cứng 100MB** của GitHub — push bị từ chối thẳng
- Git LFS thì tiêu gần hết 1GB quota free; hết quota là **kẹt cả push lẫn clone**, mà đã đẩy lên LFS thì rất khó gỡ

### Cách lấy model

```powershell
.\venv_train\Scripts\python.exe download_models.py          # tải cái còn thiếu
.\venv_train\Scripts\python.exe download_models.py --check  # chỉ kiểm tra
.\venv_train\Scripts\python.exe download_models.py --all    # tải cả vgg_transformer (145MB)
```

| Thư mục | Nguồn | Dung lượng |
|---|---|---|
| `onnx/` | HuggingFace `InfiniFlow/deepdoc` | ~400MB |
| `vietocr/weight/vgg_seq2seq.pth` | `https://vocr.vn/data/vietocr/` | 86MB |

### Tải từ GitHub Release (dự phòng)

Nếu HuggingFace hoặc vocr.vn không truy cập được:

```powershell
.\venv_train\Scripts\python.exe download_models.py --release        # models-core.zip
.\venv_train\Scripts\python.exe download_models.py --release --all  # thêm models-optional.zip
```

Script tải zip rồi tự giải nén vào đúng thư mục. Hoặc tải tay từ
[Releases](https://github.com/NSVN-VuDinhDung/09.OcrTraining/releases) và giải
nén tại gốc repo — đường dẫn trong zip đã là `onnx/...` và `vietocr/weight/...`.

| Zip | Dung lượng | Nội dung |
|---|---|---|
| `models-core.zip` | 466 MB | `onnx/` đầy đủ + `vgg_seq2seq.pth` — **đủ để pipeline chạy** |
| `models-optional.zip` | 309 MB | `vgg_transformer.pth`, `transformerocr.pth`, `cnn/encoder/decoder.onnx` (cho `ocr_onnx.py`) |

Release asset **không tính vào LFS quota** (giới hạn 2GB/file), nên đây cũng là
chỗ để lưu **weight bạn tự fine-tune** — sản phẩm riêng, không tải lại từ đâu được.

### Tạo và upload zip Release

Tạo zip:

```powershell
.\venv_train\Scripts\python.exe make_release_zips.py
```

Upload bằng `gh` CLI (tag phải khớp `RELEASE_TAG` trong `download_models.py`):

```powershell
gh release create models-v1 `
    release_assets\models-core.zip `
    release_assets\models-optional.zip `
    --title "Model weights v1" `
    --notes "onnx/ (InfiniFlow/deepdoc) + vietocr/weight/. Giải nén tại gốc repo."
```

Thêm file vào release đã có: `gh release upload models-v1 <file>`
(thêm `--clobber` để ghi đè file trùng tên).

Hoặc qua web: **Releases → Draft a new release** → đặt tag `models-v1` → kéo hai
file zip vào ô *Attach binaries* → **Publish release**.

---

## Train

```powershell
# CPU (chậm — chỉ để kiểm tra pipeline)
.\venv_train\Scripts\python.exe train.py --name v1 --iters 500

# Colab / máy có GPU
python train.py --name v1 --iters 15000 --device cuda:0 --num_workers 2

# Dataset của bạn
python train.py --name v2 --data_root D:\dataset_v1 --iters 15000 --device cuda:0

# Train tiếp từ kết quả lần trước (resume)
python train.py --name v3 --pretrain train_output\vgg_seq2seq_v2.pth --iters 8000
```

`train.py --help` để xem toàn bộ tham số. Ý nghĩa từng cái:
[training_guide.md](training_guide.md) mục 7.

Script tự làm 4 việc mà chạy `Trainer` trần không có:

1. `import compat` — vá các lỗi chặn của vietocr (xem dưới)
2. Xoá LMDB cũ — tránh train nhầm dữ liệu lần trước
3. **Đo baseline trước khi train** — không có mốc này thì accuracy sau đó vô nghĩa
4. Ghi `summary_<name>.json` + log kèm mốc thời gian

## Đầu ra

```
train_output/
├── vgg_seq2seq_<name>.pth   ← weight (chỉ ghi khi accuracy vượt kỷ lục)
├── config_<name>.yml        ← config đã dùng, để tái lập
├── train_<name>.log         ← log loss/accuracy từng mốc
└── summary_<name>.json      ← baseline vs kết quả, thời gian chạy
```

### Dùng weight mới trong project OCR

1. Copy `vgg_seq2seq_<name>.pth` sang `vietocr/weight/` của project inference
2. Sửa **một dòng** trong `module/ocr.py`, class `TextRecognizer`:
   ```python
   config['weights'] = r"vietocr\weight\vgg_seq2seq_<name>.pth"
   ```

**Benchmark A/B trước khi thay.** Fine-tune có thể làm tệ đi. Chạy `t_ocr.py` với
weight cũ và mới trên cùng bộ tài liệu **không nằm trong tập train**, gồm cả loại
đã fine-tune lẫn loại khác, rồi so sánh. Xem
[training_guide.md](training_guide.md) mục 10.2.

---

## Tạo dataset từ tài liệu thật

```powershell
# Cắt dòng + sinh label nháp bằng chính pipeline detection
.\venv_train\Scripts\python.exe tools\export_ocr_lines.py --inputs D:\tailieu --output_dir D:\dataset_v1

# Mở D:\dataset_v1\review.txt (sắp theo score tăng dần), sửa dòng sai trong
# annotation_raw.txt, rồi chia train/val:
.\venv_train\Scripts\python.exe tools\export_ocr_lines.py --split D:\dataset_v1\annotation_raw.txt
```

Dataset synthetic trong `dataset_sample/` **chỉ để hiểu format và test pipeline**,
không dùng fine-tune thật.

---

## compat.py — các bản vá bắt buộc

`vietocr==0.3.13` (2021) không chạy được trên môi trường hiện đại nếu không vá.
Tất cả đều đã kiểm chứng thực tế, chi tiết trong [train_log.md](train_log.md).

| Vấn đề | Triệu chứng | Cách vá |
|---|---|---|
| numpy 2.x gỡ `np.fromstring` | `ValueError: The binary mode of fromstring is removed` | chuyển sang `np.frombuffer` |
| numpy 2.x gỡ `np.sctypes` (imgaug) | `AttributeError: np.sctypes was removed` | dựng lại dict |
| LMDB `map_size=1TB` trên Windows | `lmdb.Error: Insufficient system resources` | chặn trần 4GiB |
| **Sampler vứt batch không đầy** | val rỗng → `UnboundLocalError: 'prob'`, hoặc **im lặng train trên 9% dữ liệu** | `Trainer.data_gen` dùng `batch_sampler` |

Cái thứ tư nguy hiểm nhất vì **thường không crash**. Đo trên dataset 400 dòng,
`batch_size=32`: train giữ 32/340 mẫu (9%), val giữ 0/60 (0%).

Trên Colab (Python 3.11/3.12) chỉ cần `pip install "numpy<2"` là xử lý được hai
vấn đề numpy; hai vấn đề còn lại vẫn cần `compat.py`.

---

## Nguồn gốc và giấy phép

Pipeline OCR gốc từ [DeepDoc](https://github.com/infiniflow/ragflow/blob/main/deepdoc/README.md)
(InfiniFlow, Apache-2.0); bản thay recognition bằng VietOCR từ
[hoaivannguyen/deepdoc_vietocr](https://github.com/hoaivannguyen/deepdoc_vietocr).
Header license Apache-2.0 giữ nguyên trong các file gốc.

- VietOCR: https://github.com/pbcquoc/vietocr
- PP-OCRv5: https://arxiv.org/html/2507.05595v1
- YOLOv10: https://arxiv.org/pdf/2405.14458
