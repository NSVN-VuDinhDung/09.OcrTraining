# Nhật ký training

Ghi theo thứ tự thời gian: làm gì, gặp gì, xử lý thế nào, kết quả ra sao.
Mọi số liệu trong file này là **đo thật**, không suy đoán.

Môi trường:

| | |
|---|---|
| Máy | Windows 11 Enterprise 10.0.22631, i5-12500 (12 logical cores), **không có GPU** |
| Python | 3.14.3 (bản duy nhất trên máy) |
| Torch | 2.13.0+cpu |
| numpy | 2.4.4 |
| vietocr | 0.3.13 |
| venv | `venv_train/` — tách hoàn toàn khỏi venv inference của project 19 |

---

## 2026-08-14 — Phiên 1: dựng môi trường và train thử

### Mục tiêu

Xác minh **toàn bộ pipeline train chạy được đầu-cuối** trên môi trường hiện đại,
và ghi lại đúng những gì phải sửa. Không nhằm ra model dùng được — máy không có
GPU, dataset là synthetic.

### 10:45 — Tách project

Copy từ `19.DeepdocOcr/deepdoc_vietocr` sang `20.TrainingOcr`, **không copy**
`venv/`, `log/`, `layouts_outputs/`, `__pycache__/`, `.git/`.

Model (`onnx/` ~400MB + `vietocr/weight/` ~403MB) copy vào local nhưng
**loại khỏi git** — xem [README.md](README.md) mục "Model không nằm trong repo".

### 10:52 — Dựng venv training

**Vấn đề 1: numpy 1.x không cài được trên Python 3.14**

Kế hoạch ban đầu là `pip install "numpy<2"` (cách sạch nhất, và vẫn là cách
khuyến nghị trên Colab). Nhưng numpy 1.26 chỉ hỗ trợ tới Python 3.12, không có
wheel cho 3.14. Máy chỉ có Python 3.14.

→ Chuyển sang **vá runtime**, tạo [compat.py](compat.py).

**Vấn đề 2: `pip install vietocr` fail**

```
KeyError: '__version__'
ERROR: Failed to build 'pillow' when getting requirements to build wheel
```

`vietocr` ghim `pillow==10.2.0`, `einops==0.2.0`, `gdown==4.4.0` — không có wheel
cho Python 3.14, pip cố build từ source rồi vỡ.

→ `pip install --no-deps vietocr` rồi cài dependency bằng bản mới. Hoạt động bình thường.

**Vấn đề 3: `albucore` lệch phiên bản**

```
ModuleNotFoundError: No module named 'numkong'
```

`albucore` 0.2.13 (bản mới nhất) cần `numkong`, nhưng `albumentations` 2.0.8 yêu
cầu `albucore==0.0.24`.

→ Ghim `albucore==0.0.24`.

### 11:00 — Lỗi chặn #1: numpy 2 gỡ `np.sctypes`

```
File "venv_train\Lib\site-packages\imgaug\imgaug.py", line 45
    NP_FLOAT_TYPES = set(np.sctypes["float"])
AttributeError: `np.sctypes` was removed in the NumPy 2.0 release.
```

Xảy ra ngay tại `from vietocr.model.trainer import Trainer` — chưa làm gì đã chết.
`imgaug` được `trainer.py` import ở top-level nên **không thể tránh** bằng cách
tắt augmentation.

→ Dựng lại `np.sctypes` theo đúng nội dung nó có ở numpy 1.x.

Đồng thời vá `np.bool/np.float/np.int/np.complex` (alias bị gỡ ở numpy 1.24).
**Cố ý không vá `np.object`/`np.str`** — numpy 2 dự định định nghĩa lại chúng
thành scalar type, chỉ riêng `hasattr()` đã kích hoạt `FutureWarning`, mà stack
này không cần chúng.

### 11:03 — Lỗi chặn #2: numpy 2 gỡ `np.fromstring`

```
ValueError: The binary mode of fromstring is removed, use frombuffer instead
```

Hai chỗ dùng:
- `vietocr/loader/dataloader.py:71` `get_bucket()`
- `vietocr/tool/create_dataset.py:13` `checkImageIsValid()`

→ Chuyển chế độ binary sang `np.frombuffer`, giữ nguyên chế độ text (có `sep`).

### 11:03 — Ghi nhận: `OneCycleLR` chia cho 0 khi `iters` quá nhỏ

Chạy thử `--iters 10` để đo tốc độ:

```
File "torch\optim\lr_scheduler.py", line 2586, in get_lr
    pct = (step_num - start_step) / (end_step - start_step)
ZeroDivisionError: division by zero
```

`trainer.py:65` gọi `OneCycleLR(total_steps=iters, pct_start=0.1)`. Với `iters=10`,
số bước warm-up = `floor(10 * 0.1) - 1 = 0` → chia cho 0.

**Không vá** — chỉ cần `iters >= 50`. Ghi lại để không mất thời gian debug.

### 11:04 — Lỗi chặn #3: LMDB `map_size` 1TB trên Windows

```
lmdb.Error: valid_smoke500: Insufficient system resources exist to
            complete the requested service.
```

`create_dataset.py:47` gọi `lmdb.open(path, map_size=1099511627776)` — **1 TiB**.
Trên Linux đây là sparse mapping gần như miễn phí. Trên Windows, LMDB tạo ngay
file có kích thước khai báo:

```
$ ls -la train_smoke500
-rw-r--r-- 1 dungvd1 1099511627776 data.mdb      <- khai báo 1TB
$ du -sh train_smoke500
3.8M                                              <- thực tế 3.8MB
```

Map thứ nhất (train) tạo được, map thứ hai (valid) thì hết tài nguyên. Trainer
**luôn** tạo 2 dataset → luôn vỡ trên Windows.

→ Chặn trần `map_size` ở 4 GiB (đủ cho vài trăm nghìn ảnh dòng).

### 11:10 — Ổ C: đầy 100%

Phát hiện `C:` còn **464MB/200GB**. Dọn cache tái tạo được: pip cache 543M,
NuGet v3-cache 451M, CrashDumps 64M, Temp (TFSTemp / DockerDesktopUpdates /
Temporary ASP.NET Files) ~1.3G → **11GB trống**.

Cố ý **giữ** `~/.cache/torch` (549M) — đó là weight VGG19 pretrained mà
`TextRecognizer` tải về do `config['cnn']['pretrained'] = True`; xoá thì lần
chạy OCR sau phải tải lại.

Training ghi toàn bộ vào ổ D nên không ảnh hưởng.

### 13:15 — Lỗi chặn #4: sampler vứt batch không đầy ← NGHIÊM TRỌNG NHẤT

```
File "vietocr\model\trainer.py", line 190, in predict
    return pred_sents, actual_sents, img_files, prob
UnboundLocalError: cannot access local variable 'prob' where it is not
                   associated with a value
```

Thông báo lỗi **không liên quan gì** tới nguyên nhân thật. `prob` chỉ được gán
trong vòng lặp `for batch in self.valid_gen` → vòng lặp chạy **0 lần** → tập
validation rỗng.

Nguyên nhân, `vietocr/loader/dataloader.py:137`:

```python
batches = [_ for _ in batches if len(_) == self.batch_size]
```

Ảnh được gom "cụm" theo chiều rộng sau resize về cao 32px, để mỗi batch có ảnh
cùng bề rộng. Nhưng dòng trên **chỉ giữ batch đầy đúng `batch_size`** — mọi phần
dư của mỗi cụm bị vứt, không cảnh báo.

Đo trên dataset 400 dòng:

| batch_size | train giữ được | valid giữ được |
|---|---|---|
| **32** (mặc định) | **32/340 = 9%** | **0/60 = 0%** |
| 16 | 80/340 = 23% | 0/60 = 0% |
| 8 | 232/340 = 68% | 8/60 = 13% |
| 4 | 300/340 = 88% | 28/60 = 46% |

Kích thước cụm lớn nhất: train `[58, 20, 16, 15, 15, 15, 14, 14, ...]` (32 cụm),
valid `[11, 6, 5, 4, 4, 4, 3, 3, ...]` (24 cụm).

**Điều nguy hiểm nhất không phải crash.** Nếu tình cờ có một cụm đầy 32 ảnh, bạn
train ngon lành trên 9% dữ liệu, log vẫn đẹp, và không bao giờ biết. Ảnh dòng
thật có bề rộng rất đa dạng nên cụm luôn nhỏ.

**Lần vá đầu của tôi SAI.** Tôi chỉ bỏ bộ lọc đi, và gặp:

```
ValueError: setting an array element with a sequence. The requested array has
            an inhomogeneous shape after 3 dimensions.
            The detected shape was (32, 3, 32) + inhomogeneous part.
```

Nguyên nhân sâu hơn: `trainer.py:315` xây `DataLoader(dataset, batch_size=32,
sampler=ClusterRandomSampler(...))`. Với `sampler` + `batch_size`, DataLoader
**tự cắt** luồng index thành khối 32. Bộ lọc kia tồn tại để mọi khối do sampler
phát ra đều đúng 32, nhờ đó ranh giới cắt trùng khít ranh giới cụm. Bỏ bộ lọc
làm ranh giới lệch → một batch trộn ảnh từ nhiều cụm khác bề rộng → collate vỡ.

**Cách vá đúng:** chuyển sang `batch_sampler`. DataLoader dùng nguyên các batch
ta đưa, không cắt lại. Mỗi batch vẫn lấy từ đúng một cụm nên đồng nhất bề rộng,
và batch lẻ được giữ. Vá bằng cách thay `Trainer.data_gen`.

Kiểm chứng sau khi vá:

```
train_smoke500: 340 mẫu -> sampler trả về 340 (100%)
valid_smoke500:  60 mẫu -> sampler trả về  60 (100%)
```

### 13:29 – 13:52 — TRAIN THÀNH CÔNG

```
python train.py --name smoke500 --iters 500 --print_every 25 --valid_every 100
```

| Tham số | Giá trị |
|---|---|
| Dataset | 340 train / 60 val, synthetic, 8 font, 3 mức nhiễu |
| iters | 500 |
| batch_size | 32 |
| max_lr | 1e-4 (fine-tune, thấp hơn mặc định 1e-3 10 lần) |
| device | cpu |
| pretrain | `vietocr/weight/vgg_seq2seq.pth` |

**Kết quả:**

| Mốc | valid loss | acc full seq | acc per char |
|---|---|---|---|
| **baseline** (weight gốc) | — | **0.5833** | **0.8534** |
| iter 100 | 0.828 | 0.8000 | 0.9929 |
| iter 200 | 0.822 | 0.8167 | 0.9943 |
| iter 300 | 0.821 | 0.8500 | 0.9938 |
| iter 400 | 0.820 | **0.8667** | 0.9949 |
| iter 500 | 0.821 | 0.8667 | 0.9947 |
| **thay đổi** | | **+0.2833** | **+0.1414** |

Thời gian: **1341s (22 phút)**, `2.66 s/iter` trên CPU.
Weight xuất ra: `train_output/vgg_seq2seq_smoke500.pth` (86MB).

Đo lại lần hai để kiểm tra tái lập: baseline `0.5833 / 0.8534`,
sau train `0.8667 / 0.9949` — khớp chính xác, con số là deterministic.

### 13:52 — Bug nhỏ trong `train.py` của tôi

```
TypeError: Object of type float32 is not JSON serializable
when serializing dict item 'per_char'
```

`compute_accuracy()` trả `np.float32`. Training đã xong và weight đã ghi, chỉ
bước ghi `summary_*.json` vỡ. Đã sửa bằng `float()`. File JSON được tạo lại
bằng cách đo lại thật (5s/lần), không phải điền tay.

---

## Đọc kết quả thế nào — và điều KHÔNG được kết luận

### Pipeline: xác minh xong

Toàn bộ đường train chạy được đầu-cuối. Loss giảm (1.017 → 0.771), accuracy
tăng đơn điệu qua 5 mốc validation, LR đi đúng hình one-cycle (warm-up
5.35e-05 → đỉnh 1.00e-04 → anneal 1.62e-09), weight được ghi ra đúng lúc
accuracy vượt kỷ lục. Không còn lỗi chặn nào.

### `+28%` KHÔNG có nghĩa model tốt hơn

Đây là điểm quan trọng nhất của phiên này.

Model được train trên **ảnh synthetic** — render font Windows lên nền trắng,
thêm nhiễu giả. Nó học rất nhanh phân bố đó (chỉ 500 iter đã +28% full-seq),
nhưng phân bố đó **không giống ảnh scan thật**: không có bóng giấy, vệt mực, nhoè
do nén JPEG nhiều lần, méo do đặt lệch máy scan, chữ dính nhau.

Nói cách khác, `0.8667` là "đọc tốt ảnh do chính script này sinh ra", không phải
"đọc tốt tài liệu của bạn". Rất có thể weight này đọc tài liệu thật **kém hơn**
weight gốc — đúng hiện tượng catastrophic forgetting mô tả ở
[training_guide.md](training_guide.md) mục 3.3.

**Không dùng `vgg_seq2seq_smoke500.pth` cho production.**

### Dấu hiệu đã tới giới hạn của dataset này

`valid loss` gần như phẳng từ iter 200: `0.822 → 0.821 → 0.820 → 0.821`,
và ở iter 500 accuracy dừng lại (0.8667, per_char giảm nhẹ 0.9949 → 0.9947).
Với 340 mẫu thì model đã hút hết thông tin có thể hút. Train thêm chỉ dẫn tới
overfit.

Đây chính là hình dạng mà mục 9.3 của guide gọi là "bắt đầu chạm ngưỡng" — và
là lý do phải theo `valid loss` chứ không chỉ theo accuracy.

---

## Kết luận phiên 1

**Đạt được:**

- Môi trường train dựng xong, tách khỏi venv inference
- 4 lỗi chặn của vietocr đã tìm ra và vá, tất cả xác minh bằng lỗi thật ([compat.py](compat.py))
- Pipeline chạy đầu-cuối, có số liệu thật
- `train.py` có sẵn kiểm tra dataset, xoá LMDB, đo baseline, ghi summary
- Đo được tốc độ CPU thực tế: **2.66 s/iter** → 15.000 iter ≈ **11 giờ** trên máy này

**Chưa đạt (theo đúng dự kiến):**

- Không có model dùng được. Cần GPU + dataset thật.
- Chưa test đường GPU (`--device cuda:0`) vì máy không có GPU

**Việc tiếp theo, theo thứ tự:**

1. **Chẩn đoán trước** — chạy `t_ocr.py` trên tài liệu thật, xem ảnh có bounding
   box, xác định lỗi nằm ở detection hay recognition
   ([training_guide.md](training_guide.md) mục 1). Nếu là detection thì fine-tune
   recognition vô ích.
2. **Thử cách rẻ** — tăng DPI, nới `unclip_ratio`, tăng `pad_y`. Nhiều khi đủ.
3. **Tạo dataset thật** — `tools/export_ocr_lines.py`, sửa tay, 1.000–3.000 dòng
4. **Train trên Colab** — T4, `--iters 15000 --device cuda:0`, ước 1,5–3 giờ
5. **Benchmark A/B** trên tài liệu ngoài tập train, gồm cả trong lẫn ngoài domain
6. **Chỉ khi tốt hơn thật** mới thay weight trong `module/ocr.py`

---

## Mẫu ghi cho phiên sau

```
## YYYY-MM-DD — Phiên N: <mục tiêu>

### Cấu hình
dataset / iters / batch_size / max_lr / device / pretrain

### Kết quả
| Mốc | valid loss | acc full seq | acc per char |
baseline / các mốc / thay đổi
thời gian, s/iter, file weight

### Vấn đề gặp phải
### Kết luận — có thay weight production không, vì sao
```
