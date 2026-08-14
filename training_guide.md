# Hướng dẫn fine-tune VietOCR cho tiếng Việt

> Tài liệu step-by-step để tăng độ chính xác OCR tiếng Việt của project này,
> training trên **Google Colab** (không cần máy chủ GPU).
>
> Mọi thông số trong tài liệu đều **đọc trực tiếp từ package `vietocr` đã cài
> trong `venv/`** và kiểm chứng bằng dataset mẫu, không phải chép từ tutorial.

---

## Mục lục

1. [Trước khi train: xác định lỗi nằm ở đâu](#1)
2. [Cơ chế: hai model độc lập trong pipeline](#2)
3. [Cơ chế fine-tuning — vì sao hoạt động và vì sao dễ hỏng](#3)
4. [Kiến trúc `vgg_seq2seq`](#4)
5. [Dataset: format và cách tạo](#5)
6. [Tìm dataset trên HuggingFace](#6)
7. [Toàn bộ tham số cấu hình](#7)
8. [Google Colab — step by step](#8)
9. [Đọc chỉ số: train thành công hay thất bại](#9)
10. [Đưa weight về project](#10)
11. [Các lỗi đã biết của vietocr](#11)
12. [Xử lý sự cố](#12)

---

<a name="1"></a>
## 1. Trước khi train: xác định lỗi nằm ở đâu

**Đừng bỏ qua bước này.** Fine-tune sai chỗ là mất vài ngày công vô ích.

Chạy pipeline hiện tại trên 10–20 trang tài liệu thật:

```powershell
.\venv\Scripts\python.exe t_ocr.py --inputs=D:\tailieu --output_dir=D:\ketqua
```

Mở ảnh có vẽ bounding box trong `D:\ketqua`, đối chiếu với file `.txt`:

| Bạn thấy | Lỗi ở | Việc cần làm |
|---|---|---|
| Box ôm trọn dòng chữ, nhưng **text sai** | **Recognition** | ✅ Fine-tune VietOCR — đọc tiếp tài liệu này |
| Box **cắt mất** phần trên/dưới chữ | Detection | Tăng `pad_y` trong `expand_box()` |
| Box **dính 2 dòng** làm 1, hoặc **bỏ sót** dòng | Detection | Chỉnh `DBPostProcess`, xem mục 1.2 |
| Chữ mờ nhoè ngay trên **ảnh gốc** | Chất lượng đầu vào | Tăng DPI, xem mục 1.2 |

Với tài liệu tiếng Việt, khoảng 80% trường hợp là lỗi recognition — nhất là
sai dấu thanh (`BÀO HIẾM` thay vì `BẢO HIỂM`).

### 1.2. Ba cách rẻ nên thử TRƯỚC khi train

Fine-tune tốn vài ngày. Ba cách sau tốn vài phút và nhiều khi giải quyết được
phần lớn vấn đề:

**a) Tăng DPI render PDF** — rẻ nhất, hay bị bỏ qua nhất.
[api.py](api.py) đang dùng 300 dpi (`_PDF_RENDER_DPI`), CLI dùng 216 dpi
(`72*3` trong `module/__init__.py`). Ảnh vào nét hơn thì **cả detection lẫn
recognition đều tốt lên**, không phải train gì.

**b) Nới `unclip_ratio`** trong [module/ocr.py](module/ocr.py) (`DBPostProcess`):
```python
"box_thresh": 0.5,      # hạ xuống -> bắt được dòng mờ hơn, nhưng tăng box rác
"unclip_ratio": 1.5,    # tăng lên  -> box nở rộng hơn, đỡ cắt chữ
```

**c) Tăng `pad_y`** trong `expand_box()` — đã làm ở v4 (`pad_y=10`) và có hiệu quả rõ.

> Tiếng Việt có ký tự 2 tầng dấu (`ắ ổ ợ ữ`). Detection của PaddleOCR khớp bbox
> sát vùng text, mà VietOCR lại resize crop về cao 32px — dấu tầng trên bị cắt
> thì model **không thể** đọc đúng, dù có fine-tune giỏi đến đâu. Nó không nhìn
> thấy cái dấu đó.

---

<a name="2"></a>
## 2. Cơ chế: hai model độc lập trong pipeline

```
Ảnh trang
   │
   ├─► [1] TEXT DETECTION — DBNet (PaddleOCR), ONNX
   │        onnx/det.onnx
   │        Tìm toạ độ từng dòng chữ -> danh sách bbox 4 điểm
   │
   ├─► sorted_boxes()      sắp xếp trên->dưới, trái->phải
   ├─► expand_box()        nới bbox 3px ngang / 10px dọc
   ├─► get_rotate_crop_image()   cắt + nắn phối cảnh -> ảnh 1 dòng
   │
   └─► [2] TEXT RECOGNITION — VietOCR vgg_seq2seq, PyTorch
            vietocr/weight/vgg_seq2seq.pth
            Ảnh 1 dòng -> chuỗi ký tự + xác suất
```

| | Detection | Recognition |
|---|---|---|
| Model | DBNet (PP-OCR) | VietOCR `vgg_seq2seq` |
| Định dạng | ONNX (đã đóng băng) | **PyTorch `.pth`** |
| Fine-tune | ✗ Rất khó — không có weight gốc, phải train lại bằng PaddleOCR rồi export ONNX | ✅ **Dễ — mục tiêu của tài liệu này** |
| Data cần | ảnh trang + toạ độ box | ảnh dòng + text |

Đây là lý do tài liệu này chỉ nói về recognition: nó là phần **vừa dễ fine-tune
nhất, vừa là nguồn lỗi lớn nhất** với tiếng Việt.

---

<a name="3"></a>
## 3. Cơ chế fine-tuning — vì sao hoạt động và vì sao dễ hỏng

### 3.1. Fine-tune khác train from scratch thế nào

**Train from scratch:** khởi tạo trọng số ngẫu nhiên, học mọi thứ từ đầu —
từ "nét cong này là chữ o" đến "sau chữ 'Hợp' thường là chữ 'đồng'".
Cần hàng triệu mẫu và vài tuần GPU.

**Fine-tune (transfer learning):** nạp `vgg_seq2seq.pth` đã được train sẵn trên
lượng lớn dữ liệu tiếng Việt, rồi train **tiếp** với learning rate thấp trên
data của bạn.

Model đã biết đọc tiếng Việt rồi. Bạn chỉ đang dạy nó quen với:
- **font** trong tài liệu của bạn
- **chất lượng scan** (nhiễu, mờ, nghiêng, bóng giấy) của máy scan của bạn
- **từ vựng** đặc thù (thuật ngữ bảo hiểm, tên riêng, mã số)

Vì thế cần ít data hơn hàng nghìn lần: **1.000–3.000 dòng thường là đủ.**

### 3.2. Cơ chế bên trong: gradient nhỏ, không phá cái đã học

Learning rate quyết định mỗi bước học thay đổi trọng số bao nhiêu:

```
trọng_số_mới = trọng_số_cũ − learning_rate × gradient
```

- LR cao (`1e-3`, dùng khi train from scratch): mỗi bước nhảy xa, học nhanh
  nhưng **xoá sạch** kiến thức cũ.
- LR thấp (`1e-4` trở xuống, dùng khi fine-tune): nhích từng chút, **điều chỉnh**
  cái đã có thay vì viết đè.

### 3.3. Catastrophic forgetting — rủi ro lớn nhất

Nếu train 2.000 ảnh hợp đồng bảo hiểm với LR cao và nhiều iteration, model sẽ:
- đọc hợp đồng bảo hiểm **rất tốt**
- đọc **mọi thứ khác tệ đi rõ rệt**

Model không "thêm" kiến thức, nó **dịch chuyển** trọng số về phía data mới.

Ba cách phòng, dùng đồng thời:

1. **LR thấp** — `max_lr = 1e-4` thay vì `1e-3` mặc định
2. **Ít iteration** — 10.000–20.000, không phải 100.000
3. **Validation đa dạng** — tập val phải có cả data đúng domain lẫn data chung.
   Nếu val chỉ toàn hợp đồng bảo hiểm, bạn sẽ thấy accuracy tăng đều
   trong khi model đang hỏng dần ở mọi thứ khác mà không biết.

### 3.4. Vì sao PHẢI có tập validation riêng

Tập val là ảnh model **chưa từng thấy** khi train. Nếu để lẫn ảnh train vào val,
mọi con số accuracy đều là ảo — model chỉ đang đọc thuộc lòng.

Đây là cơ chế duy nhất phát hiện **overfitting**: train loss giảm đều
nhưng val accuracy đứng yên hoặc giảm.

---

<a name="4"></a>
## 4. Kiến trúc `vgg_seq2seq`

```
Ảnh dòng chữ (H=32px cố định, W co giãn 32..512px)
   │
   ├─► CNN backbone: VGG19-BN
   │     Trích đặc trưng thị giác. Nén chiều cao xuống 1, giữ chiều ngang
   │     -> chuỗi vector đặc trưng theo trục ngang
   │
   ├─► Encoder: GRU 2 chiều (hidden 256)
   │     Đọc chuỗi đặc trưng, tạo "hiểu biết" về toàn dòng
   │
   └─► Decoder: GRU + Attention (hidden 256, embed 256)
         Sinh TỪNG ký tự một. Mỗi bước, attention quyết định
         "nhìn vào vùng nào của ảnh" để đoán ký tự tiếp theo
   │
   └─► Chuỗi ký tự + xác suất (return_prob=True)
```

Điểm quan trọng về mặt thực hành:

- **Input là ảnh MỘT DÒNG**, không phải cả trang. Chính là thứ
  `get_rotate_crop_image()` trả ra trong pipeline của bạn.
- **Chiều cao ép về 32px** (`image_height: 32`). Ảnh dòng cao 60px cũng bị nén
  xuống 32 — nên chi tiết dấu thanh rất dễ mất nếu bbox cắt sát.
- Chiều rộng gom theo **bucket** (`ClusterRandomSampler`) để các ảnh cùng batch
  có độ rộng gần nhau, giảm padding thừa.

### Vì sao chọn seq2seq mà không phải transformer

`module/ocr.py` đang dùng `vgg_seq2seq`, và có sẵn dòng comment để đổi sang
`vgg_transformer`. Theo README của project: transformer **chậm hơn rất nhiều
mà chính xác không hơn bao nhiêu**. Fine-tune cứ bám `vgg_seq2seq`.

---

<a name="5"></a>
## 5. Dataset: format và cách tạo

### 5.1. Format bắt buộc

Xem thư mục [dataset_sample/](dataset_sample/) — nó chính là ví dụ chạy được.

```
dataset_v1/
├── images/
│   ├── line_0000.jpg      ảnh MỘT DÒNG, ví dụ 521×44 px
│   └── ...
├── annotation_train.txt
└── annotation_val.txt
```

`annotation_train.txt` — đường dẫn + **TAB** + label:

```
images/line_0000.jpg	Hợp đồng bảo hiểm nhân thọ số 0123456789
images/line_0001.jpg	Điều 1: Phạm vi bảo hiểm và quyền lợi được hưởng
```

Bắt buộc: **UTF-8**, xuống dòng **LF**, ngăn cách bằng **TAB**, đường dẫn tương
đối so với `data_root`, label **không chứa TAB**.

Chi tiết đầy đủ + hai cái bẫy: [dataset_sample/README.md](dataset_sample/README.md).

### 5.2. Bảng vocab — kiểm tra trước khi train

VietOCR chỉ sinh được ký tự nằm trong `config['vocab']`. Bảng mặc định gồm
toàn bộ chữ cái tiếng Việt có dấu, chữ số, và các dấu câu ASCII:

```
aAàÀảẢãÃáÁạẠăĂằẰẳẲẵẴắẮặẶâÂầẦẩẨẫẪấẤậẬbBcCdDđĐeE...0123456789!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~ 
```

**Ký tự KHÔNG có trong bảng này** sẽ bị bỏ khi encode, khiến label sai lệch:
- gạch dài `—` (em dash) và `–` (en dash) — rất hay gặp trong văn bản Word
- nháy cong `"` `"` `'` `'` — Word tự động thay nháy thẳng bằng nháy cong
- `…` `•` `§` `°` `™`

Cách xử lý, chọn một:
1. **Chuẩn hoá label** — thay `—` → `-`, `"` → `"` trước khi ghi annotation (khuyên dùng)
2. **Mở rộng vocab** — thêm ký tự vào `config['vocab']`. Nhưng làm vậy sẽ **đổi
   kích thước lớp output**, khiến weight pretrained không nạp được đúng phần đó
   (`load_weights` sẽ in `missmatching shape` rồi bỏ qua). Chỉ làm khi thực sự cần.

Đoạn kiểm tra nhanh, chạy trước khi train:

```python
from vietocr.tool.config import Cfg
vocab = Cfg.load_config_from_name('vgg_seq2seq')['vocab']
missing = set()
for line in open('annotation_train.txt', encoding='utf-8'):
    missing |= {c for c in line.rstrip('\n').split('\t')[1] if c not in vocab}
print('Ký tự ngoài vocab:', missing or 'không có (tốt)')
```

### 5.3. Cách tốt nhất: cắt dataset từ tài liệu THẬT

Với use case của bạn, dataset tự tạo gần như chắc chắn giá trị hơn dataset public.
Bạn đã có sẵn mọi thứ cần thiết: tài liệu thật đúng domain, pipeline detection
chạy tốt để cắt dòng tự động, và model hiện tại để tạo label nháp.

Project có sẵn script [tools/export_ocr_lines.py](tools/export_ocr_lines.py):

```powershell
# Bước 1 — cắt dòng + sinh label nháp
.\venv\Scripts\python.exe tools\export_ocr_lines.py `
    --inputs D:\tailieu_that --output_dir D:\dataset_v1

# Bước 2 — mở D:\dataset_v1\review.txt (đã sắp xếp theo score TĂNG DẦN)
#          sửa các dòng sai trong D:\dataset_v1\annotation_raw.txt

# Bước 3 — chia train/val
.\venv\Scripts\python.exe tools\export_ocr_lines.py --split D:\dataset_v1\annotation_raw.txt
```

Script dùng **đúng** `expand_box(pad_x=3, pad_y=10)` như pipeline inference,
nên ảnh dataset khớp với ảnh model sẽ gặp lúc chạy thật. Đây là chi tiết quan
trọng: nếu train trên ảnh crop kiểu khác, model học một phân bố khác với lúc dùng.

Mẹo hiệu suất: dùng `--max_score 0.9` để chỉ xuất những dòng model **chưa chắc
chắn**. Dòng score 0.99 thường model đã đọc đúng, sửa cũng không dạy được gì mới.

**Sửa tay nhanh hơn gõ từ đầu rất nhiều** — thường chỉ cần sửa 1–2 ký tự mỗi dòng.

### 5.4. Cần bao nhiêu dữ liệu

| Số dòng | Kỳ vọng |
|---|---|
| < 500 | Quá ít, overfit gần như chắc chắn |
| 1.000–3.000 | **Điểm ngọt cho fine-tune một domain** |
| 5.000–10.000 | Tốt, cải thiện rõ, vẫn sửa tay được trong vài ngày |
| > 50.000 | Cân nhắc train from scratch |

**1.000 dòng đúng domain > 50.000 dòng synthetic không đúng domain.**

Chia train/val: **90/10**. Val nên có ít nhất 200–300 dòng để accuracy có ý nghĩa
thống kê — với 50 dòng val thì mỗi dòng chiếm 2%, nhiễu quá lớn để đánh giá.

---

<a name="6"></a>
## 6. Tìm dataset trên HuggingFace

### 6.1. Bộ lọc

Vào `huggingface.co/datasets`:

- **Task:** `image-to-text` — đúng task. Đừng lọc `object-detection` (đó là data
  cho detection, không dùng trực tiếp cho recognition được).
- **Language:** `vi` — quan trọng nhất
- **Từ khoá:** `vietnamese ocr`, `vietnamese text recognition`,
  `vietnamese handwriting`, `vietnamese scene text`, `vietocr`

### 6.2. Tiêu chí đánh giá, theo thứ tự ưu tiên

**1. Ảnh là dòng đã crop hay cả trang?**
Mở tab *Dataset Viewer* nhìn trực tiếp. Ảnh dòng đơn (dài, dẹt, cao 32–64px)
→ dùng được gần như ngay. Ảnh cả trang → phải tự crop theo annotation,
công sức gấp nhiều lần.

**2. Có đủ dấu thanh không?**
Một số dataset gắn nhãn "Vietnamese" nhưng label bị mất dấu. Soi label trong
Dataset Viewer, tìm ký tự 2 tầng: `ắ ẳ ộ ữ ự`. Đây đúng là điểm yếu bạn cần chữa.

**3. Domain có khớp tài liệu của bạn không?** ← quyết định giá trị thực tế
Chữ viết tay không giúp gì cho việc đọc hợp đồng in. Scene text (biển hiệu,
ảnh đường phố) cũng không. Bạn cần **văn bản hành chính / tài liệu in / scan**.

**4. Real hay synthetic?**
Đọc dataset card tìm chữ "synthetic"/"generated". Model fine-tune trên synthetic
thuần thường **không cải thiện** trên scan thật — scan thật có nhiễu, mờ, nghiêng,
vệt mực mà render không tái tạo được.

**5. License** — kiểm tra nếu dùng thương mại.

### 6.3. Nói thẳng về thực tế

Hệ sinh thái dataset OCR tiếng Việt trên HF khá mỏng, và phần lớn là handwriting
hoặc scene text — cả hai đều **không khớp** với tài liệu hành chính scan.
Tôi không liệt kê tên dataset cụ thể vì không xác minh được chúng còn tồn tại
và đúng như mô tả; bạn tự search theo bộ lọc trên rồi đánh giá theo 5 tiêu chí.

**Khuyến nghị:** dùng dataset public để **bổ sung số lượng**, còn phần lõi
nên tự cắt từ tài liệu thật (mục 5.3). Trộn theo tỉ lệ khoảng 70% data thật
của bạn + 30% data public là hợp lý.

### 6.4. Convert dataset HF về format VietOCR

Đại ý (chạy trên Colab):

```python
from datasets import load_dataset
import os

ds = load_dataset("ten/dataset", split="train")
os.makedirs("hf_data/images", exist_ok=True)

with open("hf_data/annotation.txt", "w", encoding="utf-8", newline="\n") as f:
    for i, s in enumerate(ds):
        name = f"images/hf_{i:06d}.jpg"
        s["image"].convert("RGB").save(f"hf_data/{name}", quality=95)
        text = s["text"].replace("\t", " ").replace("\n", " ").strip()
        if text:
            f.write(f"{name}\t{text}\n")
```

Tên cột (`image`, `text`) khác nhau tuỳ dataset — in `ds.features` ra xem trước.

---

<a name="7"></a>
## 7. Toàn bộ tham số cấu hình

Đây là config `vgg_seq2seq` **thật**, đọc từ package đã cài. Cột cuối là giá trị
tôi khuyên cho fine-tune.

### 7.1. `dataset`

| Tham số | Mặc định | Ý nghĩa | Fine-tune |
|---|---|---|---|
| `data_root` | `./img/` | Thư mục gốc, đường dẫn trong annotation tính từ đây | trỏ tới dataset của bạn |
| `train_annotation` | `annotation_train.txt` | File annotation train | giữ |
| `valid_annotation` | `annotation_val_small.txt` | File annotation val | `annotation_val.txt` |
| `name` | `data` | Tên dataset — **quyết định tên thư mục LMDB** (`train_<name>`) | đổi mỗi lần đổi data |
| `image_height` | `32` | Chiều cao ép cố định | **giữ nguyên** |
| `image_min_width` | `32` | Chiều rộng tối thiểu | giữ |
| `image_max_width` | `512` | Chiều rộng tối đa, dài hơn bị co lại | tăng 768 nếu dòng rất dài |

> `name` quan trọng hơn vẻ ngoài: LMDB được cache theo tên này. Đổi dataset mà
> giữ nguyên `name` và không xoá thư mục cũ → **train lại dữ liệu cũ**.

### 7.2. `trainer`

| Tham số | Mặc định | Ý nghĩa | Fine-tune |
|---|---|---|---|
| `batch_size` | `32` | Số ảnh mỗi bước | 32 (T4 16GB thoải mái) |
| `iters` | `100000` | Tổng số bước train | **10000–20000** |
| `print_every` | `200` | In train loss mỗi N bước | 100 |
| `valid_every` | `4000` | Chạy validation mỗi N bước | **500–1000** |
| `checkpoint` | `./checkpoint/...pth` | Đường dẫn checkpoint | (xem cảnh báo dưới) |
| `export` | `./weights/transformerocr.pth` | **File weight cuối cùng bạn dùng** | trỏ vào Google Drive |
| `log` | `./train.log` | File log | trỏ vào Drive |
| `metrics` | `null` | Số mẫu val dùng để tính accuracy. `null` = toàn bộ | 1000 (hoặc null nếu val nhỏ) |

**`valid_every` quan trọng hơn bạn nghĩ:** `train()` **chỉ ghi file weight khi
`acc_full_seq` vượt kỷ lục cũ**, và chỉ kiểm tra tại các mốc validation. Đặt
`valid_every=4000` với `iters=10000` nghĩa là chỉ có **2 cơ hội** lưu file trong
cả phiên train. Colab ngắt giữa chừng là mất sạch. Đặt 500–1000.

### 7.3. `optimizer`

| Tham số | Mặc định | Ý nghĩa | Fine-tune |
|---|---|---|---|
| `max_lr` | `0.001` | Learning rate đỉnh của OneCycleLR | **`0.0001`** (thấp hơn 10 lần) |
| `pct_start` | `0.1` | Tỉ lệ số bước dùng để warm-up | 0.1 |

Scheduler là `OneCycleLR(total_steps=iters, max_lr, pct_start)`:

```
LR
 │      ╭──╮
 │    ╭─╯  ╰──╮
 │  ╭─╯       ╰────╮
 │╭─╯               ╰─────╮
 └──────────────────────────► iterations
  ↑ warmup 10%    ↑ anneal 90%
```

Warm-up tránh phá trọng số pretrained ngay bước đầu; anneal cho model hội tụ mượt.

> **Hệ quả cần biết:** `total_steps = iters`. Nếu Colab ngắt ở bước 6000/20000
> rồi bạn chạy lại, lịch LR **bắt đầu lại từ đầu** — model nhận cú sốc LR cao
> lần nữa. Xem mục 8.7 về resume.

Optimizer là `AdamW(betas=(0.9, 0.98), eps=1e-9)`, loss là
`LabelSmoothingLoss(smoothing=0.1)` — không cấu hình được qua config, không cần đụng.

### 7.4. `aug` — augmentation

| Tham số | Mặc định | Ý nghĩa | Fine-tune |
|---|---|---|---|
| `image_aug` | `true` | Bật biến đổi ảnh ngẫu nhiên (nghiêng, mờ, nhiễu, co giãn) | **`true`** |
| `masked_language_model` | `true` | Che ngẫu nhiên ký tự đầu vào decoder | `true` |

`image_aug` đặc biệt quan trọng khi dataset nhỏ: nó nhân số biến thể lên,
chống overfit. Chỉ tắt khi dataset đã rất lớn và đa dạng.

### 7.5. `dataloader`

| Tham số | Mặc định | Fine-tune |
|---|---|---|
| `num_workers` | `3` | 2 trên Colab; **0 nếu chạy Windows** (Windows hay treo với multiprocessing) |
| `pin_memory` | `true` | true trên GPU, false trên CPU |

### 7.6. `pretrain` và `weights` — điểm dễ nhầm nhất

```python
config['pretrain']   # <- Trainer NẠP TỪ ĐÂY khi khởi tạo
config['weights']    # <- chỉ Predictor (inference) dùng
```

`Trainer.__init__(config, pretrained=True)` gọi
`download_weights(config['pretrain'])`. Hàm đó:

```python
def download_weights(uri, ...):
    if uri.startswith('http'):
        return download(url=uri, ...)
    return uri                      # <- đường dẫn local trả về nguyên vẹn
```

Nghĩa là **đặt `config['pretrain']` thành đường dẫn file local là fine-tune từ
weight local**. Đây chính là cơ chế dùng để:
- fine-tune từ `vgg_seq2seq.pth` có sẵn trong project
- **resume** từ weight của phiên Colab trước

Nếu bạn chỉ đổi `config['weights']` mà quên `config['pretrain']`, model sẽ tải
weight gốc từ internet và **mọi kết quả train trước đó bị bỏ qua** — im lặng,
không báo lỗi.

---

<a name="8"></a>
## 8. Google Colab — step by step

### 8.1. Chuẩn bị trước khi mở Colab

1. Nén dataset thành **1 file zip**: `dataset_v1.zip` chứa `images/`,
   `annotation_train.txt`, `annotation_val.txt`
2. Upload lên Google Drive, ví dụ `MyDrive/ocr/dataset_v1.zip`
3. Upload weight khởi điểm `vgg_seq2seq.pth` lên `MyDrive/ocr/` (89 MB)
4. Tạo sẵn thư mục `MyDrive/ocr/output/` để nhận checkpoint

> **Zip, không phải hàng nghìn file rời.** Đọc trực tiếp nhiều file nhỏ từ Drive
> cực chậm, sẽ thành nút cổ chai nặng hơn cả GPU. Giải nén ra ổ **local** của
> Colab rồi mới train.

### 8.2. Bật GPU

`Runtime → Change runtime type → T4 GPU → Save`

**Cell 1 — xác nhận:**
```python
!nvidia-smi
```
Phải thấy `Tesla T4` và `15360MiB`. Không thấy thì bạn đang chạy CPU, train sẽ
chậm gấp ~50 lần.

### 8.3. Cài đặt

**Cell 2:**
```python
!pip install -q vietocr==0.3.13
!pip install -q "numpy<2"
```

> **`numpy<2` là bắt buộc, không phải tuỳ chọn.** Tôi đã kiểm chứng: với numpy 2.x
> pipeline train **không chạy được**, hai chỗ vỡ:
> - `vietocr/loader/dataloader.py` và `tool/create_dataset.py` dùng `np.fromstring`
>   → `ValueError: The binary mode of fromstring is removed`
> - `imgaug/imgaug.py:45` dùng `np.sctypes`
>   → `AttributeError: np.sctypes was removed in the NumPy 2.0 release`
>
> Sau khi cài `numpy<2`, **phải Restart runtime** (`Runtime → Restart session`)
> rồi chạy lại từ cell sau, nếu không Python vẫn giữ numpy cũ trong bộ nhớ.

**Cell 3 — kiểm tra dependency train:**
```python
import imgaug, albumentations, prefetch_generator, lmdb, numpy as np
print("numpy", np.__version__)   # phải là 1.x
```

`vietocr` khai báo `imgaug`, `albumentations==1.4.2`, `prefetch_generator==1.0.1`
là dependency, và `model/trainer.py` **import chúng ở top-level** — thiếu bất kỳ
cái nào là không import được `Trainer`, kể cả khi bạn tắt augmentation.
Nếu cell trên báo thiếu:
```python
!pip install -q imgaug albumentations prefetch_generator
```

### 8.4. Mount Drive và giải nén

**Cell 4:**
```python
from google.colab import drive
drive.mount('/content/drive')

DRIVE = '/content/drive/MyDrive/ocr'
!mkdir -p {DRIVE}/output
!cp {DRIVE}/dataset_v1.zip /content/
!unzip -q -o /content/dataset_v1.zip -d /content/data
!ls /content/data | head
!wc -l /content/data/annotation_train.txt /content/data/annotation_val.txt
```

Dataset nằm ở `/content/data` (**ổ local, nhanh**), checkpoint ghi sang
`{DRIVE}/output` (**Drive, sống sót khi session chết**).

### 8.5. Kiểm tra dataset trước khi train

**Cell 5** — bước này tiết kiệm hàng giờ:
```python
import os
from vietocr.tool.config import Cfg

ROOT = '/content/data'
vocab = Cfg.load_config_from_name('vgg_seq2seq')['vocab']

for ann in ['annotation_train.txt', 'annotation_val.txt']:
    path = os.path.join(ROOT, ann)
    bad_fmt, missing_img, out_vocab, n = 0, 0, set(), 0
    for line in open(path, encoding='utf-8'):
        line = line.rstrip('\n')
        if not line.strip():
            continue
        n += 1
        parts = line.split('\t')
        if len(parts) != 2:
            bad_fmt += 1
            continue
        img, label = parts
        if not os.path.exists(os.path.join(ROOT, img)):
            missing_img += 1
        out_vocab |= {c for c in label if c not in vocab}
    print(f"{ann}: {n} dòng | sai format {bad_fmt} | thiếu ảnh {missing_img}")
    print(f"   ký tự ngoài vocab: {sorted(out_vocab) if out_vocab else 'không có'}")
```

Cả 4 con số lỗi phải bằng 0 trước khi train.

**Xem thử ảnh** — đảm bảo đúng là ảnh dòng, không phải cả trang:
```python
from PIL import Image
import matplotlib.pyplot as plt

lines = open(f'{ROOT}/annotation_train.txt', encoding='utf-8').read().splitlines()
fig, axes = plt.subplots(5, 1, figsize=(14, 6))
for ax, line in zip(axes, lines[:5]):
    img_path, label = line.split('\t')
    img = Image.open(os.path.join(ROOT, img_path))
    ax.imshow(img); ax.axis('off'); ax.set_title(f"{label}   [{img.size}]", loc='left')
plt.tight_layout(); plt.show()
```

### 8.6. Cấu hình và train

**Cell 6:**
```python
import shutil, os
from vietocr.tool.config import Cfg
from vietocr.model.trainer import Trainer

DRIVE = '/content/drive/MyDrive/ocr'
ROOT  = '/content/data'
NAME  = 'v1'          # đổi tên này mỗi lần đổi dataset

# Xoá LMDB cũ — nếu không sẽ train lại dữ liệu cũ mà không biết
for d in (f'train_{NAME}', f'valid_{NAME}'):
    shutil.rmtree(d, ignore_errors=True)

config = Cfg.load_config_from_name('vgg_seq2seq')

# --- nguồn weight khởi điểm ---
config['pretrain'] = f'{DRIVE}/vgg_seq2seq.pth'   # local -> KHÔNG tải mạng
config['device']   = 'cuda:0'

# --- dataset ---
config['dataset']['data_root']        = ROOT
config['dataset']['train_annotation'] = 'annotation_train.txt'
config['dataset']['valid_annotation'] = 'annotation_val.txt'
config['dataset']['name']             = NAME

# --- siêu tham số fine-tune ---
config['optimizer']['max_lr']   = 0.0001     # THẤP: giữ kiến thức cũ
config['optimizer']['pct_start'] = 0.1
config['trainer']['batch_size'] = 32
config['trainer']['iters']      = 15000
config['trainer']['print_every'] = 100
config['trainer']['valid_every'] = 500       # lưu file thường xuyên
config['trainer']['metrics']    = 1000

# --- đầu ra: GHI VÀO DRIVE ---
config['trainer']['export']     = f'{DRIVE}/output/vgg_seq2seq_ft.pth'
config['trainer']['checkpoint'] = f'{DRIVE}/output/ckpt.pth'
config['trainer']['log']        = f'{DRIVE}/output/train.log'

config['dataloader']['num_workers'] = 2
config['dataloader']['pin_memory']  = True

config.save(f'{DRIVE}/output/config_{NAME}.yml')   # lưu lại để tái lập
trainer = Trainer(config)
```

Khi khởi tạo, trainer sẽ build LMDB (in progress bar `Create train_v1`).
Kiểm tra dòng `Created dataset with N samples` — N phải khớp số dòng annotation
**trừ 1** (bug off-by-one, mục 11.3).

**Cell 7 — train:**
```python
trainer.train()
```

### 8.7. Sống sót qua việc Colab ngắt session

Colab free ngắt sau ~12 giờ, hoặc sớm hơn nếu hết GPU quota / bạn không tương
tác. **Khi ngắt, toàn bộ ổ đĩa máy ảo bị xoá sạch** — chỉ Drive còn.

Cấu hình ở Cell 6 đã xử lý phần lớn: `export`, `log` đều ghi thẳng vào Drive,
và `valid_every=500` nghĩa là cứ ~500 bước lại có cơ hội lưu.

**Cách resume ở phiên mới:** chạy lại Cell 2→6, chỉ đổi một dòng:

```python
config['pretrain'] = f'{DRIVE}/output/vgg_seq2seq_ft.pth'   # weight phiên trước
config['trainer']['iters'] = 8000                            # số bước CÒN LẠI
```

> **Không dùng `trainer.load_checkpoint()`.** Tôi đã kiểm chứng: hàm này **hỏng
> với `vgg_seq2seq`**. Nó gọi `self.config['transformer']['d_model']`, nhưng
> config seq2seq chỉ có `decoder_embedded / decoder_hidden / encoder_hidden /
> img_channel` — **không có `d_model`** → `KeyError`. Và ngay cả khi có, nó
> khởi tạo `ScheduledOptim(optimizer, d_model, init_lr, n_warmup_steps)` bằng
> `**config['optimizer']` = `{max_lr, pct_start}` — sai hoàn toàn tên tham số.
>
> Hàm này chỉ dùng được với `vgg_transformer`.

Hạn chế của cách resume qua `pretrain`: mất trạng thái optimizer và lịch LR
(OneCycle chạy lại từ warm-up). Chấp nhận được với fine-tune LR thấp. Cách tránh
tốt nhất là **đặt `iters` đủ nhỏ để một phiên chạy xong** — 15.000 iter với
batch 32 trên T4 thường mất 1–3 giờ, thừa sức trong một phiên.

### 8.8. Thời gian dự kiến

| Cấu hình | T4 (Colab free) |
|---|---|
| 15.000 iters, batch 32 | ~1,5–3 giờ |
| Build LMDB 3.000 ảnh | < 1 phút |
| 1 lần validation (1000 mẫu) | ~30–60 giây |

---

<a name="9"></a>
## 9. Đọc chỉ số: train thành công hay thất bại

### 9.1. Log train

Mỗi `print_every` bước:
```
iter: 000100 - train loss: 1.234 - lr: 2.31e-05 - load time: 1.20 - gpu time: 8.50
```

| Trường | Ý nghĩa | Cần thấy gì |
|---|---|---|
| `train loss` | Loss trung bình trên train | Giảm dần, không nhảy loạn |
| `lr` | Learning rate hiện tại | Tăng trong 10% đầu, rồi giảm dần |
| `load time` | Thời gian nạp dữ liệu | **Phải nhỏ hơn nhiều so với gpu time** |
| `gpu time` | Thời gian tính toán | — |

> `load time` > `gpu time` nghĩa là GPU đang **chờ dữ liệu** — thường do đọc
> ảnh trực tiếp từ Drive thay vì ổ local. Sửa bằng cách copy data về `/content`.

### 9.2. Log validation

Mỗi `valid_every` bước:
```
iter: 000500 - valid loss: 0.987 - acc full seq: 0.7412 - acc per char: 0.9563
```

| Chỉ số | Định nghĩa | Diễn giải |
|---|---|---|
| `valid loss` | Loss trên tập val | **Chỉ số quan trọng nhất để phát hiện overfit** |
| `acc full seq` | Tỉ lệ dòng đúng **hoàn toàn** — sai 1 ký tự = sai cả dòng | Khắt khe. Đây là chỉ số quyết định việc lưu file |
| `acc per char` | Tỉ lệ ký tự đúng | Sát với "chất lượng cảm nhận" hơn |

Hai chỉ số này tính bởi `compute_accuracy()` với `mode='full_sequence'` và
`mode='per_char'`.

### 9.3. Bốn kịch bản

**✅ Thành công:**
```
iter 000500 - valid loss: 1.102 - acc full seq: 0.6120 - acc per char: 0.9210
iter 001000 - valid loss: 0.943 - acc full seq: 0.6890 - acc per char: 0.9445
iter 002000 - valid loss: 0.812 - acc full seq: 0.7530 - acc per char: 0.9620
```
Train loss ↓, valid loss ↓, accuracy ↑. Cứ để chạy tiếp.

**⚠️ Overfitting** — vấn đề hay gặp nhất khi dataset nhỏ:
```
iter 002000 - valid loss: 0.812 - acc full seq: 0.7530
iter 003000 - valid loss: 0.856 - acc full seq: 0.7490    <- valid loss TĂNG
iter 004000 - valid loss: 0.921 - acc full seq: 0.7410    <- accuracy GIẢM
```
Train loss vẫn giảm nhưng valid loss tăng = model học thuộc lòng train set.
**Dừng ngay.** File `export` đang giữ weight tại đỉnh (iter 2000) — vẫn dùng được.
Lần sau: giảm `iters`, tăng data, hoặc giữ `image_aug: true`.

**❌ Không học được gì:**
```
iter 000500 - valid loss: 2.341 - acc full seq: 0.0120
iter 002000 - valid loss: 2.339 - acc full seq: 0.0140
```
Loss đứng im ở mức cao. Thường do: annotation sai format (label lệch với ảnh),
`pretrain` trỏ sai file, hoặc LR quá thấp. Kiểm tra lại Cell 5.

**❌ Catastrophic forgetting** — nguy hiểm vì **nhìn log thì vẫn đẹp**:
```
acc full seq: 0.95    trên tập val của bạn
```
nhưng model đọc tài liệu khác tệ hơn hẳn so với trước. Log không phát hiện được.
Chỉ phát hiện bằng **benchmark A/B ở mục 10**.

### 9.4. Con số bao nhiêu là tốt

Không có ngưỡng tuyệt đối — phải **so với chính weight gốc trên cùng tập val**.

Cách làm: trước khi train, chạy validation với weight gốc để lấy mốc baseline.

```python
base_full, base_char = trainer.precision(1000)   # chạy NGAY sau khi tạo Trainer
print(f"BASELINE (weight gốc): full_seq={base_full:.4f} per_char={base_char:.4f}")
```

Sau khi train xong, so với con số này. Không cải thiện thì fine-tune vô ích —
và điều đó xảy ra thường xuyên hơn người ta tưởng.

### 9.5. Xem model sai ở đâu

```python
trainer.visualize_prediction(sample=20, errorcase=True)
```
Hiện ảnh kèm `pred` vs `actual`, **chỉ những case sai**. Cực kỳ hữu ích:
- Sai toàn ở dấu thanh → tăng `pad_y` trong detection, không phải lỗi recognition
- Sai ở ảnh mờ/nghiêng → cần thêm data loại đó
- `actual` nhìn đã sai so với ảnh → **label của bạn sai**, phải sửa dataset

---

<a name="10"></a>
## 10. Đưa weight về project

### 10.1. Tải về

```python
from google.colab import files
files.download('/content/drive/MyDrive/ocr/output/vgg_seq2seq_ft.pth')
```
Hoặc tải thẳng từ Drive.

### 10.2. Benchmark A/B — LÀM TRƯỚC KHI THAY

**Đừng thay weight rồi mới kiểm tra.** Fine-tune có thể làm tệ đi.

1. Giữ nguyên `vgg_seq2seq.pth` cũ
2. Copy weight mới vào `vietocr/weight/vgg_seq2seq_ft.pth`
3. Chuẩn bị một bộ tài liệu thật **không nằm trong dataset train**
4. Chạy `t_ocr.py` với weight cũ → lưu kết quả
5. Đổi `config['weights']` trong [module/ocr.py](module/ocr.py) sang file mới → chạy lại
6. So sánh output text của hai lần

Quan trọng: bộ tài liệu benchmark nên gồm **cả loại đã fine-tune lẫn loại khác**,
để phát hiện catastrophic forgetting (mục 9.3).

### 10.3. Tích hợp

Sửa đúng một dòng trong [module/ocr.py](module/ocr.py), class `TextRecognizer`:

```python
config['weights'] = r"vietocr\weight\vgg_seq2seq_ft.pth"
```

Không cần đụng gì khác — `api.py`, `t_ocr.py`, `full_pipeline.py` đều dùng chung
`TextRecognizer`.

### 10.4. Lưu ý về bản ONNX

Fine-tune xong bạn **mất đường dùng `module/ocr_onnx.py`** cho đến khi export
lại model mới sang ONNX (bước riêng, thêm việc). Nếu tốc độ chưa phải vấn đề
cấp bách thì cứ dùng bản PyTorch.

---

<a name="11"></a>
## 11. Các lỗi đã biết của vietocr

Toàn bộ mục này **đã kiểm chứng thực tế** trên `vietocr==0.3.13` trong `venv/`.

### 11.1. Không tương thích numpy 2.x — chặn hoàn toàn

| File | Dòng lỗi | Thông báo |
|---|---|---|
| `loader/dataloader.py` (`get_bucket`) | `np.fromstring(dim_img, dtype=np.int32)` | `ValueError: The binary mode of fromstring is removed` |
| `tool/create_dataset.py` (`checkImageIsValid`) | `np.fromstring(imageBin, dtype=np.uint8)` | như trên |
| `imgaug/imgaug.py:45` | `set(np.sctypes["float"])` | `AttributeError: np.sctypes was removed in the NumPy 2.0 release` |

**Cách xử lý:** `pip install "numpy<2"` rồi **Restart runtime**. Đây là cách
sạch nhất — patch từng hàm cũng được nhưng `imgaug` còn nhiều chỗ dùng API cũ,
sửa không xuể.

### 11.2. `load_checkpoint()` hỏng với `vgg_seq2seq`

```python
# model/trainer.py
def load_checkpoint(self, filename):
    optim = ScheduledOptim(
        Adam(...),
        self.config['transformer']['d_model'],   # <- KHÔNG TỒN TẠI ở seq2seq
        **self.config['optimizer'])              # <- {max_lr, pct_start}
```

Config `vgg_seq2seq` có `transformer: {decoder_embedded, decoder_hidden, dropout,
encoder_hidden, img_channel}` — không có `d_model` → `KeyError`.
Và `ScheduledOptim.__init__(self, optimizer, d_model, init_lr, n_warmup_steps)`
không nhận `max_lr`/`pct_start`.

**Cách xử lý:** resume bằng `config['pretrain']` (mục 8.7).

### 11.3. `createDataset` bỏ mất dòng cuối

```python
# tool/create_dataset.py:85
nSamples = cnt-1        # cnt đếm từ 0 -> 12 ảnh hợp lệ ghi thành 11
```

Kiểm chứng: dataset mẫu 12 dòng → `Created dataset with 11 samples`,
`error = 0` (không ảnh nào bị loại).

**Cách xử lý:** thêm 1 dòng thừa vào cuối annotation.
`tools/export_ocr_lines.py --split` đã tự làm việc này.

### 11.4. LMDB bị cache im lặng

```
train_data exists. Remove folder if you want to create new dataset
```

In ra như thông báo bình thường, rất dễ lướt qua. Đổi annotation mà không xoá
thư mục LMDB → **train lại dữ liệu cũ**.

**Cách xử lý:** luôn `shutil.rmtree(f'train_{NAME}')` trước khi tạo Trainer
(đã có trong Cell 6).

### 11.5. `train()` không bao giờ gọi `save_checkpoint()`

Đọc `Trainer.train()`: chỉ có `self.save_weights(self.export_weights)`, và chỉ
khi `acc_full_seq > best_acc`. `save_checkpoint()` tồn tại nhưng **không được
gọi ở đâu trong vòng train**.

Hai hệ quả:
- File `export` luôn là **weight tốt nhất**, không phải weight cuối cùng — đây
  là điều tốt, bạn không cần tự chọn checkpoint.
- Nhưng nếu accuracy **không bao giờ cải thiện** so với lần validation đầu tiên,
  **không file nào được ghi ra cả**. Train 3 tiếng rồi không có gì.
  → Lý do nữa để đặt `valid_every` nhỏ.

### 11.6. Dependency train không nằm trong `requirements.txt` của project

`module/trainer.py` import ở top-level: `imgaug`, `albumentations`,
`prefetch_generator`. Project này chỉ cài dependency cho **inference** nên
không có chúng — đúng, vì không cần khi chỉ chạy OCR.

Chỉ cài chúng **trên Colab**, đừng cài vào venv local:
`albumentations` kéo theo `opencv-python-headless`, sẽ **đè lên `opencv-python`**
mà pipeline OCR đang dùng.

---

<a name="12"></a>
## 12. Xử lý sự cố

| Triệu chứng | Nguyên nhân | Cách sửa |
|---|---|---|
| `ValueError: The binary mode of fromstring is removed` | numpy 2.x | `pip install "numpy<2"` + Restart runtime |
| `AttributeError: np.sctypes was removed` | numpy 2.x + imgaug | như trên |
| `ModuleNotFoundError: imgaug / albumentations / prefetch_generator` | thiếu dep train | `pip install imgaug albumentations prefetch_generator` |
| `KeyError: 'd_model'` | gọi `load_checkpoint()` | Dùng `config['pretrain']` để resume (8.7) |
| `Created dataset with N-1 samples` | bug off-by-one | Bình thường, thêm 1 dòng thừa |
| `train_data exists. Remove folder...` | LMDB cache | Xoá `train_<name>`, `valid_<name>` |
| `CUDA out of memory` | batch quá lớn | `batch_size` 32 → 16 → 8 |
| Không có file weight nào được ghi | accuracy chưa từng cải thiện | Giảm `valid_every`; kiểm tra `pretrain` đúng chưa |
| Loss đứng im ~2.3 | annotation lệch ảnh, hoặc `pretrain` sai | Chạy lại Cell 5; xem `visualize_prediction()` |
| `load time` >> `gpu time` | đọc data từ Drive | Copy dataset về `/content` |
| Accuracy cao nhưng dùng thật tệ hơn | catastrophic forgetting | Benchmark A/B (10.2); giảm `iters`, giảm `max_lr` |
| Label có `?` hoặc mất ký tự | ký tự ngoài vocab | Chuẩn hoá `—`→`-`, `"`→`"` (mục 5.2) |
| Windows treo khi train local | multiprocessing dataloader | `num_workers = 0` |

---

## Tóm tắt: thứ tự hành động

1. **Chẩn đoán** — chạy `t_ocr.py`, xem ảnh có box, xác định lỗi det hay rec (mục 1)
2. **Thử cách rẻ** — tăng DPI, nới `unclip_ratio`, tăng `pad_y` (mục 1.2)
3. **Tạo dataset** — `tools/export_ocr_lines.py` trên tài liệu thật, sửa tay, ~1.000–3.000 dòng (mục 5.3)
4. **Kiểm tra dataset** — Cell 5 trên Colab, 4 con số lỗi phải bằng 0 (mục 8.5)
5. **Lấy baseline** — `trainer.precision(1000)` trước khi train (mục 9.4)
6. **Train** — `max_lr=1e-4`, `iters=15000`, `valid_every=500` (mục 8.6)
7. **Đọc chỉ số** — valid loss tăng là dừng (mục 9.3)
8. **Benchmark A/B** — trên tài liệu ngoài tập train, cả trong lẫn ngoài domain (mục 10.2)
9. **Chỉ khi tốt hơn thật** mới thay weight trong `module/ocr.py`
