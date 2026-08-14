# Dataset mẫu cho VietOCR

Thư mục này minh hoạ **đúng format** mà VietOCR yêu cầu. Xem [../training_guide.md](../training_guide.md) để biết quy trình đầy đủ.

```
dataset_sample/
├── images/                    15 ảnh DÒNG chữ (không phải cả trang)
│   ├── line_0000.jpg
│   └── ...
├── annotation_train.txt       12 dòng
├── annotation_val.txt         3 dòng
├── make_sample_dataset.py     script sinh ra thư mục này
└── README.md
```

## Format file annotation

Mỗi dòng: **đường dẫn ảnh** + **một ký tự TAB** + **label**.

```
images/line_0000.jpg	Hợp đồng bảo hiểm nhân thọ số 0123456789
images/line_0001.jpg	Điều 1: Phạm vi bảo hiểm và quyền lợi được hưởng
```

Quy tắc bắt buộc:

| Quy tắc | Lý do |
|---|---|
| Ngăn cách bằng **TAB** (`\t`), không phải space | `create_dataset.py` dùng `split('\t')` |
| Encoding **UTF-8** | Tiếng Việt có dấu |
| Xuống dòng **LF** (`\n`), không phải CRLF | CRLF làm label dính ký tự `\r` |
| Đường dẫn **tương đối** so với `data_root` | `os.path.join(root_dir, imageFile)` |
| **Không** có TAB trong label | Sẽ tách nhầm thành 3 field |

Kiểm tra nhanh trên Windows PowerShell (phải thấy ký tự tab):
```powershell
Get-Content dataset_sample\annotation_train.txt -Encoding UTF8 -TotalCount 2
```

## Ảnh phải là DÒNG chữ, không phải cả trang

Đây là điểm hay nhầm nhất. VietOCR chỉ nhận ảnh **một dòng chữ đã crop**:

```
✅ ĐÚNG:  [ Hợp đồng bảo hiểm nhân thọ số 0123456789 ]     521 × 44 px
❌ SAI:   ảnh A4 cả trang 2480 × 3508 px
```

Ảnh được resize về **chiều cao 32px** cố định, chiều rộng co theo tỉ lệ
(giới hạn `image_min_width=32`, `image_max_width=512`).

Đây đúng bằng thứ mà `get_rotate_crop_image()` trong [../module/ocr.py](../module/ocr.py)
tạo ra ở mỗi lần OCR — nên bạn có thể tái sử dụng chính pipeline detection
của project để cắt dataset. Xem phần "Tạo dataset từ tài liệu thật" trong training guide.

## Đã kiểm chứng

Nạp thư mục này qua `OCRDataset` của vietocr:

```
nSamples          = 11          ← 12 dòng nhưng chỉ 11 (xem cảnh báo dưới)
img tensor shape  = (3, 32, 380)   ← đúng: 3 kênh, cao 32px
label decode      = 'Hợp đồng bảo hiểm nhân thọ số 0123456789'
```

Label đi qua vocab rồi decode ngược lại **khớp 100%** — mọi ký tự tiếng Việt
trong dataset mẫu đều nằm trong bảng vocab mặc định.

## ⚠️ Hai cái bẫy của vietocr

### 1. Dòng cuối annotation LUÔN bị bỏ

`vietocr/tool/create_dataset.py:85` có lỗi off-by-one:

```python
cnt = 0
for ...:
    cnt += 1          # 12 ảnh hợp lệ -> cnt = 12
nSamples = cnt-1      # nhưng ghi 11
```

Không có cảnh báo nào (`error = 0`, tất cả ảnh đều hợp lệ). Đó là lý do
dataset mẫu 12 dòng chỉ nạp được 11.

**Cách xử lý:** thêm một dòng thừa vào cuối mỗi file annotation (lặp lại dòng
cuối cũng được). Với dataset vài nghìn dòng thì mất 1 mẫu không đáng kể,
nhưng nên biết để không hoang mang khi số liệu lệch.

### 2. LMDB bị cache — sửa annotation mà không xoá thì train dữ liệu cũ

`OCRDataset.__init__` chỉ tạo LMDB khi thư mục chưa tồn tại:

```
train_data exists. Remove folder if you want to create new dataset
```

Dòng này in ra như thông báo bình thường, rất dễ lướt qua. **Mỗi lần đổi
dataset phải xoá thư mục LMDB** (`train_<name>` và `valid_<name>`),
nếu không bạn train lại đúng dữ liệu cũ mà không hề biết.

## Sinh lại thư mục này

```powershell
.\venv\Scripts\python.exe dataset_sample\make_sample_dataset.py
```

## Lưu ý về giá trị thực tế

Dataset này là **ảnh sinh tổng hợp** (render font lên nền + nhiễu nhẹ), chỉ dùng để:
- hiểu format
- test pipeline train chạy được đầu-cuối

**Không dùng nó để fine-tune thật.** Model fine-tune trên synthetic thuần
thường không cải thiện trên ảnh scan thật, vì scan thật có nhiễu, mờ, nghiêng,
vệt mực, bóng giấy mà render không tái tạo được. Dataset thật phải cắt từ
chính tài liệu bạn xử lý.
