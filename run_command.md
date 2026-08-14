# Lệnh chạy — project OcrTraining

> Project này là môi trường **training**, venv là `venv_train`.
> Xem [training_guide.md](training_guide.md) để hiểu ý nghĩa từng tham số,
> [train_log.md](train_log.md) để xem nhật ký các lần train.

Đường dẫn local: `D:\02.Source\02.Sample\20.TrainingOcr`
Repo: https://github.com/NSVN-VuDinhDung/09.OcrTraining

## 1. Cài đặt môi trường (chỉ cần làm 1 lần)

### Tạo virtual environment
```powershell
cd D:\02.Source\02.Sample\20.TrainingOcr
python -m venv venv_train
.\venv_train\Scripts\python.exe -m pip install --upgrade pip
```

### Cài PyTorch CPU
```powershell
.\venv_train\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### Cài vietocr + dependencies training
```powershell
.\venv_train\Scripts\python.exe -m pip install --no-deps vietocr
.\venv_train\Scripts\python.exe -m pip install pyyaml gdown lmdb einops tqdm requests matplotlib opencv-python six scipy scikit-image imageio prefetch_generator huggingface_hub
.\venv_train\Scripts\python.exe -m pip install --no-deps imgaug albumentations
.\venv_train\Scripts\python.exe -m pip install "albucore==0.0.24" pydantic simsimd stringzilla
```

> **Vì sao `--no-deps`:** `vietocr` ghim `pillow==10.2.0`, `einops==0.2.0`,
> `gdown==4.4.0` — không có wheel cho Python 3.13+, pip sẽ cố build từ source
> rồi fail. Bản mới hoạt động bình thường.
> `albucore` phải ghim `0.0.24` để khớp `albumentations` 2.0.8 (bản mới cần `numkong`).

### Tải model (~810MB, repo không chứa)
```powershell
.\venv_train\Scripts\python.exe download_models.py           # từ nguồn gốc
.\venv_train\Scripts\python.exe download_models.py --release # từ GitHub Release
.\venv_train\Scripts\python.exe download_models.py --check   # chỉ kiểm tra
```

---

## 2. Training

### Sinh dataset mẫu (synthetic — chỉ để test pipeline)
```powershell
.\venv_train\Scripts\python.exe dataset_sample\make_sample_dataset.py --n 400
```

### Tạo dataset từ tài liệu THẬT
```powershell
.\venv_train\Scripts\python.exe tools\export_ocr_lines.py --inputs D:\tailieu --output_dir D:\dataset_v1
# sửa các dòng sai trong D:\dataset_v1\annotation_raw.txt (xem review.txt), rồi:
.\venv_train\Scripts\python.exe tools\export_ocr_lines.py --split D:\dataset_v1\annotation_raw.txt
```

### Train
```powershell
# CPU — chỉ để kiểm tra pipeline (~2.66 s/iter trên i5-12500)
.\venv_train\Scripts\python.exe train.py --name v1 --iters 500

# GPU / Colab
python train.py --name v1 --iters 15000 --device cuda:0 --num_workers 2

# Dataset của bạn
python train.py --name v2 --data_root D:\dataset_v1 --iters 15000 --device cuda:0

# Train tiếp từ kết quả lần trước
python train.py --name v3 --pretrain train_output\vgg_seq2seq_v2.pth --iters 8000
```

### Xem kết quả
```powershell
Get-Content train_output\train_v1.log -Encoding UTF8 | Select-String "valid loss"
Get-Content train_output\summary_v1.json -Encoding UTF8
```

---

## 3. Chạy OCR

### Cú pháp
```powershell
.\venv_train\Scripts\python.exe t_ocr.py --inputs=<đường_dẫn_ảnh_hoặc_thư_mục> --output_dir=<thư_mục_output>
```

### Ví dụ — OCR một ảnh
```powershell
.\venv_train\Scripts\python.exe t_ocr.py --inputs=D:\testocr\image_10.png --output_dir=D:\testocr
```

### Ví dụ — OCR cả thư mục ảnh
```powershell
.\venv_train\Scripts\python.exe t_ocr.py --inputs=D:\testocr --output_dir=D:\testocr\output
```

### Xem log output
```powershell
Get-Content log\t_ocr.log -Encoding UTF8 | Select-Object -Last 30
```

> **Lưu ý:** Output không hiện trên terminal — tất cả được ghi vào `log/t_ocr.log`.  
> Lần đầu chạy sẽ tự động download model VietOCR (~548MB). Từ lần sau sẽ dùng cache.

---

## 4. Chạy API Server (FastAPI)

```powershell
cd D:\02.Source\02.Sample\20.TrainingOcr
.\venv_train\Scripts\python.exe api.py
```

> API khởi động tại `http://localhost:8000`  
> Swagger UI: `http://localhost:8000/docs`  
> Health check: `http://localhost:8000/health`

### Hoặc chạy với uvicorn trực tiếp (hỗ trợ reload khi phát triển)
```powershell
.\venv_train\Scripts\uvicorn.exe api:app --host 0.0.0.0 --port 8000 --reload
```

### Lỗi "port 8000 already in use" — Kill server cũ trước khi restart
```powershell
Get-NetTCPConnection -LocalPort 8000 -State Listen | Select-Object -ExpandProperty OwningProcess | ForEach-Object { Stop-Process -Id $_ -Force ; Write-Host "Killed PID $_" }
```

---

## 5. Chạy Layout Recognizer

```powershell
.\venv_train\Scripts\python.exe t_recognizer.py --inputs=D:\testocr\image_10.png --threshold=0.2 --mode=layout --output_dir=D:\testocr
```

## 6. Chạy Table Structure Recognizer

```powershell
.\venv_train\Scripts\python.exe t_recognizer.py --inputs=D:\testocr\image_10.png --threshold=0.2 --mode=tsr --output_dir=D:\testocr
```

---

## 7. Đóng gói model thành zip

Model (~810MB) **không nằm trong git** — file 145MB vượt giới hạn cứng 100MB của
GitHub, và Git LFS thì tiêu gần hết 1GB quota free, hết quota là kẹt cả push lẫn
clone. Thay vào đó dùng **GitHub Release**: không tính vào LFS quota, giới hạn
2GB/file.

```powershell
cd D:\02.Source\02.Sample\20.TrainingOcr
.\venv_train\Scripts\python.exe make_release_zips.py
```

Sinh ra trong `release_assets/` (đã gitignore):

| Zip | Dung lượng | Nội dung |
|---|---|---|
| `models-core.zip` | 465.5 MB | `onnx/` đầy đủ + `vgg_seq2seq.pth` — đủ để pipeline chạy |
| `models-optional.zip` | 309.3 MB | `vgg_transformer.pth`, `transformerocr.pth`, `cnn/encoder/decoder.onnx` |

> **Không dùng `Compress-Archive` của PowerShell.** Nó ghi dấu `\` vào tên entry
> trong zip, khiến `unzip` trên Linux/Colab tạo ra file tên literal
> `onnx\det.onnx`. `make_release_zips.py` dùng `zipfile` của Python nên đường dẫn
> luôn là `/`, giải nén được trên cả Windows lẫn Linux.

---

## 8. Upload zip lên GitHub Release

Tag **phải là `models-v1`** — khớp với `RELEASE_TAG` trong `download_models.py`.
Đặt tag khác thì `download_models.py --release` sẽ tải 404.

```powershell
gh release create models-v1 `
    release_assets\models-core.zip `
    release_assets\models-optional.zip `
    --title "Model weights v1" `
    --notes "onnx/ (InfiniFlow/deepdoc) + vietocr/weight/. Giải nén tại gốc repo."
```

### Thêm / thay file vào release đã có
```powershell
gh release upload models-v1 release_assets\models-core.zip --clobber
```

### Kiểm tra sau khi upload
```powershell
gh release view models-v1
.\venv_train\Scripts\python.exe download_models.py --release --force
```

### Hoặc upload qua web
Repo → **Releases** → *Draft a new release* → tag `models-v1` → kéo hai file zip
vào ô *Attach binaries* → **Publish release**.

> Weight bạn tự fine-tune cũng nên để trên Release (tag riêng, ví dụ
> `weights-v1`) — đó là sản phẩm riêng, không tải lại từ HuggingFace hay
> vocr.vn được.

---

## 9. Git

### Lần đầu (đã làm xong)
```powershell
cd D:\02.Source\02.Sample\20.TrainingOcr
git init
git branch -M main
git remote add origin https://github.com/NSVN-VuDinhDung/09.OcrTraining.git
git add -A
git commit -m "Khởi tạo môi trường training fine-tune VietOCR"
git push -u origin main
```

### Commit và push thay đổi
```powershell
git add -A
git status --short          # xem lại trước khi commit
git commit -m "Nội dung thay đổi"
git push origin main
```

### Kiểm tra không có model lọt vào commit
```powershell
git diff --cached --name-only | Select-String -Pattern "\.pth$|\.onnx$|\.mdb$|venv"
```
Không ra dòng nào là đúng. Nếu có, kiểm tra lại `.gitignore`.

### Xem dung lượng sẽ push
```powershell
git count-objects -vH
```

### Đăng nhập lại khi token hết hạn
```powershell
gh auth login -h github.com
gh auth status
```

### Lỗi "Updates were rejected because the remote contains work that you do not have"
Xảy ra khi repo trên GitHub đã có commit (ví dụ README do GitHub tự tạo):
```powershell
git fetch origin
git log origin/main --oneline          # xem trên remote có gì
git ls-tree -r --name-only origin/main
git merge origin/main --allow-unrelated-histories -X ours -m "Merge initial commit"
git push -u origin main
```
`-X ours` giữ phiên bản local khi trùng file. **Xem nội dung remote trước** —
đừng merge mù, có thể ghi đè thứ cần giữ.
