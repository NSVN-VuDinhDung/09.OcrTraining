# Hướng dẫn cài đặt và chạy DeepDoc + VietOCR

## 1. Cài đặt môi trường (chỉ cần làm 1 lần)

### Tạo virtual environment
```powershell
cd d:\02.Source\02.Sample\19.DeepdocOcr\deepdoc_vietocr
python -m venv venv
```

### Upgrade pip
```powershell
.\venv\Scripts\python.exe -m pip install --upgrade pip
```

### Cài PyTorch CPU
```powershell
.\venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### Cài các dependencies còn lại
```powershell
.\venv\Scripts\python.exe -m pip install -r requirements.txt --no-deps
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

> **Lưu ý:** `vietocr` yêu cầu phiên bản cụ thể của một số package (pillow==10.2.0, einops==0.2.0, gdown==4.4.0) nhưng các phiên bản mới hơn vẫn hoạt động bình thường với Python 3.14+.

---

## 2. Chạy OCR

### Cú pháp
```powershell
.\venv\Scripts\python.exe t_ocr.py --inputs=<đường_dẫn_ảnh_hoặc_thư_mục> --output_dir=<thư_mục_output>
```

### Ví dụ — OCR một ảnh
```powershell
.\venv\Scripts\python.exe t_ocr.py --inputs=D:\testocr\image_10.png --output_dir=D:\testocr
```

### Ví dụ — OCR cả thư mục ảnh
```powershell
.\venv\Scripts\python.exe t_ocr.py --inputs=D:\testocr --output_dir=D:\testocr\output
```

### Xem log output
```powershell
Get-Content log\t_ocr.log -Encoding UTF8 | Select-Object -Last 30
```

> **Lưu ý:** Output không hiện trên terminal — tất cả được ghi vào `log/t_ocr.log`.  
> Lần đầu chạy sẽ tự động download model VietOCR (~548MB). Từ lần sau sẽ dùng cache.

---

## 3. Chạy API Server (FastAPI)

```powershell
cd d:\02.Source\02.Sample\19.DeepdocOcr\deepdoc_vietocr
.\venv\Scripts\python.exe api.py
```

> API khởi động tại `http://localhost:8000`  
> Swagger UI: `http://localhost:8000/docs`  
> Health check: `http://localhost:8000/health`

### Hoặc chạy với uvicorn trực tiếp (hỗ trợ reload khi phát triển)
```powershell
.\venv\Scripts\uvicorn.exe api:app --host 0.0.0.0 --port 8000 --reload
```

### Lỗi "port 8000 already in use" — Kill server cũ trước khi restart
```powershell
Get-NetTCPConnection -LocalPort 8000 -State Listen | Select-Object -ExpandProperty OwningProcess | ForEach-Object { Stop-Process -Id $_ -Force ; Write-Host "Killed PID $_" }
```

---

## 4. Chạy Layout Recognizer

```powershell
.\venv\Scripts\python.exe t_recognizer.py --inputs=D:\testocr\image_10.png --threshold=0.2 --mode=layout --output_dir=D:\testocr
```

## 4. Chạy Table Structure Recognizer

```powershell
.\venv\Scripts\python.exe t_recognizer.py --inputs=D:\testocr\image_10.png --threshold=0.2 --mode=tsr --output_dir=D:\testocr
```
