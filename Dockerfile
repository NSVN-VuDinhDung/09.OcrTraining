# Build stage: chỉ cài dependencies
FROM python:3.14-slim AS builder

WORKDIR /app

# System deps cần cho opencv + pdfplumber
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# PyTorch CPU trước (lớn nhất, cache riêng layer)
RUN pip install --no-cache-dir torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu

# Dependencies còn lại
COPY requirements.txt .
RUN pip install --no-cache-dir fastapi "uvicorn[standard]"
RUN pip install --no-cache-dir --no-deps vietocr
RUN pip install --no-cache-dir -r requirements.txt


# Runtime stage
FROM python:3.14-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages từ builder
COPY --from=builder /usr/local/lib/python3.14 /usr/local/lib/python3.14
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy source (bỏ qua venv local, log, output)
COPY conf/       conf/
COPY module/     module/
COPY onnx/       onnx/
COPY utils/      utils/
COPY vietocr/    vietocr/
COPY api.py      api.py

# Tạo thư mục log
RUN mkdir -p log

# Model VietOCR sẽ được download lần đầu và cache tại đây
ENV HF_HOME=/app/.cache/huggingface
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface

EXPOSE 8000

# Chạy với 1 worker (OCR model không thread-safe, dùng --workers 1)
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
