"""Đóng gói model thành zip để upload lên GitHub Release.

    python make_release_zips.py

Sinh ra trong release_assets/ (đã gitignore):
    models-core.zip      onnx/ đầy đủ + vgg_seq2seq.pth   — đủ để pipeline chạy
    models-optional.zip  vgg_transformer, transformerocr, cnn/encoder/decoder.onnx

Sau đó upload (tag phải khớp RELEASE_TAG trong download_models.py):

    gh release create models-v1 release_assets/models-core.zip \
        release_assets/models-optional.zip \
        --title "Model weights v1" \
        --notes "onnx/ (InfiniFlow/deepdoc) + vietocr/weight/. Giải nén tại gốc repo."

Hai chi tiết quan trọng về cách zip được tạo:

1. Đường dẫn trong zip dùng "/" chứ không phải "\\", và tính từ GỐC REPO
   (`onnx/det.onnx`, `vietocr/weight/vgg_seq2seq.pth`). Nhờ vậy giải nén tại gốc
   là đúng chỗ, và giải nén được cả trên Windows lẫn Linux/Colab.
   `Compress-Archive` của PowerShell ghi dấu "\\" vào entry name — `unzip` trên
   Linux sẽ tạo ra file tên literal `onnx\\det.onnx`. Vì thế script này dùng
   `zipfile` của Python.

2. Bỏ `onnx/.cache/` — metadata do `snapshot_download` của HuggingFace để lại,
   không cần thiết.
"""
import os
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "release_assets")
SKIP_DIRS = {".cache", "__pycache__"}

# (tên zip, danh sách nguồn). Nguồn là str = cả thư mục, hoặc (thư mục, [file...])
BUNDLES = [
    ("models-core", [
        "onnx",
        ("vietocr/weight", ["vgg_seq2seq.pth"]),
    ]),
    ("models-optional", [
        ("vietocr/weight", ["vgg_transformer.pth", "transformerocr.pth",
                            "cnn.onnx", "encoder.onnx", "decoder.onnx"]),
    ]),
]


def collect(spec):
    files = []
    for item in spec:
        if isinstance(item, str):
            root_dir = os.path.join(HERE, item)
            if not os.path.isdir(root_dir):
                print(f"  [BỎ QUA] không có thư mục {item}/")
                continue
            for root, dirs, fns in os.walk(root_dir):
                dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
                for fn in fns:
                    p = os.path.join(root, fn)
                    arc = os.path.relpath(p, HERE).replace(os.sep, "/")
                    files.append((p, arc))
        else:
            d, names = item
            for n in names:
                p = os.path.join(HERE, d, n)
                if not os.path.exists(p):
                    print(f"  [BỎ QUA] không có {d}/{n}")
                    continue
                files.append((p, f"{d}/{n}"))
    return files


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for name, spec in BUNDLES:
        print(f"=== {name}.zip ===")
        files = collect(spec)
        if not files:
            print("  không có file nào — bỏ qua\n")
            continue
        raw = sum(os.path.getsize(p) for p, _ in files)
        out = os.path.join(OUT_DIR, f"{name}.zip")
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
            for p, arc in files:
                zf.write(p, arc)
        size = os.path.getsize(out)
        with zipfile.ZipFile(out) as zf:
            bad = zf.testzip()
        print(f"  {len(files)} file | raw {raw / 1e6:.1f} MB -> zip {size / 1e6:.1f} MB")
        print(f"  toàn vẹn: {'LỖI ' + str(bad) if bad else 'OK'}\n")

    print(f"Zip nằm trong {OUT_DIR}")
    print("Upload:  gh release create models-v1 release_assets/*.zip "
          '--title "Model weights v1" --notes "Giải nén tại gốc repo."')


if __name__ == "__main__":
    main()
