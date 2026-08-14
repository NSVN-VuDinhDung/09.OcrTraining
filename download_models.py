"""Tải model về sau khi clone repo.

    python download_models.py              # tải tất cả những gì còn thiếu
    python download_models.py --check      # chỉ kiểm tra, không tải
    python download_models.py --force      # tải lại kể cả khi đã có

Vì sao repo không chứa model
----------------------------
Tổng model ~810MB. Đưa vào Git thì:
  - file 145MB vượt giới hạn cứng 100MB của GitHub -> push bị từ chối
  - dùng Git LFS thì tiêu gần hết 1GB quota free, và mỗi lần clone tốn bandwidth;
    hết quota là kẹt CẢ push lẫn clone, mà đã đẩy lên LFS thì rất khó gỡ

Model vốn tải được từ nguồn gốc, nên script này dựng lại đầy đủ.

Nguồn
-----
  onnx/                  HuggingFace `InfiniFlow/deepdoc`   (detection, layout, TSR)
  vietocr/weight/*.pth   https://vocr.vn/data/vietocr/       (recognition)

Nếu nguồn ngoài không truy cập được, xem phần "Tải từ GitHub Release"
trong README.md — bản sao dự phòng nằm ở đó.
"""
import argparse
import os
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))

# GitHub Release — bản sao dự phòng, không tính vào LFS quota
RELEASE_TAG = "models-v1"
RELEASE_BASE = (f"https://github.com/NSVN-VuDinhDung/09.OcrTraining"
                f"/releases/download/{RELEASE_TAG}")
RELEASE_ZIPS = {
    "core": ("models-core.zip", 466,
             "onnx/ đầy đủ + vgg_seq2seq.pth — đủ để pipeline chạy"),
    "optional": ("models-optional.zip", 309,
                 "vgg_transformer.pth, transformerocr.pth, cnn/encoder/decoder.onnx"),
}

# (đường dẫn đích, URL, kích thước xấp xỉ MB)
VIETOCR_WEIGHTS = [
    ("vietocr/weight/vgg_seq2seq.pth", "https://vocr.vn/data/vietocr/vgg_seq2seq.pth", 86),
    ("vietocr/weight/vgg_transformer.pth", "https://vocr.vn/data/vietocr/vgg_transformer.pth", 145),
]

# File onnx bắt buộc để pipeline chạy được
ONNX_REQUIRED = ["det.onnx", "layout.onnx", "tsr.onnx"]


def human(n):
    return f"{n / 1024 / 1024:.1f} MB"


def download_file(url, dest, force=False):
    import requests

    if os.path.exists(dest) and not force:
        print(f"  [bỏ qua] {dest} đã có ({human(os.path.getsize(dest))})")
        return True

    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print(f"  [tải] {url}")
    try:
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            done = 0
            tmp = dest + ".part"
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
                    done += len(chunk)
                    if total:
                        pct = done * 100 // total
                        print(f"\r        {pct:3d}%  {human(done)} / {human(total)}",
                              end="", flush=True)
            print()
            os.replace(tmp, dest)
        print(f"  [xong] {dest} ({human(os.path.getsize(dest))})")
        return True
    except Exception as exc:
        print(f"  [LỖI] {dest}: {exc}")
        if os.path.exists(dest + ".part"):
            os.remove(dest + ".part")
        return False


def download_onnx(force=False):
    onnx_dir = os.path.join(HERE, "onnx")
    have = [f for f in ONNX_REQUIRED if os.path.exists(os.path.join(onnx_dir, f))]
    if len(have) == len(ONNX_REQUIRED) and not force:
        print(f"  [bỏ qua] onnx/ đã đủ {len(have)}/{len(ONNX_REQUIRED)} file bắt buộc")
        return True

    print("  [tải] HuggingFace InfiniFlow/deepdoc (~400MB, mất vài phút)")
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("  [LỖI] thiếu huggingface_hub: pip install huggingface_hub")
        return False

    try:
        snapshot_download(repo_id="InfiniFlow/deepdoc", local_dir=onnx_dir)
        print("  [xong] onnx/")
        return True
    except Exception as exc:
        print(f"  [LỖI] {exc}")
        return False


def download_from_release(which, force=False):
    """Tải zip từ GitHub Release rồi giải nén vào đúng thư mục.

    Zip giữ nguyên cấu trúc `onnx/...` và `vietocr/weight/...` tính từ gốc repo,
    nên giải nén tại gốc là đúng chỗ. Đường dẫn trong zip dùng "/" nên giải nén
    được cả trên Windows lẫn Linux/Colab.
    """
    fname, mb, desc = RELEASE_ZIPS[which]
    url = f"{RELEASE_BASE}/{fname}"
    tmp = os.path.join(HERE, fname)

    print(f"  {fname} (~{mb} MB) — {desc}")
    if not download_file(url, tmp, force=True):
        return False

    print(f"  [giải nén] {fname}")
    try:
        with zipfile.ZipFile(tmp) as zf:
            zf.extractall(HERE)
        print(f"  [xong] giải nén {len(zipfile.ZipFile(tmp).namelist())} file")
    except Exception as exc:
        print(f"  [LỖI] giải nén {fname}: {exc}")
        return False
    finally:
        os.remove(tmp)
    return True


def check():
    print("Kiểm tra model:")
    missing = 0
    onnx_dir = os.path.join(HERE, "onnx")
    for f in ONNX_REQUIRED:
        p = os.path.join(onnx_dir, f)
        if os.path.exists(p):
            print(f"  OK      onnx/{f}  ({human(os.path.getsize(p))})")
        else:
            print(f"  THIẾU   onnx/{f}")
            missing += 1
    for rel, _, mb in VIETOCR_WEIGHTS:
        p = os.path.join(HERE, rel)
        if os.path.exists(p):
            print(f"  OK      {rel}  ({human(os.path.getsize(p))})")
        else:
            print(f"  THIẾU   {rel}  (~{mb} MB)")
            missing += 1
    print(f"\n{missing} file còn thiếu" if missing else "\nĐầy đủ.")
    return missing


def main(a):
    if a.check:
        sys.exit(1 if check() else 0)

    print("=" * 70)
    print("TẢI MODEL " + ("từ GitHub Release" if a.release else "từ nguồn gốc"))
    print("=" * 70)

    if a.release:
        ok = download_from_release("core", a.force)
        if a.all:
            ok &= download_from_release("optional", a.force)
        else:
            print(f"  [bỏ qua] {RELEASE_ZIPS['optional'][0]} — dùng --all nếu cần")
    else:
        print("\n[1/2] onnx/ — detection, layout, table structure")
        ok = download_onnx(a.force)

        print("\n[2/2] vietocr/weight/ — recognition")
        for rel, url, _ in VIETOCR_WEIGHTS:
            if not a.all and "vgg_transformer" in rel:
                print(f"  [bỏ qua] {rel} — chỉ cần khi đổi sang vgg_transformer (--all để tải)")
                continue
            ok &= download_file(url, os.path.join(HERE, rel), a.force)

    print("\n" + "=" * 70)
    check()
    if not ok:
        if a.release:
            print("\nTải từ Release lỗi. Thử nguồn gốc: python download_models.py")
        else:
            print("\nTải từ nguồn gốc lỗi. Thử bản sao trên Release:"
                  "\n  python download_models.py --release")
        sys.exit(1)


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--check", action="store_true", help="Chỉ kiểm tra, không tải")
    p.add_argument("--force", action="store_true", help="Tải lại kể cả khi đã có")
    p.add_argument("--all", action="store_true",
                   help="Tải cả model tuỳ chọn (vgg_transformer, ocr_onnx...)")
    p.add_argument("--release", action="store_true",
                   help="Tải zip từ GitHub Release thay vì nguồn gốc "
                        "(dùng khi HuggingFace/vocr.vn không truy cập được)")
    main(p.parse_args())
