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

HERE = os.path.dirname(os.path.abspath(__file__))

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
    print("TẢI MODEL")
    print("=" * 70)

    print("\n[1/2] onnx/ — detection, layout, table structure")
    ok_onnx = download_onnx(a.force)

    print("\n[2/2] vietocr/weight/ — recognition")
    ok_w = True
    for rel, url, _ in VIETOCR_WEIGHTS:
        if not a.all and "transformer" in rel:
            print(f"  [bỏ qua] {rel} — chỉ cần khi đổi sang vgg_transformer (--all để tải)")
            continue
        ok_w &= download_file(url, os.path.join(HERE, rel), a.force)

    print("\n" + "=" * 70)
    check()
    if not (ok_onnx and ok_w):
        print("\nCó lỗi khi tải. Xem phần 'Tải từ GitHub Release' trong README.md")
        sys.exit(1)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--check", action="store_true", help="Chỉ kiểm tra, không tải")
    p.add_argument("--force", action="store_true", help="Tải lại kể cả khi đã có")
    p.add_argument("--all", action="store_true", help="Tải cả vgg_transformer.pth (145MB)")
    main(p.parse_args())
