"""Cắt dataset VietOCR từ tài liệu THẬT bằng chính pipeline detection của project.

Ý tưởng: chạy detection (DBNet) + recognition (VietOCR) trên tài liệu của bạn,
lưu MỖI DÒNG thành 1 ảnh + label nháp do model đoán. Bạn chỉ việc mở file
annotation lên SỬA những dòng sai — nhanh hơn gõ từ đầu rất nhiều.

Script này CHỈ ĐỌC module/ocr.py, không sửa gì trong đó.

Cách dùng
---------
    # Cắt dòng từ ảnh / PDF / cả thư mục
    .\venv\Scripts\python.exe tools\export_ocr_lines.py --inputs D:\tailieu --output_dir D:\dataset_v1

    # Chỉ lấy dòng model KHÔNG chắc chắn (đây mới là dòng đáng sửa)
    .\venv\Scripts\python.exe tools\export_ocr_lines.py --inputs D:\tailieu --output_dir D:\dataset_v1 --max_score 0.9

Đầu ra
------
    output_dir/
      images/                 ảnh từng dòng
      annotation_raw.txt      <path>\t<text model đoán>   <- SỬA FILE NÀY
      review.txt              danh sách dòng score thấp, ưu tiên sửa trước

Sau khi sửa xong, chia train/val bằng:
    .\venv\Scripts\python.exe tools\export_ocr_lines.py --split D:\dataset_v1\annotation_raw.txt
"""
import argparse
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # cắt dataset không cần GPU

import numpy as np
from PIL import Image


def split_annotation(path: str, val_ratio: float = 0.1, seed: int = 42) -> None:
    """Chia annotation thành train/val. Val phải là ảnh model CHƯA thấy khi train."""
    with open(path, encoding="utf-8") as f:
        rows = [ln for ln in f.read().splitlines() if ln.strip()]

    random.Random(seed).shuffle(rows)
    n_val = max(1, int(len(rows) * val_ratio))
    val, train = rows[:n_val], rows[n_val:]

    # vietocr bỏ dòng cuối cùng của annotation (bug off-by-one trong
    # create_dataset.py:85 -> nSamples = cnt-1). Lặp dòng cuối để không mất mẫu thật.
    if train:
        train.append(train[-1])
    if val:
        val.append(val[-1])

    base = os.path.dirname(path)
    for name, rows_ in (("annotation_train.txt", train), ("annotation_val.txt", val)):
        out = os.path.join(base, name)
        with open(out, "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(rows_) + "\n")
        print(f"{out}: {len(rows_)} dòng (đã +1 dòng bù bug off-by-one)")


def iter_images(inputs: str):
    """Sinh (tên_gốc, PIL.Image) từ ảnh, PDF, hoặc thư mục."""
    import pdfplumber

    def one_file(path):
        ext = path.rsplit(".", 1)[-1].lower() if "." in path else ""
        stem = os.path.splitext(os.path.basename(path))[0]
        if ext == "pdf":
            with pdfplumber.open(path) as pdf:
                for i, page in enumerate(pdf.pages):
                    yield f"{stem}_p{i:03d}", page.to_image(resolution=300).annotated.convert("RGB")
        elif ext in ("jpg", "jpeg", "png", "webp", "bmp", "tif", "tiff"):
            yield stem, Image.open(path).convert("RGB")

    if os.path.isdir(inputs):
        for root, _, files in os.walk(inputs):
            for fn in sorted(files):
                yield from one_file(os.path.join(root, fn))
    else:
        yield from one_file(inputs)


def main(args):
    from module.ocr import OCR, expand_box

    img_dir = os.path.join(args.output_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    ocr = OCR()
    detector = ocr.text_detector[0]

    rows, review, n_img = [], [], 0

    for stem, pil_img in iter_images(args.inputs):
        arr = np.array(pil_img)
        dt_boxes, _ = detector(arr)
        if dt_boxes is None or len(dt_boxes) == 0:
            print(f"  {stem}: không phát hiện dòng nào")
            continue

        dt_boxes = ocr.sorted_boxes(dt_boxes)
        crops = []
        for box in dt_boxes:
            # Dùng ĐÚNG tham số padding của pipeline inference để ảnh dataset
            # khớp với ảnh model sẽ gặp lúc chạy thật
            padded = expand_box(box, pad_x=3, pad_y=10, img_shape=arr.shape)
            crops.append(ocr.get_rotate_crop_image(arr, padded))

        rec_res, _ = ocr.text_recognizer[0](crops)

        for crop, (text, score) in zip(crops, rec_res):
            text = (text or "").strip()
            if not text:
                continue
            if score > args.max_score or score < args.min_score:
                continue
            h, w = crop.shape[:2]
            if h < 8 or w < 8:
                continue

            name = f"{stem}_{n_img:05d}.jpg"
            Image.fromarray(crop).save(os.path.join(img_dir, name), quality=95)
            rows.append(f"images/{name}\t{text}")
            review.append((float(score), name, text))
            n_img += 1

        print(f"  {stem}: {len(dt_boxes)} dòng phát hiện, {n_img} ảnh đã lưu (cộng dồn)")

    ann = os.path.join(args.output_dir, "annotation_raw.txt")
    with open(ann, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(rows) + "\n")

    review.sort(key=lambda r: r[0])
    with open(os.path.join(args.output_dir, "review.txt"), "w", encoding="utf-8", newline="\n") as f:
        f.write("# Sắp xếp theo score tăng dần — sửa từ trên xuống\n")
        f.write("# score\tfile\ttext model đoán\n")
        for s, name, text in review:
            f.write(f"{s:.4f}\t{name}\t{text}\n")

    print(f"\n{len(rows)} dòng -> {ann}")
    print(f"Mở review.txt, sửa các dòng score thấp trong annotation_raw.txt, rồi chạy:")
    print(f"  python tools/export_ocr_lines.py --split {ann}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--inputs", help="Ảnh, PDF, hoặc thư mục chứa tài liệu thật")
    p.add_argument("--output_dir", default="./dataset_v1", help="Thư mục xuất dataset")
    p.add_argument("--min_score", type=float, default=0.0,
                   help="Bỏ dòng có score < ngưỡng (lọc rác detection)")
    p.add_argument("--max_score", type=float, default=1.0,
                   help="Bỏ dòng có score > ngưỡng. Đặt 0.9 để chỉ lấy dòng model chưa chắc chắn")
    p.add_argument("--split", help="Chia file annotation có sẵn thành train/val rồi thoát")
    args = p.parse_args()

    if args.split:
        split_annotation(args.split)
    elif args.inputs:
        main(args)
    else:
        p.error("cần --inputs hoặc --split")
