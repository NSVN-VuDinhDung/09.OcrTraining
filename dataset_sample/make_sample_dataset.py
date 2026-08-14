"""Sinh dataset mẫu cho VietOCR — để bạn thấy ĐÚNG format cần có.

Chạy:
    # 15 dòng, chỉ để xem format
    python dataset_sample/make_sample_dataset.py

    # 400 dòng, đủ để chạy train thử và có đường loss có ý nghĩa
    python dataset_sample/make_sample_dataset.py --n 400

Sinh ra ảnh từng DÒNG chữ (không phải cả trang) + file annotation.
Đây là format duy nhất VietOCR nhận. Xem dataset_sample/README.md.

LƯU Ý QUAN TRỌNG: đây là ảnh SINH TỔNG HỢP (render font lên nền + nhiễu giả).
Dùng để hiểu format và test pipeline train chạy được — KHÔNG dùng để fine-tune
thật. Model train trên synthetic thuần thường không cải thiện trên ảnh scan
thật. Dataset thật phải cắt từ chính tài liệu bạn xử lý:
    python tools/export_ocr_lines.py --inputs <tài liệu> --output_dir <đích>
"""
import argparse
import os
import random

from PIL import Image, ImageDraw, ImageFilter, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(HERE, "images")

# --- Kho ngữ liệu: ghép mảnh để sinh nhiều câu khác nhau -------------------
# Ưu tiên ký tự 2 tầng dấu (ắ ẳ ộ ữ ự ể) — đúng điểm yếu của pipeline
HEADS = [
    "Hợp đồng bảo hiểm nhân thọ", "Giấy chứng nhận quyền sử dụng đất",
    "Đơn đề nghị cấp lại", "Biên bản nghiệm thu công trình",
    "Phụ lục hợp đồng kinh tế", "Quyết định bổ nhiệm cán bộ",
    "Thông báo nộp phí bảo hiểm", "Giấy ủy quyền sử dụng tài khoản",
    "Bảng kê chi tiết các khoản mục", "Tờ khai thuế thu nhập cá nhân",
    "Đề nghị thanh toán chi phí", "Biên nhận hồ sơ gốc",
]
MIDS = ["số", "mã số", "ký hiệu", "số hiệu", "số thứ tự", "mã hồ sơ"]
NAMES = [
    "Nguyễn Thị Bích Ngọc", "Trần Đức Thắng", "Lê Hoàng Mỹ Duyên",
    "Phạm Quốc Cường", "Vũ Đình Dũng", "Hoàng Thị Tuyết Mai",
    "Đặng Ngọc Ánh", "Bùi Xuân Trường", "Đỗ Thị Kiều Oanh",
    "Ngô Việt Hưng", "Dương Thuỳ Linh", "Lý Bảo Khánh",
]
PLACES = [
    "Thành phố Hồ Chí Minh", "Quận Cầu Giấy, Thành phố Hà Nội",
    "Tỉnh Bà Rịa - Vũng Tàu", "Huyện Đức Trọng, Tỉnh Lâm Đồng",
    "Phường Bến Nghé, Quận 1", "Thị xã Sơn Tây, Hà Nội",
    "Tỉnh Thừa Thiên Huế", "Quận Hải Châu, Đà Nẵng",
]
CLAUSES = [
    "Điều {n}: Phạm vi bảo hiểm và quyền lợi được hưởng",
    "Điều {n}: Nghĩa vụ đóng phí của bên mua bảo hiểm",
    "Điều {n}: Thời hạn hiệu lực và điều kiện gia hạn",
    "Điều {n}: Trách nhiệm bồi thường khi xảy ra sự kiện",
    "Điều {n}: Điều khoản loại trừ trách nhiệm bảo hiểm",
    "Điều {n}: Giải quyết tranh chấp và luật áp dụng",
]
MONEY_LABELS = [
    "Tổng số tiền", "Số tiền bảo hiểm", "Mức khấu trừ", "Phí thường niên",
    "Giá trị hoàn lại", "Tổng cộng phải nộp", "Số tiền tạm ứng",
]
MISC = [
    "Kính gửi: Ông/Bà {name} - Trưởng phòng Kế toán",
    "Người đại diện theo pháp luật: {name}",
    "Đề nghị quý khách kiểm tra kỹ trước khi ký xác nhận",
    "Xác nhận đã nhận đủ hồ sơ gốc và bản sao y",
    "Nơi cấp: {place}",
    "Ngày ký: {d:02d}/{m:02d}/{y} tại {place}",
    "Thời hạn hiệu lực: từ {d:02d}/{m:02d}/{y} đến hết 31/12/{y2}",
    "Số tài khoản: {acc} tại Ngân hàng TMCP Ngoại thương Việt Nam",
    "Địa chỉ thường trú: số {sn} đường Nguyễn Văn Cừ, {place}",
    "Chữ ký và ghi rõ họ tên của người nhận: {name}",
]

FONT_CANDIDATES = [
    r"C:\Windows\Fonts\times.ttf",
    r"C:\Windows\Fonts\arial.ttf",
    r"C:\Windows\Fonts\calibri.ttf",
    r"C:\Windows\Fonts\segoeui.ttf",
    r"C:\Windows\Fonts\tahoma.ttf",
    r"C:\Windows\Fonts\verdana.ttf",
    r"C:\Windows\Fonts\georgia.ttf",
    r"C:\Windows\Fonts\consola.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]


def money(rng):
    v = rng.choice([rng.randrange(1, 999) * 10**6, rng.randrange(1, 99) * 10**7,
                    rng.randrange(100, 999) * 10**3])
    return f"{v:,}".replace(",", ".")


def gen_line(rng) -> str:
    kind = rng.randrange(6)
    if kind == 0:
        return f"{rng.choice(HEADS)} {rng.choice(MIDS)} {rng.randrange(10**8, 10**10)}"
    if kind == 1:
        return rng.choice(CLAUSES).format(n=rng.randrange(1, 40))
    if kind == 2:
        return f"{rng.choice(MONEY_LABELS)}: {money(rng)} đồng"
    if kind == 3:
        return f"{rng.choice(HEADS)} - {rng.choice(PLACES)}"
    if kind == 4:
        return (f"{rng.choice(NAMES)}, sinh ngày {rng.randrange(1, 29):02d}/"
                f"{rng.randrange(1, 13):02d}/19{rng.randrange(60, 99)}")
    y = rng.randrange(2020, 2030)
    return rng.choice(MISC).format(
        name=rng.choice(NAMES), place=rng.choice(PLACES),
        d=rng.randrange(1, 29), m=rng.randrange(1, 13), y=y, y2=y + 5,
        acc=f"{rng.randrange(1000, 9999)} {rng.randrange(1000, 9999)} {rng.randrange(1000, 9999)}",
        sn=rng.randrange(1, 500),
    )


def pick_fonts():
    found = [p for p in FONT_CANDIDATES if os.path.exists(p)]
    if not found:
        raise SystemExit("Không tìm thấy font hỗ trợ tiếng Việt")
    return found


def render(text, font_path, size, level, rng) -> Image.Image:
    """level 0 = sạch, 1 = nhiễu nhẹ, 2 = nhiễu mạnh (giả lập scan kém)."""
    font = ImageFont.truetype(font_path, size)
    tmp = ImageDraw.Draw(Image.new("L", (1, 1)))
    x0, y0, x1, y1 = tmp.textbbox((0, 0), text, font=font)
    pad_x, pad_y = rng.randrange(4, 14), rng.randrange(3, 10)
    w, h = x1 - x0 + pad_x * 2, y1 - y0 + pad_y * 2

    bg = 255 if level == 0 else rng.randrange(228, 253)
    fg = 0 if level == 0 else rng.randrange(0, 75)
    img = Image.new("L", (w, h), color=bg)
    ImageDraw.Draw(img).text((pad_x - x0, pad_y - y0), text, fill=fg, font=font)

    if level >= 1:
        px = img.load()
        amount = 0.010 if level == 1 else 0.035
        spread = 40 if level == 1 else 70
        for _ in range(int(w * h * amount)):
            xx, yy = rng.randrange(w), rng.randrange(h)
            px[xx, yy] = max(0, min(255, px[xx, yy] + rng.randint(-spread, spread)))
        angle = rng.uniform(-0.7, 0.7) if level == 1 else rng.uniform(-1.6, 1.6)
        img = img.rotate(angle, resample=Image.BICUBIC, fillcolor=bg, expand=True)
    if level == 2:
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 0.9)))

    return img.convert("RGB")


def main(n, seed, val_ratio):
    rng = random.Random(seed)
    os.makedirs(IMG_DIR, exist_ok=True)
    for f in os.listdir(IMG_DIR):
        if f.endswith(".jpg"):
            os.remove(os.path.join(IMG_DIR, f))

    fonts = pick_fonts()
    print(f"Font dùng được: {len(fonts)}")

    rows, seen = [], set()
    i = 0
    while len(rows) < n:
        text = gen_line(rng)
        if text in seen:          # tránh trùng -> val không bị lộ vào train
            continue
        seen.add(text)
        img = render(text, rng.choice(fonts), rng.choice([18, 20, 24, 28, 32, 36]),
                     rng.choices([0, 1, 2], weights=[2, 5, 3])[0], rng)
        name = f"line_{i:05d}.jpg"
        img.save(os.path.join(IMG_DIR, name), quality=rng.randrange(80, 97))
        rows.append((f"images/{name}", text))
        i += 1

    rng.shuffle(rows)
    n_val = max(2, int(len(rows) * val_ratio))
    val, train = rows[:n_val], rows[n_val:]

    # vietocr bỏ dòng CUỐI của annotation (bug off-by-one, create_dataset.py:85
    # -> nSamples = cnt-1). Lặp dòng cuối để không mất mẫu thật.
    for fname, subset in (("annotation_train.txt", train), ("annotation_val.txt", val)):
        subset = subset + [subset[-1]]
        path = os.path.join(HERE, fname)
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            for img_path, text in subset:
                f.write(f"{img_path}\t{text}\n")
        print(f"{fname}: {len(subset)} dòng (gồm 1 dòng bù bug off-by-one)")

    print(f"Ảnh: {len(rows)} file trong {IMG_DIR}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=15, help="Số dòng sinh ra")
    p.add_argument("--seed", type=int, default=20260814)
    p.add_argument("--val_ratio", type=float, default=0.15)
    a = p.parse_args()
    main(a.n, a.seed, a.val_ratio)
