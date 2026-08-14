r"""Fine-tune VietOCR — script train cho project này.

Chạy (Windows):
    .\venv_train\Scripts\python.exe train.py --iters 500

Chạy trên Colab:
    !python train.py --iters 15000 --device cuda:0 --num_workers 2

Xem training_guide.md để hiểu ý nghĩa từng tham số.

Điểm khác so với chạy Trainer trần:
  1. import compat       -> vá numpy 2.x + LMDB Windows (xem compat.py)
  2. Xoá LMDB cũ         -> tránh train nhầm dữ liệu cũ (bug cache của vietocr)
  3. Đo BASELINE trước   -> có mốc so sánh, nếu không thì con số accuracy vô nghĩa
  4. Ghi log kèm mốc thời gian
"""
import compat  # noqa: F401  — PHẢI đứng trước mọi import vietocr

import argparse
import json
import os
import shutil
import time
from datetime import datetime

from vietocr.model.trainer import Trainer
from vietocr.tool.config import Cfg

HERE = os.path.dirname(os.path.abspath(__file__))


def check_dataset(root, ann_files, vocab):
    """Bắt lỗi dataset TRƯỚC khi train — tiết kiệm hàng giờ."""
    ok = True
    for ann in ann_files:
        path = os.path.join(root, ann)
        if not os.path.exists(path):
            print(f"  [LỖI] không có {path}")
            ok = False
            continue
        bad_fmt = missing = n = 0
        out_vocab = set()
        for line in open(path, encoding="utf-8"):
            line = line.rstrip("\n")
            if not line.strip():
                continue
            n += 1
            parts = line.split("\t")
            if len(parts) != 2:
                bad_fmt += 1
                continue
            img, label = parts
            if not os.path.exists(os.path.join(root, img)):
                missing += 1
            out_vocab |= {c for c in label if c not in vocab}
        status = "OK" if (bad_fmt == 0 and missing == 0 and not out_vocab) else "CÓ VẤN ĐỀ"
        print(f"  {ann}: {n} dòng | sai format {bad_fmt} | thiếu ảnh {missing} | "
              f"ký tự ngoài vocab {sorted(out_vocab) if out_vocab else 'không'} -> {status}")
        if bad_fmt or missing:
            ok = False
    return ok


def main(a):
    t_start = time.time()
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("=" * 78)
    print(f"BẮT ĐẦU: {stamp}")
    print("=" * 78)

    config = Cfg.load_config_from_name("vgg_seq2seq")

    # --- weight khởi điểm -------------------------------------------------
    # Trainer nạp từ config['pretrain'] (KHÔNG phải config['weights']).
    # download_weights() trả nguyên đường dẫn nếu không bắt đầu bằng http,
    # nên trỏ file local là fine-tune từ file đó. Để resume phiên trước,
    # trỏ pretrain vào weight đã export.
    config["pretrain"] = a.pretrain
    config["device"] = a.device

    # --- dataset ----------------------------------------------------------
    config["dataset"]["data_root"] = a.data_root
    config["dataset"]["train_annotation"] = a.train_ann
    config["dataset"]["valid_annotation"] = a.valid_ann
    config["dataset"]["name"] = a.name

    # --- siêu tham số -----------------------------------------------------
    config["optimizer"]["max_lr"] = a.max_lr        # THẤP để không xoá kiến thức cũ
    config["optimizer"]["pct_start"] = 0.1
    config["trainer"]["batch_size"] = a.batch_size
    config["trainer"]["iters"] = a.iters
    config["trainer"]["print_every"] = a.print_every
    config["trainer"]["valid_every"] = a.valid_every  # nhỏ -> lưu file thường xuyên
    config["trainer"]["metrics"] = a.metrics

    # --- đầu ra -----------------------------------------------------------
    out = os.path.join(HERE, a.output_dir)
    os.makedirs(out, exist_ok=True)
    config["trainer"]["export"] = os.path.join(out, f"vgg_seq2seq_{a.name}.pth")
    config["trainer"]["checkpoint"] = os.path.join(out, f"ckpt_{a.name}.pth")
    config["trainer"]["log"] = os.path.join(out, f"train_{a.name}.log")

    config["dataloader"]["num_workers"] = a.num_workers
    config["dataloader"]["pin_memory"] = a.device.startswith("cuda")

    print("\n[1/5] Kiểm tra dataset")
    if not check_dataset(a.data_root, [a.train_ann, a.valid_ann], config["vocab"]):
        raise SystemExit("Dataset có lỗi — sửa trước khi train")

    print("\n[2/5] Xoá LMDB cũ (tránh train nhầm dữ liệu của lần trước)")
    for d in (f"train_{a.name}", f"valid_{a.name}"):
        if os.path.isdir(d):
            shutil.rmtree(d, ignore_errors=True)
            print(f"  đã xoá {d}/")
        else:
            print(f"  {d}/ chưa có")

    cfg_path = os.path.join(out, f"config_{a.name}.yml")
    config.save(cfg_path)
    print(f"\n  config đã lưu: {cfg_path}")

    print("\n[3/5] Khởi tạo Trainer (nạp pretrain + build LMDB)")
    t0 = time.time()
    trainer = Trainer(config)
    print(f"  xong sau {time.time() - t0:.1f}s")

    print("\n[4/5] BASELINE — đo accuracy của weight gốc TRƯỚC khi train")
    print("  (không có mốc này thì con số sau khi train vô nghĩa)")
    t0 = time.time()
    base_full, base_char = trainer.precision(a.metrics)
    print(f"  acc full seq = {base_full:.4f} | acc per char = {base_char:.4f} "
          f"({time.time() - t0:.1f}s)")

    print(f"\n[5/5] TRAIN {a.iters} iters, batch {a.batch_size}, "
          f"max_lr {a.max_lr}, device {a.device}")
    t0 = time.time()
    trainer.train()
    train_secs = time.time() - t0

    print("\n" + "=" * 78)
    print("SAU KHI TRAIN")
    final_full, final_char = trainer.precision(a.metrics)
    # compute_accuracy() trả np.float32 cho per_char -> json.dump ném
    # "TypeError: Object of type float32 is not JSON serializable"
    base_full, base_char = float(base_full), float(base_char)
    final_full, final_char = float(final_full), float(final_char)
    print(f"  baseline : full_seq {base_full:.4f} | per_char {base_char:.4f}")
    print(f"  sau train: full_seq {final_full:.4f} | per_char {final_char:.4f}")
    print(f"  thay đổi : full_seq {final_full - base_full:+.4f} | "
          f"per_char {final_char - base_char:+.4f}")

    exported = config["trainer"]["export"]
    if os.path.exists(exported):
        mb = os.path.getsize(exported) / 1024 / 1024
        print(f"\n  Weight đã xuất: {exported} ({mb:.1f} MB)")
    else:
        print("\n  [CẢNH BÁO] KHÔNG có file weight nào được ghi!")
        print("  vietocr chỉ gọi save_weights() khi acc_full_seq vượt kỷ lục cũ.")
        print("  Nghĩa là accuracy chưa từng cải thiện. Giảm valid_every hoặc xem lại data.")

    summary = {
        "started": stamp,
        "finished": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_seconds": round(time.time() - t_start, 1),
        "train_seconds": round(train_secs, 1),
        "sec_per_iter": round(train_secs / a.iters, 3),
        "device": a.device,
        "iters": a.iters,
        "batch_size": a.batch_size,
        "max_lr": a.max_lr,
        "baseline": {"full_seq": round(base_full, 4), "per_char": round(base_char, 4)},
        "final": {"full_seq": round(final_full, 4), "per_char": round(final_char, 4)},
        "export": exported if os.path.exists(exported) else None,
    }
    sp = os.path.join(out, f"summary_{a.name}.json")
    with open(sp, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"  Tóm tắt: {sp}")
    print(f"  Tổng thời gian: {summary['total_seconds']}s "
          f"({summary['sec_per_iter']}s/iter)")
    print("=" * 78)


if __name__ == "__main__":
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--name", default="smoke", help="Tên run — quyết định tên LMDB và file xuất")
    p.add_argument("--data_root", default="./dataset_sample/")
    p.add_argument("--train_ann", default="annotation_train.txt")
    p.add_argument("--valid_ann", default="annotation_val.txt")
    p.add_argument("--pretrain", default=r"vietocr\weight\vgg_seq2seq.pth",
                   help="Weight khởi điểm. Đường dẫn local hoặc URL http")
    p.add_argument("--output_dir", default="train_output")
    p.add_argument("--device", default="cpu", help="cpu | cuda:0")
    p.add_argument("--iters", type=int, default=500)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_lr", type=float, default=0.0001, help="THẤP khi fine-tune")
    p.add_argument("--print_every", type=int, default=25)
    p.add_argument("--valid_every", type=int, default=100)
    p.add_argument("--metrics", type=int, default=None,
                   help="Số mẫu val để tính accuracy. None = toàn bộ")
    p.add_argument("--num_workers", type=int, default=0,
                   help="0 trên Windows (multiprocessing hay treo), 2 trên Colab")
    main(p.parse_args())
