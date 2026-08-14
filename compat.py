"""Vá tương thích cho vietocr 0.3.13 + imgaug 0.4.0 trên môi trường hiện đại.

    import compat        # PHẢI đứng trước mọi import từ vietocr

Ba nhóm vấn đề, tất cả đã kiểm chứng thực tế (xem train_log.md):

A. numpy 2.x đã gỡ API mà vietocr/imgaug còn dùng
B. LMDB map_size 1TB không tạo được 2 lần trên Windows
C. (thông tin) OneCycleLR chia cho 0 khi iters quá nhỏ — không vá, xem ghi chú cuối file
"""
import os

import numpy as np

_PATCHED = []

# ---------------------------------------------------------------------------
# A. numpy 2.x
# ---------------------------------------------------------------------------
# | Thư viện | Chỗ dùng                                  | API bị gỡ      |
# |----------|-------------------------------------------|----------------|
# | vietocr  | loader/dataloader.py:71 get_bucket()      | np.fromstring  |
# | vietocr  | tool/create_dataset.py:13 checkImageIsValid | np.fromstring |
# | imgaug   | imgaug/imgaug.py:45                       | np.sctypes     |
#
# Cách sạch hơn là `pip install "numpy<2"`, nhưng numpy 1.x KHÔNG có wheel cho
# Python 3.14 (numpy 1.26 chỉ tới Python 3.12). Trên Colab (Python 3.11/3.12)
# thì cứ hạ numpy, đơn giản và ít rủi ro hơn.


def _patch_fromstring():
    """np.fromstring chế độ binary -> np.frombuffer.

    numpy 2 vẫn giữ np.fromstring cho chế độ text (có `sep`), chỉ chế độ binary
    ném ValueError. Giữ nguyên đường text, chuyển đường binary sang frombuffer
    (đúng thứ numpy khuyến nghị trong chính thông báo lỗi).
    """
    orig = np.fromstring

    def fromstring(string, dtype=float, count=-1, sep="", like=None):
        if sep == "":
            return np.frombuffer(string, dtype=dtype, count=count)
        return orig(string, dtype=dtype, count=count, sep=sep)

    np.fromstring = fromstring
    _PATCHED.append("np.fromstring -> np.frombuffer (chế độ binary)")


def _patch_sctypes():
    """Dựng lại np.sctypes đúng như nội dung ở numpy 1.x.

    imgaug chỉ đọc np.sctypes['float'] và ['int'] để lập bảng dtype hợp lệ.
    """
    if hasattr(np, "sctypes"):
        return
    np.sctypes = {
        "int": [np.int8, np.int16, np.int32, np.int64],
        "uint": [np.uint8, np.uint16, np.uint32, np.uint64],
        "float": [np.float16, np.float32, np.float64],
        "complex": [np.complex64, np.complex128],
        "others": [bool, object, bytes, str, np.void],
    }
    _PATCHED.append("np.sctypes (dựng lại từ numpy 1.x)")


def _patch_aliases():
    """np.bool / np.float / np.int / np.complex — alias bị gỡ ở numpy 1.24.

    KHÔNG đụng np.object và np.str: numpy 2 dự định định nghĩa lại chúng thành
    scalar type, chỉ riêng hasattr() đã kích hoạt FutureWarning. Stack này
    không cần chúng.
    """
    for name, target in (("bool", bool), ("float", float),
                         ("int", int), ("complex", complex)):
        if name not in np.__dict__:
            setattr(np, name, target)
            _PATCHED.append(f"np.{name} -> builtin {target.__name__}")


# ---------------------------------------------------------------------------
# B. LMDB map_size trên Windows
# ---------------------------------------------------------------------------
# vietocr/tool/create_dataset.py:47
#     env = lmdb.open(outputPath, map_size=1099511627776)     # 1 TiB
#
# Trên Linux đây là sparse mapping, gần như miễn phí. Trên Windows, LMDB tạo
# ngay file có kích thước khai báo. Tạo được MỘT map 1TB, nhưng tới map thứ hai
# (dataset valid) thì:
#     lmdb.Error: valid_x: Insufficient system resources exist to complete
#                          the requested service.
#
# Trainer luôn tạo 2 dataset (train + valid) -> luôn vỡ trên Windows.
# Vá bằng cách chặn trần map_size. 4 GiB thừa sức cho vài trăm nghìn ảnh dòng
# (dataset 400 ảnh chỉ chiếm 3.8 MB thật).

LMDB_MAP_SIZE = int(os.environ.get("LMDB_MAP_SIZE", 4 * 1024**3))  # 4 GiB


def _patch_lmdb_map_size():
    try:
        import lmdb
    except ImportError:
        return

    orig_open = lmdb.open

    def open_capped(path, **kw):
        ms = kw.get("map_size")
        if ms and ms > LMDB_MAP_SIZE:
            kw["map_size"] = LMDB_MAP_SIZE
        return orig_open(path, **kw)

    lmdb.open = open_capped
    _PATCHED.append(f"lmdb.open map_size <= {LMDB_MAP_SIZE / 1024**3:.0f} GiB")


# ---------------------------------------------------------------------------
# C. ClusterRandomSampler vứt bỏ batch không đầy  <-- LỖI NGHIÊM TRỌNG NHẤT
# ---------------------------------------------------------------------------
# vietocr/loader/dataloader.py:137
#     batches = [_ for _ in batches if len(_) == self.batch_size]
#
# Ảnh được gom thành "cụm" theo chiều rộng sau khi resize về cao 32px, để mỗi
# batch có ảnh cùng bề rộng (giảm padding). Nhưng dòng trên CHỈ GIỮ batch đầy
# đúng batch_size — mọi phần dư của mỗi cụm bị vứt, KHÔNG cảnh báo gì.
#
# Đo thực tế trên dataset 400 dòng, batch_size=32:
#     train: 340 mẫu, 32 cụm -> chỉ giữ  32 mẫu   (9%)
#     valid:  60 mẫu, 24 cụm -> chỉ giữ   0 mẫu   (0%)
#
# Valid rỗng làm Trainer.predict() không chạy vòng lặp nào rồi ném
# `UnboundLocalError: cannot access local variable 'prob'` — thông báo lỗi
# chẳng liên quan gì tới nguyên nhân thật.
#
# Nguy hiểm hơn là trường hợp KHÔNG crash: nếu tình cờ có một cụm đầy, bạn
# train ngon lành trên 9% dữ liệu và không bao giờ biết.
#
# Giữ batch lẻ là an toàn: mỗi cụm vốn đã đồng nhất chiều rộng, nên batch lẻ
# vẫn không phải padding thêm gì.


# Vì sao bộ lọc đó tồn tại (và vì sao không thể chỉ bỏ nó đi):
#     vietocr/model/trainer.py:315
#         DataLoader(dataset, batch_size=..., sampler=ClusterRandomSampler(...), ...)
#
# Với `sampler` + `batch_size`, DataLoader tự cắt luồng index thành từng khối
# `batch_size`. Bộ lọc kia đảm bảo mọi khối do sampler phát ra đều đúng
# batch_size, nhờ đó ranh giới cắt của DataLoader trùng khít ranh giới cụm.
#
# Nếu chỉ bỏ bộ lọc, ranh giới lệch -> một batch trộn ảnh từ nhiều cụm khác
# chiều rộng -> Collator vỡ:
#     ValueError: setting an array element with a sequence.
#                 The requested array has an inhomogeneous shape after 3 dimensions
#
# Cách đúng: chuyển sang `batch_sampler` — DataLoader dùng nguyên các batch ta
# đưa, không cắt lại. Mỗi batch vẫn lấy từ đúng MỘT cụm nên đồng nhất chiều
# rộng, và batch lẻ được giữ.


class _ClusterBatchSampler:
    """Phát ra từng batch index, mỗi batch lấy từ đúng một cụm chiều rộng.

    Thay cho ClusterRandomSampler + batch_size của DataLoader. Giữ batch lẻ
    nên không mất dữ liệu, mà vẫn đảm bảo mỗi batch đồng nhất chiều rộng.
    """

    def __init__(self, data_source, batch_size, shuffle=True):
        self.data_source = data_source
        self.batch_size = batch_size
        self.shuffle = shuffle

    def _batches(self):
        import random as _random
        out = []
        for _cluster, indices in self.data_source.cluster_indices.items():
            idx = list(indices)
            if self.shuffle:
                _random.shuffle(idx)
            out += [idx[i:i + self.batch_size]
                    for i in range(0, len(idx), self.batch_size)]
        if self.shuffle:
            _random.shuffle(out)
        return out

    def __iter__(self):
        return iter(self._batches())

    def __len__(self):
        return len(self._batches())


def _patch_data_gen():
    try:
        from torch.utils.data import DataLoader
        from vietocr.loader.dataloader import Collator, OCRDataset
        from vietocr.model.trainer import Trainer
    except ImportError:
        return

    def data_gen(self, lmdb_path, data_root, annotation,
                 masked_language_model=True, transform=None):
        dataset = OCRDataset(
            lmdb_path=lmdb_path, root_dir=data_root, annotation_path=annotation,
            vocab=self.vocab, transform=transform,
            image_height=self.config["dataset"]["image_height"],
            image_min_width=self.config["dataset"]["image_min_width"],
            image_max_width=self.config["dataset"]["image_max_width"])

        return DataLoader(
            dataset,
            batch_sampler=_ClusterBatchSampler(dataset, self.batch_size, True),
            collate_fn=Collator(masked_language_model),
            **self.config["dataloader"])

    Trainer.data_gen = data_gen
    _PATCHED.append("Trainer.data_gen dùng batch_sampler (không vứt batch lẻ)")


# ---------------------------------------------------------------------------

def apply(verbose: bool = True) -> list:
    """Áp dụng toàn bộ patch. Gọi nhiều lần cũng an toàn (idempotent)."""
    if _PATCHED:
        if verbose:
            print(f"[compat] đã vá sẵn {len(_PATCHED)} mục")
        return _PATCHED

    if int(np.__version__.split(".")[0]) >= 2:
        _patch_fromstring()
        _patch_sctypes()
        _patch_aliases()
    _patch_lmdb_map_size()
    _patch_data_gen()

    if verbose:
        print(f"[compat] numpy {np.__version__} — đã vá {len(_PATCHED)} mục:")
        for p in _PATCHED:
            print(f"    - {p}")
    return _PATCHED


apply(verbose=False)


# ---------------------------------------------------------------------------
# C. Ghi chú: OneCycleLR chia cho 0 — KHÔNG vá, chỉ cần biết
# ---------------------------------------------------------------------------
# trainer.py:65
#     OneCycleLR(optimizer, total_steps=iters, max_lr=..., pct_start=0.1)
#
# Với `iters` quá nhỏ (< ~20), số bước warm-up = floor(iters * pct_start) - 1
# về 0 và torch ném:
#     ZeroDivisionError: division by zero   (lr_scheduler.py get_lr)
#
# Không phải bug cần vá — chỉ là `iters` phải đủ lớn. Dùng >= 50.
