import time

import numpy as np
from cs336_basics.train_bpe import Tokenizer

tok = Tokenizer.from_files(
    "artifacts/owt_bpe.pkl", "artifacts/owt_bpe.pkl", ["<|endoftext|>"]
)


# suppose we tokenize whole dataset in one go (we dont want to do that though)
def tokenize_to_npy(tok, in_path: str, out_path, dtype=np.uint16):
    assert len(tok.vocab) <= np.iinfo(dtype).max + 1, "vocab too large for dtype"
    t0 = time.perf_counter()
    ids = []
    with open(in_path, encoding="utf-8") as f:
        for i, tid in enumerate(tok.encode_iterable(f), 1):  # 1 from where to start i
            ids.append(tid)
            if i % 10_000_000 == 0:
                print(f"{i / 1e6:.0f}M tokens, {time.perf_counter() - t0:.0f}s")
    arr = np.array(ids, dtype=dtype)
    np.save(out_path, arr)
    print(
        f"{out_path}: {arr.shape[0]:,} tokens, {arr.nbytes / 2**20:.0f} MB, {time.perf_counter() - t0:.0f}s"
    )


# let's tokenize validation first
tokenize_to_npy(tok, "data/owt_train.txt", "artifacts/owt_train.npy")
tokenize_to_npy(tok, "data/owt_valid.txt", "artifacts/owt_valid.npy")
