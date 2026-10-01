import os
import pickle
import resource
import sys
import time

from cs336_basics.train_bpe import train_bpe


def peak_rss_mb(who):
    r = resource.getrusage(who).ru_maxrss
    return (
        r / 2**20 if sys.platform == "darwin" else r / 2**10
    )  # macOS: bytes, Linux: KB


if __name__ == "__main__":
    special = ["<|endoftext|>"]
    t0 = time.perf_counter()
    vocab, merges = train_bpe("data/owt_train.txt", 32_000, special)
    print(f"total: {(time.perf_counter() - t0) / 60:.1f} min")
    print(f"peak RSS, main process:   {peak_rss_mb(resource.RUSAGE_SELF):.0f} MB")
    print(f"peak RSS, largest worker: {peak_rss_mb(resource.RUSAGE_CHILDREN):.0f} MB")

    os.makedirs("artifacts", exist_ok=True)
    with open("artifacts/owt_bpe.pkl", "wb") as f:
        pickle.dump({"vocab": vocab, "merges": merges}, f)
    with open("artifacts/owt_merges.txt", "w", encoding="utf-8") as f:
        f.writelines(f"{a!r} {b!r}\n" for a, b in merges)
