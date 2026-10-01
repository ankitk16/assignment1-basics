import os
import pickle
import time
from collections import Counter
from multiprocessing import Pool

import regex as re

from cs336_basics.pretokenization_example import find_chunk_boundaries

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def _count_chunk(args):
    """Worker: pre-tokenize one byte range of the file and count pre-token strings."""
    input_path, start, end, special_tokens = args
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")

        if special_tokens:
            segments = []
            split_pat = "|".join(re.escape(t) for t in special_tokens)
            for segment in re.split(split_pat, chunk):
                segments.append(segment)  # noqa: PERF402
        else:
            segments = [chunk]
    return Counter(m.group() for seg in segments for m in re.finditer(PAT, seg))


def pretokenize(
    input_path,
    special_tokens,
    num_processes=None,
    target_chunk_bytes=64 * 2**20,
    parallel_threshold=8
    * 2**20,  # min file size to make multiple chunks--> 8 mega bytes
):
    file_size = os.path.getsize(input_path)

    # small file or no token to split on safely: a single process is faster
    if file_size < parallel_threshold or not special_tokens:
        return _count_chunk((input_path, 0, file_size, special_tokens))

    num_processes = num_processes or os.cpu_count()
    n_chunks = max(num_processes, file_size // target_chunk_bytes)
    print(n_chunks)

    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(
            f, n_chunks, special_tokens[0].encode("utf-8")
        )

    tasks = [
        (input_path, s, e, special_tokens)
        for s, e in zip(boundaries[:-1], boundaries[1:])  # noqa: RUF007
    ]

    total = Counter()

    with Pool(num_processes) as pool:
        for c in pool.imap_unordered(_count_chunk, tasks):
            total.update(c)
    return total


def train_bpe(input_path: str, vocab_size, special_tokens: list[str]):
    """
    Input
     input_path: str Path to a text file with BPE tokenizer training data.
     vocab_size: int A positive integer that defines the maximum final vocabulary size (including the initial
     byte vocabulary, vocabulary items produced from merging, and any special tokens).
    Notes:
        1. For now, it only supports spliting by only the <endoftext> special token.
    """
    assert vocab_size >= 256, "Vocab size can not be smaller than 255!"
    # construct intial vocab: 0-255 usual
    vocab_map = {i: bytes([i]) for i in range(256)}

    # we are here means merging is needed
    numSpecial = len(special_tokens)

    # start counter for pre-tokenize time
    t = time.perf_counter()
    str_counts = pretokenize(input_path=input_path, special_tokens=special_tokens)
    print(f"pretokenize: {time.perf_counter() - t: .1f}s")

    # step 1: let's create bunch of metadata that will be needed
    # start time for the main loop
    t = time.perf_counter()
    pair_counts = {}  # pair -> total count across corpus
    pair_words = {}  # pair to list of words in which this pair occurs

    # str_counts' key are str; we want keys to be iteratable plus hashable so tuple (not list since they are not hashable)
    pretoken_counts = Counter()
    for pretoken_str, count in str_counts.items():
        b = pretoken_str.encode("utf-8")  # convert into bytes as per utf-8
        pretoken_counts[tuple(bytes([x]) for x in b)] += count

    # create initial pair counts
    for word, c in pretoken_counts.items():
        for p in zip(word, word[1:]):  # noqa: RUF007
            pair_counts[p] = pair_counts.get(p, 0) + c
            pair_words.setdefault(p, set()).add(word)

    # the main merging loop
    merged_pairs = []
    current_vocab = 255
    while current_vocab < vocab_size - numSpecial - 1:
        # break if run out of pairs
        if not pair_counts:
            break
        # step 3: find max-- break ties by lexicographically max
        best_pair = max(pair_counts, key=lambda k: (pair_counts[k], k))
        merged_pairs.append(best_pair)
        current_vocab += 1
        new_token = current_vocab
        # add new vocab entry by creating concated bytes for that entry
        merged = best_pair[0] + best_pair[1]
        vocab_map[new_token] = merged
        # list of all words with this pair
        words_with_pair = pair_words.get(best_pair)
        # create deep copy of words with pair so that we can delete while looping
        affected = [x for x in words_with_pair]

        for word in affected:
            for pair in zip(word, word[1:]):  # noqa: RUF007
                pair_counts[pair] -= pretoken_counts[word]
                # discard word from words that this pair's list has
                pair_words[pair].discard(word)
                if pair_counts[pair] <= 0:
                    del pair_counts[pair]

            # apply merge in the word and update the word
            updated_word = apply_merge(word, best_pair, merged)
            word_count = pretoken_counts[word]
            pretoken_counts[updated_word] = word_count
            del pretoken_counts[word]
            # update pair count to account for potential new pairs
            for pair in zip(updated_word, updated_word[1:]):  # noqa: RUF007
                pair_counts[pair] = pair_counts.get(pair, 0) + word_count
                if pair not in pair_words:
                    pair_words[pair] = set()
                pair_words[pair].add(updated_word)

    print(f"merges: {time.perf_counter() - t:.1f}s")
    # add special token to vocab
    for t in special_tokens:
        current_vocab += 1
        new_token = current_vocab
        vocab_map[new_token] = t.encode("utf-8")

    return vocab_map, merged_pairs


def apply_merge(word, pair, merged):
    """
    In the list of ints (ids), replace all consecutive occurences of pair with the new token idx.
    merged is bytes type.
    """
    newids = []
    i = 0
    while i < len(word):
        if i < len(word) - 1 and word[i] == pair[0] and word[i + 1] == pair[1]:
            newids.append(merged)
            i += 2
        else:
            newids.append(word[i])
            # print(word[i], new_index)
            i += 1

    # we need to return a tuple
    newids = tuple(elem for elem in newids)
    return newids


def bytesToTuple(b):
    return tuple(bytes([x]) for x in b)


# ____________tokenizer class______________#
class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.bytes_to_ids = {b: i for i, b in vocab.items()}
        self.vocab = vocab
        self.rank = {pair: i for i, pair in enumerate(merges)}
        self.special_tokens = special_tokens
        self.cache = {}
        self.merged_to_pair = {(pair[0] + pair[1]): pair for pair in merges}

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        d = pickle.load(open(vocab_filepath, "rb"))  # noqa: SIM115
        vocab = d["vocab"]
        merges = d["merges"]

        return cls(vocab, merges, special_tokens)

    def _bpe(self, word: tuple[bytes, ...]) -> tuple[bytes, ...]:
        # word = list(word)
        while len(word) > 1:
            best = min(
                zip(word, word[1:]),  # noqa: RUF007
                key=lambda p: self.rank.get(p, float("inf")),
            )
            if best not in self.rank:
                break  # no mergable pair left => only path to break the loop
            word = apply_merge(word, best, best[0] + best[1])
        return word

    def _tokToIds(self, tok: tuple[bytes, ...]):
        token = []
        for b in tok:
            token.append(self.bytes_to_ids[b])
        return token

    def encode(self, text: str):

        # step-1: let's first handle special tokens (which are list[str])
        if self.special_tokens:
            # sort by largest first since one spec token can be of combined many
            specials = sorted(self.special_tokens, key=len, reverse=True)
            special_pat = "(" + "|".join(re.escape(s) for s in specials) + ")"
            segments = re.split(special_pat, text)
            self.special_set = set(self.special_tokens)
        else:
            segments = [text]

        # above step has split over text by speical tokens but kept the special tokens.

        tokens = []
        for seg in segments:
            if not seg:
                # split yields "" at edges, e.g. text starting with a special token
                continue

            if self.special_tokens and seg in self.special_set:
                tokens.append(self.bytes_to_ids[seg.encode("utf-8")])
            else:
                for m in re.finditer(PAT, seg):
                    pretoken_str = m.group()
                    pretoken_unmerged = pretoken_str.encode("utf-8")
                    if pretoken_unmerged in self.cache:
                        pretoken_merged = self.cache[pretoken_unmerged]
                    else:
                        # our _bpe method expects tuple of bytes
                        word = tuple(bytes([x]) for x in pretoken_unmerged)
                        pretoken_merged = self._bpe(word)
                        self.cache[pretoken_unmerged] = pretoken_merged
                    tokens.extend(self._tokToIds(pretoken_merged))
        # print("returning")
        return tokens

    def encode_iterable(self, iterable):
        """
        Given an iterable of strings (e.g., a Python file handle), return a generator that lazily yields token IDs,
        one at a time.
        """
        for line in iterable:
            tokens = self.encode(line)
            yield from tokens

    def decode(self, ids: list[int]) -> str:
        """
        In decode, we do not need to unmerge since all the merge has done is concated bytes. But we
        would do the same at the end to run utf-8 decoder if we had unmerged (rather unconcated) bytes.
        """
        data = b"".join(self.vocab[i] for i in ids)
        return data.decode("utf-8", errors="replace")


# vocab_map, merged = train_bpe(
#    "data/TinyStoriesV2-GPT4-valid.txt", 300, ["<|endoftext|>"]
# )
# print(merged)

# for i in range(295, 300):
#    print(i, vocab_map[i])

if __name__ == "__main__":
    vocab, merges = train_bpe(
        "cs336_basics/data/TinyStoriesV2-GPT4-valid.txt", 1000, ["<|endoftext|>"]
    )
    print(merges[:10])
