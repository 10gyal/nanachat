"""Naive tokenizer using regex (refers karpathy/minbpe)"""

import regex as re

# split rules
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""


def get_stats(ids, counts=None):
    """
    Given a list of integers, return a dictionary of counts of consecutive pairs
    e.g., [1, 2, 3, 1, 2] -> {(1, 2): 2, (2, 3): 1, (3, 1): 1}

    counts (dict, optional): existing dictionary of counts
    """
    counts = {} if counts is None else counts
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge(ids, pair, idx):
    """
    In the given list of ids, replace all occurances of the pair with the new ID idx
    e.g., ids=[1, 2, 3, 1, 2], pair=(1, 2), idx=4 -> [4, 3, 4]
    """
    new_ids = []
    i = 0
    while i < len(ids):
        if ids[i] == pair[0] and i < len(ids) - 1 and ids[i + 1] == pair[1]:
            new_ids.append(idx)
            i += 2
        else:
            new_ids.append(ids[i])
            i += 1
    return new_ids


class RegexTokenizer:
    def __init__(self, pattern=None):
        self.pattern = GPT4_SPLIT_PATTERN if pattern is None else pattern
        self.merges = {}
        self.compiled_pattern = re.compile(self.pattern)
        self.special_tokens = {}
        self.inversae_special_tokens = {}
        self.vocab = self._build_vocab()

    def _build_vocab(self):
        # vocab is simply and deterministically derived from merges
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        for special, idx in self.special_tokens.items():
            vocab[idx] = special.encode("utf-8")
        return vocab

    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        ambiguous = False

        # split the text with the spilt rule
        text_chunks = re.findall(self.compiled_pattern, text)

        # input text preprocessing
        ids = [list(ch.encode("utf-8")) for ch in text_chunks]

        # iteratively merge the most common pairs to create new tokens
        merges = {}
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for i in range(num_merges):
            # count the number of times every consecutive pair appears
            stats = {}
            for chunk_ids in ids:
                get_stats(chunk_ids, stats)
            # find the pair with the highest frequency count
            pair = max(stats, key=stats.get)

            # check if the merge is ambiguous - i.e., the max value is not unique
            pair_count = stats[pair]
            pairs_with_max_count = [
                pair for pair, count in stats.items() if count == pair_count
            ]
            if len(pairs_with_max_count) > 1:
                ambiguous = True

            # create new token by assigning it the next available id
            idx = 256 + i

            # replace all ocurrances of the pair in ids with idx
            ids = [merge(chunk_ids, pair, idx) for chunk_ids in ids]

            # save the merge
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]

            # prints
            if verbose:
                print(
                    f"merge {i+1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {stats[pair]} occurrences"
                )

        # save class vars
        self.merges = merges
        self.vocab = vocab
        return ambiguous

    def _encode_chunk(self, text_bytes):
        # return the token ids
        ids = list(text_bytes)
        while len(ids) >= 2:
            # find the pair with the lowest merge index
            stats = get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))

            if pair not in self.merges:
                break  # nothing to merge further

            # otherwise merge the best pair i.e.,e the lowest merge index
            idx = self.merges[pair]
            ids = merge(ids, pair, idx)
        return ids

    def encode_ordinary(self, text):
        "Encoding that ignores any special tokens."
        text_chunks = re.findall(self.compiled_pattern, text)

        # all chunks are encoded separately and then the results are joined
        ids = []
        for chunk in text_chunks:
            chunk_bytes = chunk.encode("utf-8")
            chunk_ids = self._encode_chunk(chunk_bytes)
            ids.extend(chunk_ids)
        return ids
