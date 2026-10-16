"""OpenPI's discrete-state prompt from token pieces, verified against the tokenizer.

OpenPI (models/tokenizer.py) tokenizes `Task: <task>, State: b0 b1 ...;\\nAction: ` with BOS every
step. The runtime cannot run SentencePiece, so the bundle carries each task's head (BOS through
"State:"), one piece per digit, the separator "▁" (or "▁-" for bin -1) and the tail.
"""
import numpy as np

DIGITS = "0123456789"
TOKENS_PER_STATE = 4  # separator + up to three digits


def clean(task):
    return task.strip().replace("_", " ").replace("\n", " ")


def bins(state_norm):
    return np.digitize(state_norm, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1


class StatePrompt:
    def __init__(self, model_path):
        import sentencepiece
        self.sp = sentencepiece.SentencePieceProcessor(model_file=str(model_path))
        self.sep, self.neg = self.sp.piece_to_id("▁"), self.sp.piece_to_id("▁-")
        self.pieces = {c: self.sp.piece_to_id(c) for c in DIGITS}
        unknown = self.sp.unk_id()
        assert unknown not in (self.sep, self.neg, *self.pieces.values()), "state pieces are not single tokens"
        head = self.head("x")
        full = self.full("x", [5])
        assert full[:len(head) + 2] == head + [self.sep, self.pieces["5"]], "head is not a prefix of the prompt"
        self.tail = full[len(head) + 2:]

    def full(self, task, values):
        text = f"Task: {clean(task)}, State: {' '.join(map(str, values))};\nAction: "
        return self.sp.encode(text, add_bos=True)

    def head(self, task):
        return self.sp.encode(f"Task: {clean(task)}, State:", add_bos=True)

    def compose(self, head, values):
        ids = list(head)
        for v in values:
            v = int(v)
            ids += [self.neg if v < 0 else self.sep] + [self.pieces[c] for c in str(abs(v))]
        return ids + self.tail

    def verify(self, task, state_dim, seed=0):
        head, rng, n = self.head(task), np.random.default_rng(seed), 0
        for v in range(-1, 256):
            assert self.compose(head, [v] * state_dim) == self.full(task, [v] * state_dim), (task, v)
            n += 1
        for _ in range(500):
            values = rng.integers(-1, 256, state_dim)
            assert self.compose(head, values) == self.full(task, values), (task, values)
            n += 1
        return n

    def capacity(self, tasks, state_dim):
        return max(len(self.head(t)) for t in tasks) + TOKENS_PER_STATE * state_dim + len(self.tail)
