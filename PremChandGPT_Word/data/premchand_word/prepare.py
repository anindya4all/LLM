"""
Word-level tokenizer for Premchand's Hindi corpus.

Unlike char-level tokenization (vocab ~65) or GPT-2 BPE (trained on English),
this builds a vocabulary directly from the Premchand corpus at the word level.
Result: the model learns Hindi word patterns and generates coherent Hindi text.

Usage:
    python data/premchand_word/prepare.py

Outputs:
    train.bin, val.bin  - token IDs as uint16 numpy arrays
    meta.pkl            - vocab, stoi/itos dicts, tokenizer_type='word'
"""
import os
import re
import pickle
from collections import Counter
import numpy as np

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
MIN_FREQ = 2          # words appearing fewer times become <UNK>
MAX_VOCAB = 30000     # cap vocabulary size (rare words → <UNK>)
TRAIN_SPLIT = 0.9

# Special tokens (kept at fixed indices for easy reference)
PAD_TOKEN  = '<PAD>'   # 0
UNK_TOKEN  = '<UNK>'   # 1
BOS_TOKEN  = '<BOS>'   # 2  (begin of sentence / book boundary)
EOS_TOKEN  = '<EOS>'   # 3  (end of sentence / book boundary)

SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]

# --------------------------------------------------------------------------
# Tokenization
# --------------------------------------------------------------------------
# Devanagari punctuation that should be split off as their own tokens:
#   ।  (danda - sentence end)   ॥  (double danda - section end)
#   common ASCII punctuation inside Hindi prose
_PUNCT_RE = re.compile(r'([।॥\.,!?;:"\'\(\)\[\]\-–—/\\<>])')

def tokenize(text: str) -> list[str]:
    """Split Hindi text into word-level tokens.

    Strategy:
    - Pad punctuation with spaces so they become distinct tokens.
    - Split on whitespace.
    - Skip empty strings and pure-whitespace tokens.
    """
    text = _PUNCT_RE.sub(r' \1 ', text)
    tokens = [t for t in text.split() if t.strip()]
    return tokens

# --------------------------------------------------------------------------
# Load corpus
# --------------------------------------------------------------------------
corpus_path = os.path.join(os.path.dirname(__file__), '..', 'premchandcorpus', 'premchand_corpus.txt')
corpus_path = os.path.abspath(corpus_path)

# Fallback: look for input.txt next to this file (symlink / copy)
local_input = os.path.join(os.path.dirname(__file__), 'input.txt')
if not os.path.exists(corpus_path) and os.path.exists(local_input):
    corpus_path = local_input

if not os.path.exists(corpus_path):
    raise FileNotFoundError(
        f"Corpus not found at {corpus_path}.\n"
        "Copy premchand_corpus.txt to data/premchand_word/ as input.txt, or "
        "ensure data/premchandcorpus/premchand_corpus.txt exists."
    )

print(f"Reading corpus from: {corpus_path}")
with open(corpus_path, 'r', encoding='utf-8') as f:
    raw_text = f.read()

print(f"Corpus size: {len(raw_text):,} characters")

# --------------------------------------------------------------------------
# Build vocabulary
# --------------------------------------------------------------------------
all_tokens = tokenize(raw_text)
print(f"Total word tokens: {len(all_tokens):,}")

freq = Counter(all_tokens)
print(f"Unique tokens before filtering: {len(freq):,}")

# Keep only tokens above MIN_FREQ, sorted by frequency (most common first)
vocab_tokens = [tok for tok, cnt in freq.most_common() if cnt >= MIN_FREQ]
vocab_tokens = vocab_tokens[:MAX_VOCAB - len(SPECIAL_TOKENS)]  # leave room for specials

# Final vocab: specials first, then corpus words
vocab = SPECIAL_TOKENS + vocab_tokens
vocab_size = len(vocab)
print(f"Final vocabulary size: {vocab_size:,}  (min_freq={MIN_FREQ}, max={MAX_VOCAB})")

stoi = {w: i for i, w in enumerate(vocab)}
itos = {i: w for i, w in enumerate(vocab)}

unk_id = stoi[UNK_TOKEN]
coverage = sum(freq[t] for t in vocab_tokens) / len(all_tokens) * 100
print(f"Vocab covers {coverage:.1f}% of corpus tokens  (rest → <UNK>)")

# --------------------------------------------------------------------------
# Encode
# --------------------------------------------------------------------------
def encode(tokens: list[str]) -> list[int]:
    return [stoi.get(t, unk_id) for t in tokens]

ids = encode(all_tokens)

n = len(ids)
train_ids = ids[:int(n * TRAIN_SPLIT)]
val_ids   = ids[int(n * TRAIN_SPLIT):]

print(f"Train tokens: {len(train_ids):,}")
print(f"Val   tokens: {len(val_ids):,}")

# vocab_size fits in uint16 (max 65535) given MAX_VOCAB ≤ 30000
dtype = np.uint16 if vocab_size <= 65535 else np.uint32

out_dir = os.path.dirname(__file__)
np.array(train_ids, dtype=dtype).tofile(os.path.join(out_dir, 'train.bin'))
np.array(val_ids,   dtype=dtype).tofile(os.path.join(out_dir, 'val.bin'))
print("Saved train.bin and val.bin")

# --------------------------------------------------------------------------
# Save meta
# --------------------------------------------------------------------------
meta = {
    'vocab_size':      vocab_size,
    'stoi':            stoi,
    'itos':            itos,
    'tokenizer_type':  'word',          # sample.py uses this to pick encode/decode
    'unk_token':       UNK_TOKEN,
    'bos_token':       BOS_TOKEN,
    'eos_token':       EOS_TOKEN,
    'min_freq':        MIN_FREQ,
    'corpus_tokens':   len(all_tokens),
}
with open(os.path.join(out_dir, 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)
print("Saved meta.pkl")
print(f"\nDone. vocab_size={vocab_size}  (set this in your training config)")
