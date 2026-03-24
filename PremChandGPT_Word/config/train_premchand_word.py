# Train a word-level GPT on Premchand's Hindi corpus.
#
# After running data/premchand_word/prepare.py, check its output for
# "Final vocabulary size: XXXX" and set vocab_size below accordingly.
#
# Training on CPU / MacBook:
#   python train.py config/train_premchand_word.py --device=cpu --compile=False
#
# Training on GPU:
#   python train.py config/train_premchand_word.py

out_dir = 'out-premchand-word'
eval_interval = 250
eval_iters = 200
log_interval = 10

always_save_checkpoint = False  # only save when val loss improves

wandb_log = False
wandb_project = 'premchand-word'
wandb_run_name = 'word-gpt'

dataset = 'premchand_word'

# ----------------------------------------------------------------------------
# Model — medium-small GPT tuned for a ~30k-word Hindi vocabulary.
#
# Word-level models need:
#   - Larger n_embd than char models (richer per-token representation)
#   - Shorter block_size (128 words ≈ 1-2 paragraphs, plenty of context)
#   - Moderate depth (corpus is ~200k tokens, don't over-parameterize)
# ----------------------------------------------------------------------------
block_size = 128        # context window in *words*
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2

# vocab_size=7853 is read automatically from data/premchand_word/meta.pkl
# at runtime in train.py — you do NOT need to set it here.

# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------
batch_size = 32                 # sequences per step (reduce if OOM)
gradient_accumulation_steps = 1

learning_rate = 1e-3
max_iters = 5000
lr_decay_iters = 5000
min_lr = 1e-4

beta2 = 0.99   # slightly higher because batches are small
warmup_iters = 200

# Increase max_iters and lower learning_rate for better quality:
#   max_iters = 10000
#   learning_rate = 5e-4
