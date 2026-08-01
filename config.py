import os
from pathlib import Path

import jax.numpy as jnp
import tiktoken

from GPT2_model import GPT2Config

# Paths on the GCP VM (override with DATA_DIR / CHECKPOINT_DIR env vars if needed)
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = os.environ.get("DATA_DIR", str(PROJECT_ROOT / "data"))
CHECKPOINT_DIR = os.environ.get("CHECKPOINT_DIR", str(PROJECT_ROOT / "checkpoints"))


def get_paths():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    return DATA_DIR, CHECKPOINT_DIR


def get_gpt2_config(variant="GPT2"):
    tokenizer = tiktoken.get_encoding("gpt2")
    vocab_size = tokenizer.n_vocab

    if variant == "GPT2-medium":
        num_transformer_blocks = 24
        seqlen = 1024
        embed_dim = 1024
        num_heads = 16
        batch_size = 32
    else:
        num_transformer_blocks = 12
        seqlen = 512
        embed_dim = 768
        num_heads = 12
        # 48+ often OOMs on ~16GB HBM (e.g. TPU v5e). Override with BATCH_SIZE if needed.
        batch_size = int(os.environ.get("BATCH_SIZE", "32"))

    feed_forward_dim = 4 * embed_dim
    # total tokens/ (batch_size * seqlen)
    max_steps = 550000

    config = GPT2Config(
        seqlen=seqlen,
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
        feed_forward_dim=feed_forward_dim,
        num_transformer_blocks=num_transformer_blocks,
        dropout_rate=0.1,
        top_k=50,
        top_p=0.9,
        temperature=0.9,
        repetition_penalty=1.2,
        dtype=jnp.bfloat16,
        param_dtype=jnp.float32,
    )

    training = {
        "batch_size": batch_size,
        "max_steps": max_steps,
        "init_learning_rate": 5e-4,
        "weight_decay": 1e-1,
        "checkpoint_every": int(os.environ.get("CHECKPOINT_EVERY", "10000")),
    }
    return tokenizer, config, training
