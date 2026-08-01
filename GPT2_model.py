import jax
import jax.numpy as jnp
import flax.nnx as nnx
import numpy as np
import optax
from dataclasses import dataclass
from jax.sharding import PartitionSpec as P

# Debug: dump embedding tensors to disk (works under jit via jax.debug.callback).
# Set to 0 to disable. Each dump is large (~few MB text + npz).
_EMBED_DEBUG_MAX_DUMPS = 0
_EMBED_DEBUG_DUMP_COUNT = 0
_EMBED_DEBUG_TXT = "embedding_debug.txt"
_EMBED_DEBUG_NPZ = "embedding_debug.npz"


def _write_embedding_debug(token_ids, token_embedding, position_embedding, combined):
    """Host callback: write concrete arrays to text + npz for inspection."""
    global _EMBED_DEBUG_DUMP_COUNT
    if _EMBED_DEBUG_DUMP_COUNT >= _EMBED_DEBUG_MAX_DUMPS:
        return

    # bfloat16 often arrives as an unsupported/object dtype; force float32.
    def _as_f32(a):
        a = np.asarray(a)
        try:
            return a.astype(np.float32, copy=False)
        except (TypeError, ValueError):
            return np.vectorize(float, otypes=[np.float32])(a)

    token_ids = np.asarray(token_ids).astype(np.int32, copy=False)
    token_embedding = _as_f32(token_embedding)
    position_embedding = _as_f32(position_embedding)
    combined = _as_f32(combined)

    np.savez(
        _EMBED_DEBUG_NPZ,
        token_ids=token_ids,
        token_embedding=token_embedding,
        position_embedding=position_embedding,
        combined=combined,
    )

    with open(_EMBED_DEBUG_TXT, "w") as f:
        f.write(f"token_ids shape={token_ids.shape}\n")
        f.write(np.array2string(token_ids, threshold=np.inf, max_line_width=120))
        f.write("\n\n")

        for name, arr in (
            ("token_embedding", token_embedding),
            ("position_embedding", position_embedding),
            ("combined", combined),
        ):
            f.write(f"{name} shape={arr.shape} dtype={arr.dtype}\n")
            # Write each sequence position as one line of embed_dim floats.
            flat = arr.reshape(-1, arr.shape[-1])
            for i, row in enumerate(flat):
                f.write(f"[{i}] ")
                np.savetxt(f, row[None, :], fmt="%.6g")
            f.write("\n")

    _EMBED_DEBUG_DUMP_COUNT += 1
    print(
        f"Wrote embedding debug dump #{_EMBED_DEBUG_DUMP_COUNT} to "
        f"{_EMBED_DEBUG_TXT} and {_EMBED_DEBUG_NPZ}",
        flush=True,
    )


@dataclass
class GPT2Config:
    seqlen: int = 512
    vocab_size: int = 50257
    embed_dim: int = 768
    num_heads: int = 12
    feed_forward_dim: int = 3072
    num_transformer_blocks: int = 12
    dropout_rate: float = 0.1
    top_k: int = 50
    top_p: float = 0.9
    temperature: float = 0.9
    repetition_penalty: float = 1.2
    dtype: jnp.dtype = jnp.bfloat16
    param_dtype: jnp.dtype = jnp.float32


def causal_attention_mask(seq_len):
    return jnp.tril(jnp.ones((seq_len, seq_len)))


class TransformerBlock(nnx.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ff_dim: int,
        dropout_rate: float,
        dtype: jnp.dtype,
        param_dtype: jnp.dtype,
        rngs: nnx.Rngs,
    ):
        self.layer_norm1 = nnx.LayerNorm(
            epsilon=1e-6,
            num_features=embed_dim,
            scale_init=nnx.with_partitioning(nnx.initializers.ones_init(), P('model')),
            bias_init=nnx.with_partitioning(nnx.initializers.zeros_init(), P('model')),
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.mha = nnx.MultiHeadAttention(
            num_heads=num_heads,
            in_features=embed_dim,
            kernel_init=nnx.with_partitioning(nnx.initializers.xavier_uniform(), P(None, 'model')),
            bias_init=nnx.with_partitioning(nnx.initializers.zeros_init(), P('model')),
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.dropout1 = nnx.Dropout(rate=dropout_rate, rngs=rngs)
        self.layer_norm2 = nnx.LayerNorm(
            epsilon=1e-6,
            num_features=embed_dim,
            scale_init=nnx.with_partitioning(nnx.initializers.ones_init(), P('model')),
            bias_init=nnx.with_partitioning(nnx.initializers.zeros_init(), P('model')),
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.linear1 = nnx.Linear(
            in_features=embed_dim,
            out_features=ff_dim,
            kernel_init=nnx.with_partitioning(nnx.initializers.xavier_uniform(), P(None, 'model')),
            bias_init=nnx.with_partitioning(nnx.initializers.zeros_init(), P('model')),
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.linear2 = nnx.Linear(
            in_features=ff_dim,
            out_features=embed_dim,
            kernel_init=nnx.with_partitioning(nnx.initializers.xavier_uniform(), P(None, 'model')),
            bias_init=nnx.with_partitioning(nnx.initializers.zeros_init(), P('model')),
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.dropout2 = nnx.Dropout(rate=dropout_rate, rngs=rngs)

    def __call__(self, inputs, training: bool = False):
        _, seq_len, _ = inputs.shape

        attention_output = self.mha(
            inputs_q=self.layer_norm1(inputs),
            mask=causal_attention_mask(seq_len),
            decode=False,
        )
        x = inputs + self.dropout1(attention_output, deterministic=not training)

        mlp_output = self.linear1(self.layer_norm2(x))
        mlp_output = nnx.gelu(mlp_output)
        mlp_output = self.linear2(mlp_output)
        mlp_output = self.dropout2(mlp_output, deterministic=not training)

        return x + mlp_output


class TokenAndPositionEmbedding(nnx.Module):
    def __init__(
        self,
        seqlen: int,
        vocab_size: int,
        embed_dim: int,
        dtype: jnp.dtype,
        param_dtype: jnp.dtype,
        rngs: nnx.Rngs,
    ):
        self.token_emb = nnx.Embed(
            num_embeddings=vocab_size,
            features=embed_dim,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.pos_emb = nnx.Embed(
            num_embeddings=seqlen,
            features=embed_dim,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

# The token and position embeddings have two learnable weight matrices, both from nnx.Embed:
# 
# | Parameter	         | Shape (defaults)	 | Role
# | token_emb.embedding  | (vocab_size, embed_dim) → (50257, 768) | one vector per vocab token
# | pos_emb.embedding    | (seqlen, embed_dim) → (512, 768)     | one vector per position
# Stored in param_dtype (default float32); compute may cast to bfloat16.
# The transformer table uses sinusoidal positional encoding, which is added to the token embeddings.
# Here, we use look up table for positional encoding. The model can learn whatever position patterns help next-token prediction,
# instead of being locked to a hand-design sinusoid basis. No frequency formulus, easy to train.
    def __call__(self, x):
        positions = jnp.arange(0, x.shape[1])[None, :]
        position_embedding = self.pos_emb(positions)
        token_embedding = self.token_emb(x)
        combined = token_embedding + position_embedding

        # # Dump full arrays to file (host callback; safe under jit).
        # jax.debug.callback(
        #     _write_embedding_debug,
        #     x,
        #     token_embedding,
        #     position_embedding,
        #     combined,
        # )

        return self.token_emb, combined


class GPT2(nnx.Module):
    def __init__(self, config: GPT2Config, rngs: nnx.Rngs, tokenizer=None):
        self.config = config
        self.tokenizer = tokenizer

        self.embedding_layer = TokenAndPositionEmbedding(
            config.seqlen,
            config.vocab_size,
            config.embed_dim,
            config.dtype,
            config.param_dtype,
            rngs=rngs,
        )
        self.dropout = nnx.Dropout(rate=config.dropout_rate, rngs=rngs)

        self.transformer_blocks = nnx.List([
            TransformerBlock(
                config.embed_dim,
                config.num_heads,
                config.feed_forward_dim,
                config.dropout_rate,
                config.dtype,
                config.param_dtype,
                rngs=rngs,
            )
            for _ in range(config.num_transformer_blocks)
        ])

        self.layer_norm = nnx.LayerNorm(
            epsilon=1e-6,
            num_features=config.embed_dim,
            scale_init=nnx.with_partitioning(nnx.initializers.ones_init(), P('model')),
            bias_init=nnx.with_partitioning(nnx.initializers.zeros_init(), P('model')),
            dtype=config.dtype,
            param_dtype=config.param_dtype,
            rngs=rngs,
        )

    def __call__(self, inputs, training: bool = False):
        token_embedding, x = self.embedding_layer(inputs)
        x = self.dropout(x, deterministic=not training)
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, training=training)
        x = self.layer_norm(x)
        return token_embedding.attend(x)

    def _apply_repetition_penalty(self, logits, token_ids, sample_index):
        """Down-weight logits for tokens already present up to sample_index (jit-safe)."""
        penalty = self.config.repetition_penalty
        positions = jnp.arange(token_ids.shape[0])
        valid = positions <= sample_index

        # Mark vocab entries that appear in the valid prefix.
        def body(i, presence):
            return jax.lax.cond(
                valid[i],
                lambda p: p.at[token_ids[i]].set(True),
                lambda p: p,
                presence,
            )

        presence = jax.lax.fori_loop(
            0, token_ids.shape[0], body, jnp.zeros_like(logits, dtype=bool)
        )
        # HF-style: divide positive scores, multiply negative scores.
        penalized = jnp.where(logits > 0, logits / penalty, logits * penalty)
        return jnp.where(presence, penalized, logits)

    @nnx.jit
    def sample_from(self, logits, rng):
        """Sample one token with temperature, top-k, and top-p. `rng` must be fresh each call."""
        temperature = jnp.maximum(self.config.temperature, 1e-8)
        logits = logits.astype(jnp.float32) / temperature

        top_k = self.config.top_k
        top_logits, top_indices = jax.lax.top_k(logits, k=top_k)

        # Nucleus (top-p) filtering within the top-k set.
        order = jnp.argsort(-top_logits)
        sorted_logits = top_logits[order]
        sorted_indices = top_indices[order]
        sorted_probs = jax.nn.softmax(sorted_logits)
        cum_probs = jnp.cumsum(sorted_probs)
        # Remove tokens that push cumulative probability past top_p; always keep the first.
        remove = (cum_probs - sorted_probs) > self.config.top_p
        remove = remove.at[0].set(False)
        filtered_logits = jnp.where(remove, -jnp.inf, sorted_logits)

        probs = jax.nn.softmax(filtered_logits)
        return jax.random.choice(rng, sorted_indices, p=probs)

    @nnx.jit
    def generate_step(self, padded_tokens, sample_index, rng):
        logits = self(padded_tokens)[0, sample_index]
        if self.config.repetition_penalty != 1.0:
            logits = self._apply_repetition_penalty(
                logits, padded_tokens[0], sample_index
            )
        return self.sample_from(logits, rng)

    def generate_text(self, max_tokens, start_tokens, rng_seed: int = 0):
        if self.tokenizer is None:
            raise ValueError("tokenizer must be set on the model for text generation")

        generated = []
        rng = jax.random.PRNGKey(rng_seed)
        print(self.tokenizer.decode(start_tokens), flush=True, end='')
        for _ in range(max_tokens):
            sample_index = len(start_tokens) + len(generated) - 1
            padded_tokens = jnp.array(
                start_tokens + generated + [0] * (self.config.seqlen - len(start_tokens) - len(generated))
            )[None, :]
            rng, step_rng = jax.random.split(rng)
            next_token = int(self.generate_step(padded_tokens, sample_index, step_rng))
            if next_token == self.tokenizer.encode(
                '<|endoftext|>', allowed_special={'<|endoftext|>'}
            )[0]:
                break
            generated.append(next_token)
            print(self.tokenizer.decode([next_token]), flush=True, end='')
        return self.tokenizer.decode(start_tokens + generated)


def create_model(config: GPT2Config, rngs: nnx.Rngs, tokenizer=None):
    return GPT2(config, rngs=rngs, tokenizer=tokenizer)


def _unwrap_orbax_value_leaves(tree):
    """Orbax restores NNX Params as {'value': array}; unwrap for replace_by_pure_dict."""
    if isinstance(tree, dict):
        if set(tree.keys()) == {"value"}:
            return tree["value"]
        return {k: _unwrap_orbax_value_leaves(v) for k, v in tree.items()}
    return tree


def restore_nnx_module(module, restored_state_dict):
    """Load Orbax-restored nested dict weights into an NNX module in-place."""
    graphdef, state = nnx.split(module)
    nnx.replace_by_pure_dict(state, _unwrap_orbax_value_leaves(restored_state_dict))
    nnx.update(module, state)
    return module


@nnx.jit
def loss_fn(model, batch):
    """Training loss (dropout on). Returns (loss, logits) for value_and_grad has_aux."""
    logits = model(batch[0], training=True)
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=batch[1]
    ).mean()
    return loss, logits


@nnx.jit
def eval_loss_fn(model, batch):
    """Eval loss only — no dropout, no logits kept (saves HBM)."""
    logits = model(batch[0], training=False)
    return optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=batch[1]
    ).mean()


@nnx.jit
def train_step(model: nnx.Module, optimizer: nnx.Optimizer, metrics: nnx.MultiMetric, batch):
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, _logits), grads = grad_fn(model, batch)
    metrics.update(loss=loss)
    optimizer.update(model, grads)
