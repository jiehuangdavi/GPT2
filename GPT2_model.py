import jax
import jax.numpy as jnp
import flax.nnx as nnx
import optax
from dataclasses import dataclass
from jax.sharding import PartitionSpec as P


@dataclass
class GPT2Config:
    seqlen: int = 512
    vocab_size: int = 50257
    embed_dim: int = 768
    num_heads: int = 12
    feed_forward_dim: int = 3072
    num_transformer_blocks: int = 12
    dropout_rate: float = 0.1
    top_k: int = 10
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
        self.dropout1 = nnx.Dropout(rate=dropout_rate)
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
        self.dropout2 = nnx.Dropout(rate=dropout_rate)

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

    def __call__(self, x):
        positions = jnp.arange(0, x.shape[1])[None, :]
        position_embedding = self.pos_emb(positions)
        token_embedding = self.token_emb(x)
        return self.token_emb, token_embedding + position_embedding


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
        self.dropout = nnx.Dropout(rate=config.dropout_rate)

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

    @nnx.jit
    def sample_from(self, logits):
        logits, indices = jax.lax.top_k(logits, k=self.config.top_k)
        logits = nnx.softmax(logits)
        return jax.random.choice(jax.random.PRNGKey(0), indices, p=logits)

    @nnx.jit
    def generate_step(self, padded_tokens, sample_index):
        logits = self(padded_tokens)
        return self.sample_from(logits[0][sample_index])

    def generate_text(self, max_tokens, start_tokens):
        if self.tokenizer is None:
            raise ValueError("tokenizer must be set on the model for text generation")

        generated = []
        print(self.tokenizer.decode(start_tokens), flush=True, end='')
        for _ in range(max_tokens):
            sample_index = len(start_tokens) + len(generated) - 1
            padded_tokens = jnp.array(
                start_tokens + generated + [0] * (self.config.seqlen - len(start_tokens) - len(generated))
            )[None, :]
            next_token = int(self.generate_step(padded_tokens, sample_index))
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
    logits = model(batch[0])
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=batch[1]
    ).mean()
    return loss, logits


@nnx.jit
def train_step(model: nnx.Module, optimizer: nnx.Optimizer, metrics: nnx.MultiMetric, batch):
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(model, batch)
    metrics.update(loss=loss, logits=logits)
    optimizer.update(model, grads)
