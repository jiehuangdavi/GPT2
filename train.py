import operator
import os
import time

import jax
import jax.numpy as jnp
import flax.nnx as nnx
import matplotlib.pyplot as plt
import numpy as np
import optax
import orbax.checkpoint as ocp
import wandb
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from config import get_gpt2_config, get_paths
from GPT2_model import create_model, eval_loss_fn, restore_nnx_module, train_step


def main():
    gpt2_variant = "GPT2"
    tokenizer, model_config, training_config = get_gpt2_config(gpt2_variant)
    data_dir, checkpoint_path = get_paths()

    batch_size = training_config["batch_size"]
    max_steps = training_config["max_steps"]
    init_learning_rate = training_config["init_learning_rate"]
    weight_decay = training_config["weight_decay"]
    checkpoint_every = training_config["checkpoint_every"]

    seqlen = model_config.seqlen
    dtype = model_config.dtype
    param_dtype = model_config.param_dtype

    mesh = Mesh(mesh_utils.create_device_mesh((1, 1)), ("batch", "model"))

    use_wandb = os.environ.get("WANDB_API_KEY") is not None
    if use_wandb:
        wandb.login()
        wandb.init(
            project="GPT2-pretraining",
            config={
                "architecture": gpt2_variant,
                "dataset": "OpenWebText",
                "max_steps": max_steps,
                "batch_size": batch_size,
                "dtype": str(dtype),
                "param_dtype": str(param_dtype),
                "init_learning_rate": init_learning_rate,
                "num_transformer_blocks": model_config.num_transformer_blocks,
                "seqlen": seqlen,
                "embed_dim": model_config.embed_dim,
                "num_heads": model_config.num_heads,
                "feed_forward_dim": model_config.feed_forward_dim,
                "weight_decay": weight_decay,
            },
        )

    train_bin = os.path.join(data_dir, "train.bin")
    val_bin = os.path.join(data_dir, "val.bin")
    if not os.path.exists(train_bin) or not os.path.exists(val_bin):
        raise FileNotFoundError(
            f"Expected tokenized data at {train_bin} and {val_bin}. "
            "Place train.bin and val.bin in the data directory (set DATA_DIR env var to override)."
        )

    train_data = np.memmap(train_bin, dtype=np.uint16, mode="r")
    val_data = np.memmap(val_bin, dtype=np.uint16, mode="r")

    def get_batch(train_or_eval="train"):
        data = train_data if train_or_eval == "train" else val_data
        ix = np.random.randint(0, len(data) - seqlen, (batch_size,))
        x = np.stack([(data[i : i + seqlen]).astype(np.int64) for i in ix])
        y = np.stack([(data[i + 1 : i + 1 + seqlen]).astype(np.int64) for i in ix])
        return x, y

    print(f"Checkpoint path set to: {checkpoint_path}")
    print(f"Periodic checkpoint every {checkpoint_every} steps")

    def save_checkpoint(step_value):
        items_to_save = {
            "model_state": nnx.state(model),
            "optimizer_state": nnx.state(optimizer),
            "step": step_value,
        }
        print(f"Saving checkpoint to {checkpoint_path} at step {step_value}...")
        checkpointer.save(checkpoint_path, items_to_save, force=True)
        print(f"Checkpoint saved at step {step_value}")

    schedule = optax.cosine_decay_schedule(
        init_value=init_learning_rate,
        decay_steps=max_steps,
    )
    optax_chain = optax.chain(
        optax.adamw(learning_rate=schedule, weight_decay=weight_decay)
    )

    train_metrics = nnx.metrics.Average("loss")
    val_metrics = nnx.metrics.Average("val_loss")

    start_prompt = "Once upon a time"
    start_tokens = tokenizer.encode(start_prompt)[:seqlen]

    metrics_history = {"train_loss": [], "val_loss": []}
    step = 0
    start_time = time.time()

    resume_training = True
    checkpointer = ocp.PyTreeCheckpointer()
    latest_checkpoint = None

    checkpoint_marker = os.path.join(checkpoint_path, "_CHECKPOINT_METADATA")
    if resume_training and os.path.exists(checkpoint_marker):
        try:
            latest_checkpoint = checkpointer.restore(checkpoint_path)
        except Exception as exc:
            print(f"Could not restore checkpoint: {exc}")
            latest_checkpoint = None

    if resume_training and latest_checkpoint is not None:
        print(f"Resuming training from checkpoint: {checkpoint_path}")
        with mesh:
            model = create_model(model_config, rngs=nnx.Rngs(0), tokenizer=tokenizer)
            restore_nnx_module(model, latest_checkpoint["model_state"])
            optimizer = nnx.Optimizer(model, optax_chain, wrt=nnx.Param)
            restore_nnx_module(optimizer, latest_checkpoint["optimizer_state"])
    else:
        print("Starting new training session.")
        with mesh:
            model = create_model(model_config, rngs=nnx.Rngs(0), tokenizer=tokenizer)
            optimizer = nnx.Optimizer(model, optax_chain, wrt=nnx.Param)

        p_sizes = jax.tree.map(
            lambda p: p.size if isinstance(p, jnp.ndarray) else 0, nnx.state(model)
        )
        print(f"Number of model parameters: {jax.tree.reduce(operator.add, p_sizes)}")

        print("Initial generated text:")
        with mesh:
            model.generate_text(seqlen // 10, start_tokens)

    while True:
        input_batch, target_batch = get_batch("train")
        if len(input_batch) % len(jax.devices()) != 0:
            continue

        with mesh:
            train_step(
                model,
                optimizer,
                train_metrics,
                jax.device_put(
                    (input_batch, target_batch),
                    NamedSharding(mesh, P("batch", None)),
                ),
            )

        if step % 2000 == 0:
            train_loss = float(train_metrics.compute())
            metrics_history["train_loss"].append(train_loss)

            elapsed_time = time.time() - start_time
            print(
                f"Step {step + 1}, Training loss: {train_loss:.4f}, "
                f"Elapsed Time: {elapsed_time:.2f} seconds"
            )

            input_val_batch, target_val_batch = get_batch("val")
            with mesh:
                loss = eval_loss_fn(
                    model,
                    jax.device_put(
                        (input_val_batch, target_val_batch),
                        NamedSharding(mesh, P("batch", None)),
                    ),
                )
                loss.block_until_ready()
            val_metrics.update(val_loss=loss)
            val_loss = float(val_metrics.compute())
            metrics_history["val_loss"].append(val_loss)
            if use_wandb:
                wandb.log(data={"val_loss": val_loss, "train_loss": train_loss}, step=step)
            print(f"                                   Validation loss: {val_loss:.4f}")
            train_metrics.reset()
            val_metrics.reset()
            del loss, input_val_batch, target_val_batch
            start_time = time.time()

        # Skip step 0 so a fresh run does not overwrite a prior checkpoint
        # with an untrained model before any real progress.
        if step > 0 and step % checkpoint_every == 0:
            save_checkpoint(step)

        step += 1
        if step > max_steps:
            break

    print("Final generated text:")
    with mesh:
        model.generate_text(seqlen // 10, start_tokens)

    plt.plot(metrics_history["train_loss"])
    plt.title("Training Loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.savefig(os.path.join(checkpoint_path, "training_loss.png"))
    print(f"Saved training loss plot to {os.path.join(checkpoint_path, 'training_loss.png')}")

    save_checkpoint(step)

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()

