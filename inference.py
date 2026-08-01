import os
import sys

import flax.nnx as nnx
import orbax.checkpoint as ocp
from jax.experimental import mesh_utils
from jax.sharding import Mesh

from config import get_gpt2_config, get_paths
from GPT2_model import create_model, restore_nnx_module


def main():
    tokenizer, model_config, _ = get_gpt2_config()
    _, checkpoint_path = get_paths()

    mesh = Mesh(mesh_utils.create_device_mesh((1, 1)), ("batch", "model"))

    checkpoint_marker = os.path.join(checkpoint_path, "_CHECKPOINT_METADATA")
    if not os.path.exists(checkpoint_marker):
        print(f"No checkpoint found at {checkpoint_path}. Train the model first with train.py.")
        sys.exit(1)

    checkpointer = ocp.PyTreeCheckpointer()
    restored_checkpoint = checkpointer.restore(checkpoint_path)

    with mesh:
        model = create_model(model_config, rngs=nnx.Rngs(0), tokenizer=tokenizer)
        restore_nnx_module(model, restored_checkpoint["model_state"])

    start_prompt = "Donald Trump is a "
    if len(sys.argv) > 1:
        start_prompt = " ".join(sys.argv[1:])

    start_tokens = tokenizer.encode(start_prompt)[: model_config.seqlen]
    print(f"Generating from prompt: {start_prompt!r}\n")
    with mesh:
        generated_text = model.generate_text(model_config.seqlen // 10, start_tokens)

    print(f"\nRestored model generated text:\n{generated_text}")


if __name__ == "__main__":
    main()
