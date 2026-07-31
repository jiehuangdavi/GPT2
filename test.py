import jax
import numpy as np
import numba

print(f"JAX version: {jax.__version__}")
print(f"NumPy version: {np.__version__}")
print(f"Numba version: {numba.__version__}")

# Verify TPU connection
devices = jax.devices()
print(f"Available devices: {devices}")
assert "tpu" in str(devices[0]).lower(), "TPU is not detected! Make sure your Colab runtime type is set to TPU."
