# Target Tensor API

These `.dd` files are intentional target code. They describe the Dudu tensor
library we want to make real.

`manifest.tsv` records each file's state. Graduated files have equivalent
named `dudu` targets that compile and run through `scripts/check_target_api.sh`.
A file graduates from this directory into a runnable target only when the
language, compiler diagnostics, native backend interop, and library
implementation are strong enough to compile and run it without hidden compiler
special cases.

The point is to keep the desired user experience concrete:

- write NumPy/PyTorch-like indexing in Dudu syntax
- keep tensor policy in Dudu library code
- keep BLAS/GPU backend calls behind explicit library APIs
- make view/copy/device movement visible in types and ordinary method calls
- make shape facts show up through normal Dudu generic metadata

Current graduated targets:

- `tensor_surface.dd` -> `dudu run target_tensor_surface`
- `advanced_indexing.dd` -> `dudu run target_advanced_indexing`
- `blas_backend.dd` -> `dudu run target_blas_backend`
- `gpu_backend.dd` -> `dudu run target_gpu_backend`

Current pending targets:

- `autograd_training.dd`: needs autograd modules, parameters, callable
  modules, backward, and optimizers
