# Target Tensor API

These `.dd` files are intentional target code. They describe the Dudu tensor
library we want to make real.

They are not wired into `dudu build` yet. A file graduates from this directory
into `src/` only when the language, compiler diagnostics, native backend
interop, and library implementation are strong enough to compile and run it
without hidden compiler special cases.

The point is to keep the desired user experience concrete:

- write NumPy/PyTorch-like indexing in Dudu syntax
- keep tensor policy in Dudu library code
- keep BLAS/GPU backend calls behind explicit library APIs
- make view/copy/device movement visible in types and ordinary method calls
- make shape facts show up through normal Dudu generic metadata

