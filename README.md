# dudu-datascience

Small runnable Dudu demos for array, slice, and tensor-style indexing.

This is not a production numeric library yet. It is a dogfood repo for the
indexing forms and reusable tensor surface that are useful for data science and
ML-shaped code.

The `spec/target_api` directory contains target Dudu code for the real tensor
library we want. `spec/target_api/manifest.tsv` records which targets have
graduated into runnable `dudu` targets and which are still pending.

## Run

```bash
dudu run
```

For compiler stage timings:

```bash
dudu run --timings
```

To verify the target API status:

```bash
./scripts/check_target_api.sh
```

Graduated target API files have named `dudu` targets and must run. Pending
files stay listed with an explicit missing implementation reason.

The current reusable tensor module links both OpenBLAS and OpenCL because the
GPU backend lives in the same `dudu_tensor` module as the CPU tensor surface.
That keeps the dogfood API direct while Dudu grows cleaner extension-module
boundaries.

## Demos

`src/array_demos.dd` shows fixed-array indexing:

- scalar comma indexing: `matrix[1, 2]`
- indexed assignment: `matrix[2, 3] = ...`
- generic shape/stride views through `array_view[T]`
- row views: `matrix[1, :]`
- column views: `matrix[:, 1]`
- full matrix views: `matrix[:, :]`
- rectangular patches: `matrix[1:3, 2:4]`
- one-dimensional stepped slices: `values[0:6:2]`, `values[1::2]`, `values[::3]`
- open ranges: `matrix[1:, 1:]`
- rank-3 image/channel indexing: `pixels[y, x, channel]`
- channel/range views: `pixels[0, 0, 0:3]`, `pixels[:, :, 1]`, `pixels[:, :, :]`

`src/hook_demos.dd` shows library-defined indexing:

- scalar hooks: `tensor[i]`
- pairwise gather/scatter through normal direct indexing: `tensor[rows, cols]`
- orthogonal/cartesian gather through an explicit helper object:
  `tensor.cartesian[rows, cols]`
- no compiler-owned tensor indexing operators are required

`src/tensor_demos.dd` shows a tiny tensor-like type:

- `Tensor` with scalar `[]` and `[]=`
- row-mask selection: `tensor[mask, :]`
- row-mask scatter assignment: `tensor[mask, :] = value`

`src/blas_demos.dd` shows native backend interop:

- `from c import cblas.h as blas`
- pure Dudu row-major matmul compared against `blas.cblas_sgemm`
- explicit CPU-contiguous storage boundary through `&tensor.data[0]`

`src/target_gpu_backend.dd` graduates the OpenCL target API:

- `host_tensor.to(opencl.default())` uploads CPU tensor storage
- `gpu_tensor.matmul(other_gpu_tensor)` runs an OpenCL kernel
- `gpu_tensor[0, :]` uses the same index-plan path as CPU views and stays on
  device
- `gpu_tensor.cpu()` explicitly downloads back into a Dudu tensor

`src/backend_surface_demo.dd` shows the target backend boundary shape:

- backend marker imports with `from dudu_tensor.backends import cpu`
  and `from dudu_tensor.backends import openblas`
- `tensor.to(cpu.default())`, `tensor.to(openblas.default())`, and `tensor.cpu()`
- explicit `as_array_view()` and `to_array()` materialization boundaries

`src/blas_backend_demo.dd` graduates the smaller BLAS target contract into a
runnable check:

- target-style `Tensor[f32][rows, cols]` shape assertion
- overloaded `assert_close` for tensors, views, and scalars
- `logits.cpu()`, `as_array_view()`, and `to_array()` boundary checks

`src/advanced_indexing_demo.dd` graduates the first advanced-indexing target
slice into reusable `dudu_tensor` code:

- `index_array([...])` and `bool_mask([...])`
- pairwise gather/scatter: `logits[rows, cols]`
- cartesian gather: `logits.cartesian[rows, cols]`
- runtime-shaped mask selection: `logits[mask, :]`
- mask scatter assignment: `logits[mask, :] = selected`
- column-slice scalar update: `selected[:, 0] = selected[:, 0] + 10.0`
- chained window tiles: `logits.window[2, 2][1, 2]`

`src/shape_stride_demo.dd` checks the corrected tensor storage model:

- `Tensor` and `TensorView` carry `shape`, `strides`, and `offset`
- rank-3 view: `image[:, :, 1]`
- rank-4 view: `hyper[:, 1, :, 0]`
- old row/column/patch-specific view helpers are replaced by one
  rank-independent index-plan path

`src/activation_metrics_demo.dd` graduates the first activation/loss target
slice:

- deterministic initializer: `randn[f32](rows, cols, scale)`
- elementwise helpers: `relu(tensor)`, `sigmoid(tensor)`
- scalar loss tensor: `mse_loss(predicted, expected)`
- metric method: `predicted.binary_accuracy(expected)`

The BLAS demo needs `openblas` discoverable through `pkg-config`.

Each section prints the intended result and the actual computed result. The
program exits nonzero if the final summary score drifts.

## Layout

```text
src/main.dd          entry point
src/activation_metrics_demo.dd activation, loss, metric helpers
src/array_demos.dd   fixed arrays and slices
src/advanced_indexing_demo.dd target-style index arrays, masks, cartesian helper
src/backend_surface_demo.dd backend marker and materialization boundaries
src/blas_backend_demo.dd target-style BLAS backend boundary
src/blas_demos.dd    OpenBLAS-backed matrix multiply
src/hook_demos.dd    user-defined indexing operators
src/shape_stride_demo.dd rank-3/rank-4 shape-stride view checks
src/tensor_demos.dd  small tensor-style mask demo
src/dudu_tensor.dd   reusable Tensor/TensorView/indexer slice
src/print_utils.dd   tiny print helpers
spec/target_api/     target API examples plus graduated/pending manifest
```
