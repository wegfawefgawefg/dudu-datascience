# dudu-datascience

Small runnable Dudu demos for array, slice, and tensor-style indexing.

This is not a numeric library. It is a tour of the indexing forms that are
useful for data science and ML-shaped code.

## Run

```bash
dudu run
```

For compiler stage timings:

```bash
dudu run --timings
```

## Demos

`src/array_demos.dd` shows fixed-array indexing:

- scalar comma indexing: `matrix[1, 2]`
- indexed assignment: `matrix[2, 3] = ...`
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
- pairwise gather/scatter through an indexer object: `tensor.vindex[rows, cols]`
- orthogonal/cartesian gather/scatter through an indexer object: `tensor.oindex[rows, cols]`
- no compiler-owned `vindex[]` or `oindex[]` operators are required

`src/tensor_demos.dd` shows a tiny tensor-like type:

- `Tensor` with scalar `[]` and `[]=`
- row-mask selection: `tensor[mask, :]`
- row-mask scatter assignment: `tensor[mask, :] = value`

Each section prints the intended result and the actual computed result. The
program exits nonzero if the final summary score drifts.

## Layout

```text
src/main.dd          entry point
src/array_demos.dd   fixed arrays and slices
src/hook_demos.dd    user-defined indexing operators
src/tensor_demos.dd  small tensor-style mask demo
src/print_utils.dd   tiny print helpers
```
