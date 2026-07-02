# Implementation Gaps

The target files in this directory are the spec. The implementation should make
those files compile and run without compiler-owned tensor policy.

## 1. Reusable Tensor Library

The runnable demos use `src/ndad.dd` for early dogfood. The current
module now stores `Tensor[T]` and `TensorView[T]` with rank-generic `shape`,
`strides`, and `offset` metadata. Rank-2 helpers now read dimensions through
`dim(axis)` instead of storing separate `rows` and `cols` fields; 2D
constructors remain convenience APIs for existing demos and BLAS interop, not
the core indexing representation.

The current module provides:

- `Tensor[T]` owning CPU storage with runtime `shape`, `strides`, `offset`,
  and rank-independent `dim(axis)` / `count()` helpers
- `TensorView[T]` borrowing CPU storage with runtime `shape`, `strides`, and
  `offset`, plus the same dimension/count helpers
- one rank-independent CPU view planner for scalar/slice/ellipsis/new-axis
  index items, shared by `Tensor` and `TensorView`
- scalar element access through `*idx: scalar_index`, so scalar-indexed axes
  are dropped instead of preserving old rank-2 row/column shapes
- `zeros`, `ones`, `full`, `arange`, `randn`
- `from_list`, `from_nested`
- `assert_close`, `print_tensor`
- `index_array`, `bool_mask`
- direct pairwise advanced indexing, explicit `cartesian`, and `window`
  library indexer fields
- `relu`, `sigmoid`, `mse_loss`, `binary_accuracy`

The remaining target API still needs:

- `Tensor[T][shape]` owning CPU storage
- `TensorView[T][shape]` borrowing storage
- broader assignment and advanced-index combinations through the same index
  planner; `tensor[:, col] = values` is still a narrow Dudu helper and should
  eventually become a generic tensor assignment path like the in-repo `ndad`
  target surface
- broader reductions and metrics needed by examples
- cleaner extension-module boundaries so importing the CPU tensor surface does
  not force every target to link the OpenCL helper

Construction and testing helpers should prefer module-level functions such as
`zeros[f32](...)`, `from_list[f32](...)`, and `assert_close(actual, expected)`.
Class static factories are allowed only when they add real value; the target
API should not mix styles casually.

## 2. View And Copy Boundary

Normal slices should return view-like values when storage/layout allow it:

- `x[1, :]`
- `x[:, 2]`
- `x[1:, 1:3]`
- `image[:, :, 0]`

Explicit materialization APIs are required:

- `view.to_tensor()`
- `tensor.to_array()`
- `tensor.as_array_view()`
- `tensor.to(device)`
- `tensor.cpu()`

`as_array_view()` is zero-copy only for CPU-contiguous storage. GPU tensors,
strided views, lazy expressions, and autograd values need explicit conversion
or a clear diagnostic.

## 3. Shape Metadata And Inference

Target code assumes these are usable and visible in hover/inlay:

- `Tensor[f32][2, 2]`
- `Tensor[f32][dyn, 5]`
- constructor-inferred shape from `zeros[f32](rows, cols)`
- matmul shape propagation
- mask selection producing `dyn`
- `assume_shape[...]` or equivalent explicit shape assertion

The runnable `src/ndad.dd` slice now provides method-form
`tensor.assume_shape[Rows, Cols]()` for explicit API-boundary assertions. This
currently carries compile-time metadata while preserving normal runtime
`shape`/`strides` metadata; broader shape propagation and diagnostics remain
compiler and library work.

Shape metadata now matches runtime storage metadata for the runnable CPU tensor
slice. Remaining work is to generalize more convenience constructors and
printing/reduction helpers beyond 2D and make richer shape diagnostics visible
in compiler/LSP output.

The compiler should provide general shape metadata and diagnostics. Tensor
layout, allocation, and backend behavior stay in library code.

## 4. Advanced Indexer Objects

These are library fields/objects, not compiler forms:

- `tensor[rows, cols]`
- `tensor.cartesian[rows, cols]`
- `tensor.window[height, width][row, col]`
- `sparse.coo[rows, cols]`

Required semantics:

- normal `@operator("[]")` and `@operator("[]=")` dispatch
- chained temporary indexing
- scatter assignment
- compound assignment only when the selected-value write hook is explicit
- repeated-index scatter order is library policy

Runnable status: `src/advanced_indexing_demo.dd` now covers reusable
`index_array`, `bool_mask`, `tensor[rows, cols]` pairwise gather/scatter,
`tensor.cartesian[rows, cols]` cartesian gather, and
`tensor[mask, :]` selection/scatter. It also covers reusable window tiles with
both `tiles = tensor.window[height, width]; tiles[row, col]` and direct
chained `tensor.window[height, width][row, col]` indexing. Sparse COO indexers
and richer window policies remain target work.

`src/shape_stride_demo.dd` additionally proves rank-3 and rank-4 tensor views
over the same shape/stride/offset metadata: `image[:, :, 1]` and
`hyper[:, 1, :, 0]`.

## 5. Broadcasting And Elementwise Ops

Target code uses:

- `tensor + bias`
- `selected[:, 0] = selected[:, 0] + 10.0`
- activation helpers such as `relu` and `sigmoid`

Broadcasting must be implemented as library overloads/lazy expressions, not as
compiler tensor rules.

Runnable status: `Tensor + Tensor` supports a small row/column broadcasting
slice. `TensorView + scalar` and column-slice scatter are now covered by the
target-style `selected[:, 0] = selected[:, 0] + 10.0` expression in
`src/advanced_indexing_demo.dd`. `src/activation_metrics_demo.dd` covers
`relu`, `sigmoid`, `mse_loss`, and `binary_accuracy`. Lazy expressions and
broader elementwise composition remain target work.

## 6. Native Backend Modules

The target API needs backend modules:

- `ndad.backends.cpu`
- `ndad.backends.openblas`
- `ndad.backends.opencl`

OpenBLAS should call CBLAS through normal native imports. OpenCL should own
device buffers, kernels, upload/download, and cleanup through normal Dudu/C++
RAII boundaries. CUDA/cuBLAS is not required on this AMD machine.

The runnable CPU/OpenBLAS slice now has backend marker singletons in
`ndad.backends` and exercises `from ndad.backends import cpu`,
`from ndad.backends import openblas`, `tensor.to(cpu.default())`,
`tensor.to(openblas.default())`, and `tensor.cpu()`. CPU/OpenBLAS movement
preserves CPU storage.

`src/blas_backend_demo.dd` graduates the smallest BLAS/backend target surface
into a runnable check: target-style backend imports, explicit
`Tensor[f32][rows, cols]` shape assertion, overloaded `assert_close` for
tensor/view/scalar values, and `cpu()`/`as_array_view()`/`to_array()`
boundaries.

`src/target_gpu_backend.dd` keeps the smallest OpenCL target surface as a
runnable optional check on the older `dudu_tensor` prototype: explicit
CPU-to-device transfer, device matmul, device indexing through the shared index
planner without implicit CPU copies, explicit download, and comparison against
the CPU tensor result. The current OpenCL dogfood backend materializes
rank-1/rank-2 gathered views into new contiguous device buffers; that is a
backend limitation, not a compiler tensor rule. Moving OpenCL under
`ndad.backends.opencl` is still extension-module work.

User code should move values with PyTorch-like device calls such as
`tensor.to(opencl.default())` and `tensor.cpu()`. Backend selection is library
policy; the compiler should only preserve enough type facts for diagnostics.

## 7. Autograd Prototype

`autograd_training.dd` has graduated into `src/target_autograd_training.dd`.
The runnable `mald` layer now proves that Dudu can express a package-shaped ML
surface on top of `ndad` without compiler/library-name special cases:

- `Parameter[T]` owns value and gradient tensors
- `Loss.backward()` writes computed gradients back to parameters
- `SGD.step()` and `SGD.zero_grad()` update parameters through pointers
- callable model objects use `@operator("()")`
- the target trains a tiny OR classifier through normal Dudu code

This is intentionally a minimal autograd proof. Full dynamic graph capture,
view/scatter backward rules, activation-aware backward helpers, and richer
module hierarchies remain library work. The compiler should only need normal
classes, enums, references, generics, operator overloads, and diagnostics.

## 8. Editor And Diagnostics

For the target code to feel first-class, LSP should show:

- inferred tensor/view/backend types
- shape metadata, including `dyn`
- selected `@operator("[]")` / `@operator("[]=")` signatures
- copy-vs-view/backend movement in docs
- good errors for missing hooks, incompatible shape facts, and illegal
  CPU-contiguous conversions

## Graduation Rule

A target file graduates into `src/` when it:

1. compiles with `dudu build`
2. runs and prints/verifies expected values
3. does not require compiler special cases for tensor/library names
4. has any required compiler behavior covered by fixtures in the Dudu repo
