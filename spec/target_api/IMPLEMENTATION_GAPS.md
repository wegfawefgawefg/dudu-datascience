# Implementation Gaps

The target files in this directory are the spec. The implementation should make
those files compile and run without compiler-owned tensor policy.

## 1. Reusable Tensor Library

The runnable demos now use a real `src/dudu_tensor.dd` module for the first
slice of the target API. It currently provides:

- `Tensor[T]` owning CPU storage with runtime `rows`/`cols`
- `TensorView[T]` borrowing CPU storage for normal row/column/patch slices
- `zeros`, `ones`, `full`, `arange`
- `from_list`, `from_nested`
- `assert_close`, `print_tensor`
- `index_array`, `bool_mask`
- `vindex` and `oindex` library indexer fields

The remaining target API still needs:

- `Tensor[T][shape]` owning CPU storage
- `TensorView[T][shape]` borrowing storage
- `randn`
- reductions and metrics needed by examples

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

The runnable `src/dudu_tensor.dd` slice now provides method-form
`tensor.assume_shape[Rows, Cols]()` for explicit API-boundary assertions. This
currently carries compile-time metadata while preserving normal runtime
`rows`/`cols`; broader shape propagation and diagnostics remain compiler and
library work.

The compiler should provide general shape metadata and diagnostics. Tensor
layout, allocation, and backend behavior stay in library code.

## 4. Advanced Indexer Objects

These are library fields/objects, not compiler forms:

- `tensor.vindex[rows, cols]`
- `tensor.oindex[rows, cols]`
- `tensor.window[height, width][row, col]`
- `sparse.coo[rows, cols]`

Required semantics:

- normal `@operator("[]")` and `@operator("[]=")` dispatch
- chained temporary indexing
- scatter assignment
- compound assignment only when the selected-value write hook is explicit
- repeated-index scatter order is library policy

Runnable status: `src/advanced_indexing_demo.dd` now covers reusable
`index_array`, `bool_mask`, `tensor.vindex[rows, cols]` pairwise
gather/scatter, `tensor.oindex[rows, cols]` cartesian gather, and
`tensor[mask, :]` selection/scatter. Window/tiling indexers and chained window
temporaries remain target work.

## 5. Broadcasting And Elementwise Ops

Target code uses:

- `tensor + bias`
- `selected[:, 0] = selected[:, 0] + 10.0`
- activation helpers such as `relu` and `sigmoid`

Broadcasting must be implemented as library overloads/lazy expressions, not as
compiler tensor rules.

Runnable status: `Tensor + Tensor` supports a small row/column broadcasting
slice. `TensorView + scalar`, lazy expressions, and broader elementwise
composition remain target work.

## 6. Native Backend Modules

The target API needs backend modules:

- `dudu_tensor.backends.cpu`
- `dudu_tensor.backends.openblas`
- `dudu_tensor.backends.opencl`

OpenBLAS should call CBLAS through normal native imports. OpenCL should own
device buffers, kernels, upload/download, and cleanup through normal Dudu/C++
RAII boundaries. CUDA/cuBLAS is not required on this AMD machine.

The runnable slice now has backend marker singletons in `dudu_tensor.backends`
and exercises `from dudu_tensor.backends import cpu`,
`from dudu_tensor.backends import openblas`, `tensor.to(cpu.default())`,
`tensor.to(openblas.default())`, and `tensor.cpu()`. Backend movement currently
preserves CPU storage; real device ownership and dispatch are library work.

`src/blas_backend_demo.dd` graduates the smallest BLAS/backend target surface
into a runnable check: target-style backend imports, explicit
`Tensor[f32][rows, cols]` shape assertion, overloaded `assert_close` for
tensor/view/scalar values, and `cpu()`/`as_array_view()`/`to_array()`
boundaries. It still uses the CPU-backed storage slice; true backend dispatch
remains library work.

User code should move values with PyTorch-like device calls such as
`tensor.to(opencl.default())` and `tensor.cpu()`. Backend selection is library
policy; the compiler should only preserve enough type facts for diagnostics.

## 7. Autograd Prototype

`autograd_training.dd` requires:

- `Parameter[T][shape]`
- `Module` or equivalent parameter-owning base/trait
- modules callable as `model(x)` while still allowing explicit `forward`
- parameters usable directly in tensor operations
- internally tracked tensor operations
- `loss.backward()`
- SGD optimizer
- simple loss/activation helpers

This should feel PyTorch-like at the user level. Tape/graph bookkeeping may
exist inside the library, but target user code should not instantiate a public
`Tape` just to train a model. The compiler should only need normal classes,
enums, references, generics, operator overloads, and diagnostics.

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
