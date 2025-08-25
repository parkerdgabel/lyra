# NDArray Standard Library

Lyra provides an NDArray toolkit built on top of `PackedArray` values. These functions offer shape manipulation, slicing, reductions, broadcasting, and elementwise math.

## Construction & Introspection

- `NDArray[data]`: construct from nested lists
- `NDShape[a]`: returns shape as a list of integers
- `NDType[a]`, `NDAsType[a, dtype]`: dtype is `Float64` currently

## Shape Ops

- `NDReshape[a, {dims...}]` (one `-1` allowed for inference)
- `NDTranspose[a]` or `NDTranspose[a, {perm...}]`
- `NDPermuteDims[a, {perm...}]` (alias for explicit permutation)
- `NDConcat[{a,b,...}, axis]`

## Slicing

- `NDSlice[a, axis, start, len]` (axis 0‑based; start 1‑based)
- `NDSlice[a, {start, len}]` (1D arrays)
- `NDSlice[a, {spec1, spec2, ...}]` (per‑axis specs):
  - `All` keeps full axis
  - integer index (1‑based) selects a single coordinate
  - `{start, len}` takes a range slice

## Reductions & Map

- `NDMap[a, f]` (pure function or symbol)
- `NDReduce[a, f]` overall; `NDReduce[a, f, axis]` along axis
- Convenience: `NDSum[a, axis?]`, `NDMean[a, axis?]`, `NDArgMax[a, axis?]`

## Elementwise & Broadcasting

- Binary ops with broadcasting: `NDAdd`, `NDSub`, `NDMul`, `NDDiv`, `NDPow`
- Generic elementwise: `NDEltwise[f, a, b]`
- Unary ops: `NDRelu`, `NDClip[a, min, max]`, `NDExp`, `NDLog`, `NDSqrt`, `NDSin`, `NDCos`, `NDTanh`

## Examples

```
// 2x3 array
a = NDArray[{{1,2,3},{4,5,6}}]
NDShape[a]              // => {2, 3}
NDTranspose[a]          // => {3, 2}
NDSlice[a, 1, 2, 2]     // => last 2 columns → {{2,3},{5,6}}
NDSlice[a, {All, {2,2}}]// => same via multi‑axis spec

NDSum[a]                // => 21.
NDMean[a, 1]            // => {2., 5.}
NDArgMax[a, 1]          // => {3., 3.}

row = NDArray[{10,20,30}]
NDAdd[a, row]           // => broadcasting over rows

b = NDArray[{{1,2},{3,4},{5,6}}]
NDMatMul[a, b]          // => {{22., 28.}, {49., 64.}}

NDClip[NDArray[{-1,0,2,5}], 0, 3] // => {0., 0., 2., 3.}
NDRelu[NDArray[{-1,0,2,5}]]        // => {0., 0., 2., 5.}
```
