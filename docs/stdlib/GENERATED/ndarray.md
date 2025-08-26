# NDARRAY

| Function | Usage | Summary |
|---|---|---|
| `NDAdd` | `NDAdd[a, b]` | Elementwise addition with broadcast |
| `NDArgMax` | `NDArgMax[a, axis?]` | Argmax index per axis or flattened |
| `NDArray` | `NDArray[spec]` | Create NDArray from list/shape/data (held) |
| `NDAsType` | `NDAsType[a, type]` | Cast array to a new element type |
| `NDClip` | `NDClip[a, min, max]` | Clip array values to [min,max] |
| `NDConcat` | `NDConcat[arrays, axis?]` | Concatenate arrays along axis |
| `NDCos` | `NDCos[a]` | Elementwise cos |
| `NDDiv` | `NDDiv[a, b]` | Elementwise division with broadcast |
| `NDEltwise` | `NDEltwise[f, a, b?]` | Apply custom elementwise op (held) |
| `NDExp` | `NDExp[a]` | Elementwise exp |
| `NDLog` | `NDLog[a]` | Elementwise natural log |
| `NDMap` | `NDMap[f, a]` | Map function elementwise (held) |
| `NDMatMul` | `NDMatMul[a, b]` | Matrix multiply A x B |
| `NDMean` | `NDMean[a, axes?]` | Mean over all elements or axes |
| `NDMul` | `NDMul[a, b]` | Elementwise multiplication with broadcast |
| `NDPermuteDims` | `NDPermuteDims[a, perm]` | Permute dimensions by order |
| `NDPow` | `NDPow[a, b]` | Elementwise exponentiation with broadcast |
| `NDReduce` | `NDReduce[f, a, axes?]` | Reduce over axes with function (held) |
| `NDRelu` | `NDRelu[a]` | ReLU activation (max(0, x)) |
| `NDReshape` | `NDReshape[a, shape]` | Reshape array to new shape |
| `NDShape` | `NDShape[a]` | Shape of an NDArray |
| `NDSin` | `NDSin[a]` | Elementwise sin |
| `NDSlice` | `NDSlice[a, slices]` | Slice array by ranges per axis |
| `NDSqrt` | `NDSqrt[a]` | Elementwise sqrt |
| `NDSub` | `NDSub[a, b]` | Elementwise subtraction with broadcast |
| `NDSum` | `NDSum[a, axes?]` | Sum over all elements or axes |
| `NDTanh` | `NDTanh[a]` | Elementwise tanh |
| `NDTranspose` | `NDTranspose[a, perm?]` | Transpose array axes |
| `NDType` | `NDType[a]` | Element type (f64/i64/...) of array |

## `NDArray`

- Usage: `NDArray[spec]`
- Summary: Create NDArray from list/shape/data (held)
- Examples:
  - `a := NDArray[<|"shape"->{2,2}, "data"->{1,2,3,4}|>]`

## `NDMatMul`

- Usage: `NDMatMul[a, b]`
- Summary: Matrix multiply A x B
- Examples:
  - `NDMatMul[NDArray[{{1,2},{3,4}}], NDArray[{{5,6},{7,8}}]]  ==> NDArray[...] `

## `NDShape`

- Usage: `NDShape[a]`
- Summary: Shape of an NDArray
- Examples:
  - `NDShape[a]  ==> {2,2}`
