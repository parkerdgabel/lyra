# LIST

| Function | Usage | Summary |
|---|---|---|
| `All` | `All[list, pred?]` | True if all match (optionally with pred) |
| `Any` | `Any[list, pred?]` | True if any matches (optionally with pred) |
| `ArgMax` | `ArgMax[f, list]` | 1-based index of maximal key. |
| `ArgMin` | `ArgMin[f, list]` | 1-based index of minimal key. |
| `CountBy` | `CountBy[f, list]` | Counts by key function (assoc) |
| `Drop` | `Drop[list, n]` | Drop first n (last if negative) |
| `DropWhile` | `DropWhile[pred, list]` | Drop while pred[x] holds |
| `Filter` | `Filter[pred, list]` | Keep elements where pred[x] is True |
| `Find` | `Find[pred, list]` | First element where pred[x] |
| `First` | `First[list]` | First element of a list (or Null). |
| `Init` | `Init[list]` | All but the last element. |
| `Join` | `Join[a|left, b|right, on?, how?]` | Join lists or datasets (dispatched). Overloads: Join[list1, list2]; Join[left, right, on, how?] |
| `Last` | `Last[list]` | Last element of a list (or Null). |
| `ListEdges` | `ListEdges[graph, opts?]` | List edges |
| `ListNodes` | `ListNodes[graph, opts?]` | List nodes |
| `MapAt` | `MapAt[f, subject, indexOrKey]` | Apply function at 1-based index or key. |
| `MapIndexed` | `MapIndexed[f, list]` | Map with index (1-based) |
| `MapThread` | `MapThread[f, lists]` | Map function over zipped lists (zip-with). |
| `MaxBy` | `MaxBy[f, list]` | Element with maximal derived key. |
| `MinBy` | `MinBy[f, list]` | Element with minimal derived key. |
| `Part` | `Part[subject, index]` | Index into list/assoc |
| `Partition` | `Partition[list, n, step?]` | Partition into fixed-size chunks |
| `Position` | `Position[pred, list]` | 1-based index of first match |
| `RandomChoice` | `RandomChoice[list]` | Random element from a list. |
| `Range` | `Range[a, b, step?]` | Create numeric range |
| `Reduce` | `Reduce[f, init?, list]` | Fold list with function |
| `Reject` | `Reject[pred, list]` | Drop elements where pred[x] is True |
| `ReplacePart` | `ReplacePart[subject, indexOrKey, value]` | Replace element at 1-based index or key. |
| `Rest` | `Rest[list]` | All but the first element. |
| `Sample` | `Sample[list, k]` | Sample k distinct elements from a list. |
| `Scan` | `Scan[f, init?, list]` | Prefix scan with function |
| `Shuffle` | `Shuffle[list]` | Shuffle list uniformly. |
| `Slice` | `Slice[list, start, len?]` | Slice list by start and length |
| `SubsetQ` | `SubsetQ[a, b]` | Is a subset of b? (sets, lists) |
| `Take` | `Take[list, n]` | Take first n (last if negative) |
| `TakeWhile` | `TakeWhile[pred, list]` | Take while pred[x] holds |
| `Tally` | `Tally[list]` | Counts by value (assoc) |
| `Total` | `Total[list]` | Sum elements in a list |
| `Union` | `Union[args]` | Union for lists (stable) or sets (dispatched) |
| `Unique` | `Unique[list]` | Stable deduplicate list |
| `UniqueBy` | `UniqueBy[f, list]` | Stable dedupe by derived key. |

## `ArgMax`

- Usage: `ArgMax[f, list]`
- Summary: 1-based index of maximal key.
- Tags: list, sort
- Examples:
  - `ArgMax[Identity, {2,10,5}]  ==> 2`

## `ArgMin`

- Usage: `ArgMin[f, list]`
- Summary: 1-based index of minimal key.
- Tags: list, sort
- Examples:
  - `ArgMin[Identity, {2,10,5}]  ==> 1`

## `Filter`

- Usage: `Filter[pred, list]`
- Summary: Keep elements where pred[x] is True
- Tags: generic, dataset, frame, list
- Examples:
  - `Filter[OddQ, {1,2,3,4}]  ==> {1,3}`

## `First`

- Usage: `First[list]`
- Summary: First element of a list (or Null).
- Tags: list
- Examples:
  - `First[{1,2,3}]  ==> 1`

## `Init`

- Usage: `Init[list]`
- Summary: All but the last element.
- Tags: list
- Examples:
  - `Init[{1,2,3}]  ==> {1,2}`

## `Join`

- Usage: `Join[a|left, b|right, on?, how?]`
- Summary: Join lists or datasets (dispatched). Overloads: Join[list1, list2]; Join[left, right, on, how?]
- Tags: generic, list, dataset
- Examples:
  - `Join[{1,2},{3}]  ==> {1,2,3}`

## `Last`

- Usage: `Last[list]`
- Summary: Last element of a list (or Null).
- Tags: list
- Examples:
  - `Last[{1,2,3}]  ==> 3`

## `MapAt`

- Usage: `MapAt[f, subject, indexOrKey]`
- Summary: Apply function at 1-based index or key.
- Tags: list, assoc, update
- Examples:
  - `MapAt[ToUpper, <|"a"->"x"|>, "a"]  ==> <|"a"->"X"|>`

## `MapIndexed`

- Usage: `MapIndexed[f, list]`
- Summary: Map with index (1-based)
- Tags: list, map
- Examples:
  - `MapIndexed[({#1, #2} &), {10,20}]  ==> {{10,1},{20,2}}`

## `MapThread`

- Usage: `MapThread[f, lists]`
- Summary: Map function over zipped lists (zip-with).
- Tags: list, map
- Examples:
  - `MapThread[Plus, {{1,2},{10,20}}]  ==> {11,22}`

## `MaxBy`

- Usage: `MaxBy[f, list]`
- Summary: Element with maximal derived key.
- Tags: list, sort
- Examples:
  - `MaxBy[Length, {"a","bbb","cc"}]  ==> "bbb"`

## `MinBy`

- Usage: `MinBy[f, list]`
- Summary: Element with minimal derived key.
- Tags: list, sort
- Examples:
  - `MinBy[Length, {"a","bbb","cc"}]  ==> "a"`

## `Partition`

- Usage: `Partition[list, n, step?]`
- Summary: Partition into fixed-size chunks
- Tags: list
- Examples:
  - `Partition[{1,2,3,4}, 2]  ==> {{1,2},{3,4}}`

## `RandomChoice`

- Usage: `RandomChoice[list]`
- Summary: Random element from a list.
- Tags: random, list
- Examples:
  - `SeedRandom[1]; RandomChoice[{"a","b","c"}]`

## `Range`

- Usage: `Range[a, b, step?]`
- Summary: Create numeric range
- Tags: list, math
- Examples:
  - `Range[1, 5]  ==> {1,2,3,4,5}`
  - `Range[0, 10, 2]  ==> {0,2,4,6,8,10}`

## `Reduce`

- Usage: `Reduce[f, init?, list]`
- Summary: Fold list with function
- Tags: list, fold
- Examples:
  - `Reduce[Plus, 0, {1,2,3}]  ==> 6`

## `ReplacePart`

- Usage: `ReplacePart[subject, indexOrKey, value]`
- Summary: Replace element at 1-based index or key.
- Tags: list, assoc, update
- Examples:
  - `ReplacePart[{1,2,3}, 2, 9]  ==> {1,9,3}`

## `Rest`

- Usage: `Rest[list]`
- Summary: All but the first element.
- Tags: list
- Examples:
  - `Rest[{1,2,3}]  ==> {2,3}`

## `Sample`

- Usage: `Sample[list, k]`
- Summary: Sample k distinct elements from a list.
- Tags: random, list
- Examples:
  - `SeedRandom[1]; Sample[{1,2,3,4}, 2]  ==> {3,1}`

## `Scan`

- Usage: `Scan[f, init?, list]`
- Summary: Prefix scan with function
- Tags: list, fold
- Examples:
  - `Scan[Plus, 0, {1,2,3}]  ==> {0,1,3,6}`

## `Shuffle`

- Usage: `Shuffle[list]`
- Summary: Shuffle list uniformly.
- Tags: random, list
- Examples:
  - `SeedRandom[1]; Shuffle[{1,2,3}]  ==> {3,1,2}`

## `Slice`

- Usage: `Slice[list, start, len?]`
- Summary: Slice list by start and length
- Tags: list, slice
- Examples:
  - `Slice[{10,20,30,40}, 2, 2]  ==> {20,30}`

## `SubsetQ`

- Usage: `SubsetQ[a, b]`
- Summary: Is a subset of b? (sets, lists)
- Tags: generic, set, list
- Examples:
  - `SubsetQ[HashSet[{1,2}], HashSet[{1,2,3}]]  ==> True`
  - `SubsetQ[{1,2}, {1,2,3}]  ==> True`

## `UniqueBy`

- Usage: `UniqueBy[f, list]`
- Summary: Stable dedupe by derived key.
- Tags: list, set
- Examples:
  - `UniqueBy[Length, {"a","bb","c","dd"}]  ==> {"a","bb"}`
