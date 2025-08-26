# LIST

| Function | Usage | Summary |
|---|---|---|
| `All` | `All[list, pred?]` | True if all match (optionally with pred) |
| `Any` | `Any[list, pred?]` | True if any matches (optionally with pred) |
| `CountBy` | `CountBy[f, list]` | Counts by key function (assoc) |
| `Drop` | `Drop[list, n]` | Drop first n (last if negative) |
| `DropWhile` | `DropWhile[pred, list]` | Drop while pred[x] holds |
| `Filter` | `Filter[pred, list]` | Keep elements where pred[x] is True |
| `Find` | `Find[pred, list]` | First element where pred[x] |
| `Flatten` | `Flatten[list, levels?]` | Flatten by levels (default 1) |
| `GroupBy` | `GroupBy[ds, keys]` | Group rows by key(s) |
| `Join` | `Join[left, right, on, how?]` | Join two datasets on keys |
| `MapIndexed` | `MapIndexed[f, list]` | Map with index (1-based) |
| `Part` | `Part[subject, index]` | Index into list/assoc |
| `Partition` | `Partition[list, n, step?]` | Partition into fixed-size chunks |
| `Position` | `Position[pred, list]` | 1-based index of first match |
| `Range` | `Range[a, b, step?]` | Create numeric range |
| `Reduce` | `Reduce[f, init?, list]` | Fold list with function |
| `Reject` | `Reject[pred, list]` | Drop elements where pred[x] is True |
| `Scan` | `Scan[f, init?, list]` | Prefix scan with function |
| `Slice` | `Slice[list, start, len?]` | Slice list by start and length |
| `Sort` | `Sort[ds, by, opts?]` | Sort rows by columns |
| `Take` | `Take[list, n]` | Take first n (last if negative) |
| `TakeWhile` | `TakeWhile[pred, list]` | Take while pred[x] holds |
| `Tally` | `Tally[list]` | Counts by value (assoc) |
| `Total` | `Total[list]` | Sum elements in a list |
| `Transpose` | `Transpose[rows]` | Transpose list of lists |
| `Unique` | `Unique[list]` | Stable deduplicate list |
| `Unzip` | `Unzip[pairs]` | Unzip pairs into two lists |
| `Zip` | `Zip[a, b]` | Zip two lists into pairs |

## `Filter`

- Usage: `Filter[pred, list]`
- Summary: Keep elements where pred[x] is True
- Examples:
  - `Filter[OddQ, {1,2,3,4}]  ==> {1,3}`

## `Flatten`

- Usage: `Flatten[list, levels?]`
- Summary: Flatten by levels (default 1)
- Examples:
  - `Flatten[{{1},{2,3}}]  ==> {1,2,3}`

## `GroupBy`

- Usage: `GroupBy[ds, keys]`
- Summary: Group rows by key(s)
- Examples:
  - `GroupBy[ds, dept]  ==> grouped`

## `Join`

- Usage: `Join[left, right, on, how?]`
- Summary: Join two datasets on keys
- Examples:
  - `Join[{1,2},{3}]  ==> {1,2,3}`

## `MapIndexed`

- Usage: `MapIndexed[f, list]`
- Summary: Map with index (1-based)
- Examples:
  - `MapIndexed[({#1, #2} &), {10,20}]  ==> {{10,1},{20,2}}`

## `Partition`

- Usage: `Partition[list, n, step?]`
- Summary: Partition into fixed-size chunks
- Examples:
  - `Partition[{1,2,3,4}, 2]  ==> {{1,2},{3,4}}`

## `Range`

- Usage: `Range[a, b, step?]`
- Summary: Create numeric range
- Examples:
  - `Range[1, 5]  ==> {1,2,3,4,5}`
  - `Range[0, 10, 2]  ==> {0,2,4,6,8,10}`

## `Reduce`

- Usage: `Reduce[f, init?, list]`
- Summary: Fold list with function
- Examples:
  - `Reduce[Plus, 0, {1,2,3}]  ==> 6`

## `Scan`

- Usage: `Scan[f, init?, list]`
- Summary: Prefix scan with function
- Examples:
  - `Scan[Plus, 0, {1,2,3}]  ==> {0,1,3,6}`

## `Slice`

- Usage: `Slice[list, start, len?]`
- Summary: Slice list by start and length
- Examples:
  - `Slice[{10,20,30,40}, 2, 2]  ==> {20,30}`
