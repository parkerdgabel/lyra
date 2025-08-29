# DATASET

| Function | Usage | Summary |
|---|---|---|
| `Agg` | `Agg[ds, aggs]` | Aggregate groups to single rows |
| `Aggregate` | `Aggregate[group, spec]` | Aggregate grouped data (stub) |
| `Cast` | `Cast[value, type]` | Cast a value to a target type (string, integer, real, boolean). |
| `Coalesce` | `Coalesce[values…]` | First non-null value |
| `Concat` | `Concat[inputs]` | Concatenate datasets by rows (schema-union) |
| `ConcatLayer` | `ConcatLayer[opts?]` | Concatenate along channel/feature axis |
| `DatasetFromRows` | `DatasetFromRows[rows]` | Create dataset from list of row assocs |
| `DatasetSchema` | `DatasetSchema[ds]` | Describe schema for a dataset |
| `Distinct` | `Distinct[ds, cols?]` | Drop duplicate rows (optionally by columns) |
| `DistinctBy` | `DistinctBy[dataset, keys, opts?]` | Alias for DistinctOn |
| `DistinctOn` | `DistinctOn[ds, keys, orderBy?, keepLast?]` | Keep one row per key with order policy |
| `ExplainDataset` | `ExplainDataset[ds]` | Inspect logical plan for a dataset |
| `ExplainSQL` | `ExplainSQL[ds]` | Render SQL for pushdown-capable parts |
| `FilterFIR` | `FilterFIR[x, coeffs, opts?]` | Finite impulse response filter (stub) |
| `FilterIIR` | `FilterIIR[x, coeffs, opts?]` | Infinite impulse response filter (stub) |
| `FilterRows` | `FilterRows[ds, pred]` | Filter rows by predicate (held) |
| `GroupBy` | `GroupBy[ds, keys]` | Group rows by key(s) |
| `JoinLines` | `JoinLines[lines]` | Join list into lines with 
 |
| `Limit` | `Limit[dataset, n]` | Alias for Head on Dataset |
| `RenameCols` | `RenameCols[ds, mapping]` | Rename columns via mapping |
| `SelectCols` | `SelectCols[ds, cols]` | Select subset of columns by name |
| `Table` | `Table[conn, name]` | Reference a table as a Dataset |
| `UnionByPosition` | `UnionByPosition[ds1, ds2, …]` | Union datasets by column position. |
| `WithColumns` | `WithColumns[ds, defs]` | Add/compute new columns (held) |
| `WriteDataset` | `WriteDataset[conn, table, dataset, opts?]` | Write a Dataset into a table |
| `col` | `col[name]` | Column accessor helper for Dataset expressions. |

## `Agg`

- Usage: `Agg[ds, aggs]`
- Summary: Aggregate groups to single rows
- Tags: dataset, aggregate
- Examples:
  - `Agg[grouped, <|"n"->Count[], "avg"->Mean[salary]|>]  ==> ds'`

## `Aggregate`

- Usage: `Aggregate[group, spec]`
- Summary: Aggregate grouped data (stub)
- Tags: frame, group, aggregate
- Examples:
  - `Aggregate[g, <|Count->True|>]`

## `Coalesce`

- Usage: `Coalesce[values…]`
- Summary: First non-null value
- Tags: dataset, types
- Examples:
  - `Coalesce[Null, 0, 42]  ==> 0`

## `ExplainDataset`

- Usage: `ExplainDataset[ds]`
- Summary: Inspect logical plan for a dataset
- Tags: dataset, explain
- Examples:
  - `ExplainDataset[ds]  ==> <|plan->...|>`

## `FilterFIR`

- Usage: `FilterFIR[x, coeffs, opts?]`
- Summary: Finite impulse response filter (stub)
- Examples:
  - `FilterFIR[{1,2,3}, {0.2,0.2,0.2}]  ==> {...}`

## `FilterIIR`

- Usage: `FilterIIR[x, coeffs, opts?]`
- Summary: Infinite impulse response filter (stub)
- Examples:
  - `FilterIIR[{1,2,3}, {1.0, -0.5}]  ==> {...}`

## `FilterRows`

- Usage: `FilterRows[ds, pred]`
- Summary: Filter rows by predicate (held)
- Tags: dataset, transform, filter
- Examples:
  - `FilterRows[ds, age > 30]  ==> ds'`

## `GroupBy`

- Usage: `GroupBy[ds, keys]`
- Summary: Group rows by key(s)
- Tags: frame, group
- Examples:
  - `GroupBy[ds, dept]  ==> grouped`

## `Table`

- Usage: `Table[conn, name]`
- Summary: Reference a table as a Dataset
- Tags: db, sql, dataset
- Examples:
  - `ds := Table[conn, "t"]; Head[ds,1]  ==> {<|...|>}`
