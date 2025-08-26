# DATASET

| Function | Usage | Summary |
|---|---|---|
| `Agg` | `Agg[ds, aggs]` | Aggregate groups to single rows |
| `Concat` | `Concat[inputs]` | Concatenate datasets by rows (schema-union) |
| `ConcatLayer` | `ConcatLayer[axis]` | Concatenate along axis |
| `DatasetFromRows` | `DatasetFromRows[rows]` | Create dataset from list of row assocs |
| `DatasetSchema` | `DatasetSchema[ds]` | Describe schema for a dataset |
| `Distinct` | `Distinct[ds, cols?]` | Drop duplicate rows (optionally by columns) |
| `DistinctOn` | `DistinctOn[ds, keys, orderBy?, keepLast?]` | Keep one row per key with order policy |
| `ExplainDataset` | `ExplainDataset[ds]` | Inspect logical plan for a dataset |
| `FilterRows` | `FilterRows[ds, pred]` | Filter rows by predicate (held) |
| `JoinLines` | `JoinLines[lines]` | Join list into lines with 
 |
| `Select` | `Select[ds, cols]` | Select/compute columns |
| `SelectCols` | `SelectCols[ds, cols]` | Select subset of columns by name |
| `Union` | `Union[inputs, byColumns?]` | Union multiple datasets (by columns) |
| `UnionByPosition` | `UnionByPosition[ds1, ds2, …]` | Union datasets by column position. |

## `Agg`

- Usage: `Agg[ds, aggs]`
- Summary: Aggregate groups to single rows
- Examples:
  - `Agg[grouped, <|"n"->Count[], "avg"->Mean[salary]|>]  ==> ds'`

## `ExplainDataset`

- Usage: `ExplainDataset[ds]`
- Summary: Inspect logical plan for a dataset
- Examples:
  - `ExplainDataset[ds]  ==> <|plan->...|>`

## `FilterRows`

- Usage: `FilterRows[ds, pred]`
- Summary: Filter rows by predicate (held)
- Examples:
  - `FilterRows[ds, age > 30]  ==> ds'`

## `Select`

- Usage: `Select[ds, cols]`
- Summary: Select/compute columns
- Examples:
  - `Select[ds, <|"name"->name, "age2"->age*2|>]  ==> ds'`
