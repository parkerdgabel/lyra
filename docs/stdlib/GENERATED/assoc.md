# ASSOC

| Function | Usage | Summary |
|---|---|---|
| `AssocQ` | `AssocQ[x]` | Is value an association (map)? |
| `AssociationMap` | `AssociationMap[fn, assoc]` | Map values in an association |
| `AssociationMapKV` | `AssociationMapKV[fn, assoc]` | Map key/value pairs in an association |
| `AssociationMapKeys` | `AssociationMapKeys[fn, assoc]` | Map keys in an association |
| `AssociationMapPairs` | `AssociationMapPairs[fn, assoc]` | Map to key/value pairs or assoc |
| `Columns` | `Columns[ds]` | List column names for a dataset |
| `KeySort` | `KeySort[assoc]` | Sort association by key |
| `Keys` | `Keys[subject]` | Keys/columns for assoc, rows, Dataset, or Frame |
| `Merge` | `Merge[args]` | Merge associations with optional combiner |
| `Select` | `Select[assoc|ds, pred|keys|cols]` | Select keys/columns or compute columns (dispatched). Overloads: Select[assoc, pred\|keys]; Select[ds, cols] |
| `Values` | `Values[assoc]` | List values of an association |

## `Keys`

- Usage: `Keys[subject]`
- Summary: Keys/columns for assoc, rows, Dataset, or Frame
- Tags: generic, schema, assoc, dataset, frame
- Examples:
  - `Keys[<|a->1,b->2|>]  ==> {a,b}`
  - `Keys[{<|a->1|>,<|b->2|>}]  ==> {a,b}`

## `Select`

- Usage: `Select[assoc|ds, pred|keys|cols]`
- Summary: Select keys/columns or compute columns (dispatched). Overloads: Select[assoc, pred|keys]; Select[ds, cols]
- Tags: generic, dataset, frame, assoc
- Examples:
  - `Select[ds, <|"name"->name, "age2"->age*2|>]  ==> ds'`
