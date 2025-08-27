# ASSOC

| Function | Usage | Summary |
|---|---|---|
| `AssocContainsKeyQ` | `AssocContainsKeyQ[assoc, key]` | Does association contain key? |
| `AssocDelete` | `AssocDelete[assoc, keys]` | Delete keys from association |
| `AssocDrop` | `AssocDrop[assoc, keys]` | Drop keys and return remaining assoc |
| `AssocGet` | `AssocGet[assoc, key, default?]` | Get value by key with optional default |
| `AssocInvert` | `AssocInvert[assoc]` | Invert mapping values -> list of keys |
| `AssocQ` | `AssocQ[x]` | Is value an association (map)? |
| `AssocRenameKeys` | `AssocRenameKeys[assoc, map|f]` | Rename keys by mapping or function |
| `AssocSelect` | `AssocSelect[assoc, pred|keys]` | Filter keys by predicate or list |
| `AssocSet` | `AssocSet[assoc, key, value]` | Set key to value (returns new assoc) |
| `AssociationMap` | `AssociationMap[f, assoc]` | Map values with f[v] |
| `AssociationMapKV` | `AssociationMapKV[fn, assoc]` | Map key/value pairs in an association |
| `AssociationMapKeys` | `AssociationMapKeys[f, assoc]` | Map keys with f[k] |
| `AssociationMapPairs` | `AssociationMapPairs[f, assoc]` | Map over (k,v) pairs |
| `Columns` | `Columns[ds]` | List column names for a dataset |
| `KeySort` | `KeySort[assoc]` | Sort association by key |
| `Keys` | `Keys[subject]` | Keys/columns for Assoc/rows/Dataset/Frame |
| `Lookup` | `Lookup[assoc, key, default]` | Lookup value from association |
| `Merge` | `Merge[args]` | Merge associations with optional combiner |
| `Select` | `Select[ds, cols]` | Select/compute columns |
| `SortBy` | `SortBy[keyFn, assoc]` | Sort association by derived key |
| `Values` | `Values[assoc]` | List values of an association |

## `AssocGet`

- Usage: `AssocGet[assoc, key, default?]`
- Summary: Get value by key with optional default
- Examples:
  - `AssocGet[<|"a"->1|>, "a"]  ==> 1`
  - `AssocGet[<||>, "k", 0]  ==> 0`

## `AssocSet`

- Usage: `AssocSet[assoc, key, value]`
- Summary: Set key to value (returns new assoc)
- Examples:
  - `AssocSet[<|"a"->1|>, "b", 2]  ==> <|"a"->1, "b"->2|>`

## `AssociationMap`

- Usage: `AssociationMap[f, assoc]`
- Summary: Map values with f[v]
- Tags: assoc
- Examples:
  - `AssociationMap[ToUpper, <|"a"->"x"|>]  ==> <|"a"->"X"|>`

## `Keys`

- Usage: `Keys[subject]`
- Summary: Keys/columns for Assoc/rows/Dataset/Frame
- Tags: generic, schema, assoc, dataset, frame
- Examples:
  - `Keys[<|a->1,b->2|>]  ==> {a,b}`
  - `Keys[{<|a->1|>,<|b->2|>}]  ==> {a,b}`
  - `Keys[ds] (* Columns *)`
  - `Keys[f]  (* Columns *)`

## `Select`

- Usage: `Select[ds, cols]`
- Summary: Select/compute columns
- Tags: generic, dataset, frame, assoc
- Examples:
  - `Select[ds, <|"name"->name, "age2"->age*2|>]  ==> ds'`
