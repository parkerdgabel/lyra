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
| `KeySort` | `KeySort[assoc]` | Sort association by key |
| `Keys` | `Keys[assoc]` | List keys of an association |
| `Lookup` | `Lookup[assoc, key, default]` | Lookup value from association |
| `Merge` | `Merge[args]` | Merge associations with optional combiner |
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
- Examples:
  - `AssociationMap[ToUpper, <|"a"->"x"|>]  ==> <|"a"->"X"|>`
