Functional Programming (Lyra stdlib)

Core combinators inspired by the Wolfram Language’s Functional Programming guide. Names and semantics follow Lyra conventions and the existing stdlib. Operator forms are supported where it improves ergonomics.

Highlights
- Apply: Replace head or spread a list as arguments.
  - Apply[f, {a,b,c}] -> f[a,b,c]
  - Apply[f, g[x,y]] -> f[x,y]
- Compose: Build a function composition left-to-right.
  - Compose[f, g, h] returns a pure function f[g[h[#]]]&
  - RightCompose[f, g, h] returns h[g[f[#]]]&
- Nest/NestList: Repeated application.
  - Nest[f, x, n]
  - NestList[f, x, n] -> {x, f[x], f[f[x]], ...}
- FoldList: Running reduction results.
  - FoldList[f, list]
  - FoldList[f, init, list]
- FixedPoint/FixedPointList: Iterate until no further change (with MaxIterations option, default 100).
- Through: Apply a list of functions to an argument.
  - Through[{f,g}, x] -> {f[x], g[x]}
  - Through[{f,g}] returns a pure function that maps {f,g} over its argument.
- Identity and ConstantFunction: Utility function constructors.
  - Identity[] -> #&; Identity[x] -> x
  - ConstantFunction[c] -> pure function returning c

Operator Forms (partial application)
- Map[f] -> pure function expecting a list: Map[f][xs] == Map[f, xs]
- Reduce[f] -> pure function expecting a list
- Reduce[f, init] -> pure function expecting a list
- Scan[f] and Scan[f, init] similarly
- Compose and RightCompose directly return pure functions

Notes
- These forms are eager on their data arguments but respect Lyra’s normal evaluation for function arguments.
- FixedPoint comparisons are structural equality; customize behavior by pre/post-processing your state as needed.

Operators
- `/@` (Map): `f /@ expr` is `Map[f, expr]`.
  - Example: `(#*#)& /@ {1,2,3}` -> `{1,4,9}`.
- `@@` (Apply): `f @@ expr` is `Apply[f, expr]`.
  - Example: `Plus @@ {1,2,3}` -> `6` (spreads the list as arguments to `Plus`).
- `@@@` (MapApply level 1): `f @@@ expr` is `Apply[f, expr, 1]`.
  - Example: `Plus @@@ {{1,2},{3,4}}` -> `{3,7}` (applies `Plus @@` to each element of the outer list).

SetDelayed (:=)
- Define delayed values and function rules that evaluate the right-hand side at use time.
  - Symbol (OwnValue): `y := Plus[2,3]; y` -> `5`.
  - DownValue (function): `f[x_] := x*x; f[5]` -> `25`.
  - SubValue (curried heads): `(f[a_])[b_] := a + b; f[2][3]` -> `5`.
  
Details
- `SetDelayed[lhs, rhs]` stores `rhs` unevaluated as a rule on the appropriate value store:
  - `x := rhs` -> OwnValues on `x` with rule `x -> rhs`.
  - `f[pat...] := rhs` -> DownValues on `f` with rule `f[pat...] -> rhs`.
  - `(f[...])[...] := rhs` -> SubValues on `f` with the full compound `lhs`.
- Rules fire during evaluation; the rule’s right-hand side is evaluated with the matched bindings and current environment.
