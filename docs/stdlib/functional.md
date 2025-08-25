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

