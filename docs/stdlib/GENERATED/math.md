# MATH

| Function | Usage | Summary |
|---|---|---|
| `Abs` | `Abs[x]` | Absolute value |
| `Correlation` | `Correlation[a, b]` | Pearson correlation of two numeric lists (population moments). |
| `Covariance` | `Covariance[a, b]` | Covariance of two numeric lists (population). |
| `Kurtosis` | `Kurtosis[data]` | Kurtosis (fourth standardized moment). |
| `Max` | `Max[args]` | Maximum of values or list |
| `Min` | `Min[args]` | Minimum of values or list |
| `Mode` | `Mode[data]` | Most frequent element (ties broken by first appearance). |
| `Percentile` | `Percentile[data, p|list]` | Percentile(s) of numeric data using R-7 interpolation. |
| `Plus` | `Plus[a, b, …]` | Add numbers; Listable, Flat, Orderless. |
| `Quantile` | `Quantile[data, q|list]` | Quantile(s) of numeric data using R-7 interpolation. |
| `Skewness` | `Skewness[data]` | Skewness (third standardized moment). |
| `Times` | `Times[a, b, …]` | Multiply numbers; Listable, Flat, Orderless. |

## `Correlation`

- Usage: `Correlation[a, b]`
- Summary: Pearson correlation of two numeric lists (population moments).
- Tags: math, stats
- Examples:
  - `Correlation[{1,2,3},{2,4,6}]  ==> 1.0`

## `Covariance`

- Usage: `Covariance[a, b]`
- Summary: Covariance of two numeric lists (population).
- Tags: math, stats
- Examples:
  - `Covariance[{1,2,3},{2,4,6}]  ==> 2.0`

## `Mode`

- Usage: `Mode[data]`
- Summary: Most frequent element (ties broken by first appearance).
- Tags: math, stats
- Examples:
  - `Mode[{1,2,2,3}]  ==> 2`

## `Percentile`

- Usage: `Percentile[data, p|list]`
- Summary: Percentile(s) of numeric data using R-7 interpolation.
- Tags: math, stats
- Examples:
  - `Percentile[{1,2,3,4}, 25]  ==> 1.75`

## `Plus`

- Usage: `Plus[a, b, …]`
- Summary: Add numbers; Listable, Flat, Orderless.
- Tags: math, sum
- Examples:
  - `Plus[1, 2]  ==> 3`
  - `Plus[1, 2, 3]  ==> 6`
  - `Plus[{1,2,3}]  ==> {1,2,3} (Listable)`

## `Quantile`

- Usage: `Quantile[data, q|list]`
- Summary: Quantile(s) of numeric data using R-7 interpolation.
- Tags: math, stats
- Examples:
  - `Quantile[{1,2,3,4}, 0.25]  ==> 1.75`
  - `Quantile[{1,2,3,4}, {0.25,0.5}]  ==> {1.75, 2.5}`

## `Times`

- Usage: `Times[a, b, …]`
- Summary: Multiply numbers; Listable, Flat, Orderless.
- Tags: math, product
- Examples:
  - `Times[2, 3]  ==> 6`
  - `Times[2, 3, 4]  ==> 24`
  - `Times[{2,3}, 10]  ==> {20,30}`
