Lyra ML (WL-inspired)

Overview
- Classify, Predict, Cluster, FeatureExtract, DimensionReduce return callable function objects.
- Callable models support two-argument form for options: fn[input, <|...|>].
- Use MLProperty[model, prop] to query model metadata.
- Use ClassifyMeasurements/ PredictMeasurements for quick evaluation.

Core
- Classify[train, opts] -> ClassifierFunction
- Predict[train, opts] -> PredictorFunction
- Cluster[train, opts] -> ClusterFunction
- FeatureExtract[train, opts] -> TransformerFunction
- DimensionReduce[train, opts] -> TransformerFunction

Training data forms
- Supervised pairs: {x1 -> y1, x2 -> y2, ...}
- Dataset with Target -> "col"
- Arrays: Predict[X, y], Classify[X, y] (X: NDArray, y: vector)

Calling
- clf[x] -> label
- clf[x, <| "Output" -> "Probabilities" |>] -> assoc of class->prob
- pred[x] -> numeric prediction

Properties
- MLProperty[model, "Properties"] -> list
- MLProperty[model, "Method" | "Task" | "TrainingSize" | "Classes"]

Measurements
- ClassifyMeasurements[clf, {x -> y, ...}] -> <| "Accuracy" -> r, "ConfusionMatrix" -> ... |>
- PredictMeasurements[pred, {x -> y, ...}] -> <| "RMSE" -> r |>

Notes
- v1 methods are baseline stubs (majority class and mean predictor). Interfaces are stable for future methods.
