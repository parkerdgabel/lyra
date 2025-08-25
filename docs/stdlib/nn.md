Lyra Neural Networks (WL-inspired)

Overview
- NetChain/NetGraph build neural nets as function objects.
- NetTrain returns a callable model; NetApply runs inference.
- NetProperty/NetSummary expose structure and training metadata.

Core
- NetChain[{layers...}, opts] -> Net
- NetGraph[layerAssoc, edges, opts] -> Net
- NetInitialize[net, opts] -> Net
- NetTrain[net, data, opts] -> TrainedNetFunction
- NetApply[net, x, opts] -> y
- NetProperty[net, key]
- `NetProperty[net, "LayerSummaries"]` returns a list of per-layer summaries
  with key params (e.g., Linear: Output, Bias; Activation: Type; Conv/Pool: sizes).
- NetSummary[net]
- NetEncoder[spec], NetDecoder[spec]

Layers (constructors)
- LinearLayer[output, opts]
  - Options: Bias->True|False (default True)
- ActivationLayer["ReLU"|"Sigmoid"|"Tanh"|"GELU"|"Softmax", opts]
- DropoutLayer[rate]
- FlattenLayer[]
- SoftmaxLayer[] (alias of ActivationLayer["Softmax"])
- ConvolutionLayer[outChannels, kernelSize, opts]  (1D)
- PoolingLayer["Max"|"Avg", size, opts]          (1D)
- BatchNormLayer[opts]
- LayerNormLayer[opts]
- ReshapeLayer[{d1,d2,...}] (output-only reshape in MVP)
- TransposeLayer[perm] (constructor only)
- ConcatLayer[axis] (constructor only)
- EmbeddingLayer[vocab, dim] (first layer; int token -> vector)

MVP Notes
- Current implementation is a scaffold:
  - NetApply supports a minimal forward pass for `NetChain` with `LinearLayer`,
    `ActivationLayer`, `ConvolutionLayer` (1D), `PoolingLayer` (1D), `BatchNormLayer`,
    `LayerNormLayer`, `EmbeddingLayer` (first layer), and `ReshapeLayer` (final output only)
    on numeric scalars/lists. Parameters are lazily initialized
    on first apply using a deterministic small random initializer.
  - Training updates metadata (epochs, method, batch size) but does not optimize
    parameters yet.
- Structure and names mirror Wolfram Language for future expansion.
