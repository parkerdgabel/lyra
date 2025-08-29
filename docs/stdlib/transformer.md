# Transformer Encoder–Decoder

This guide shows how to use the high-level Transformer components in Lyra’s NN API to build an encoder–decoder pipeline with masks.

## Primitives

- `MultiHeadAttention[<|SeqLen->L, ModelDim->D, NumHeads->H, Mask->...|>]`
- `PositionalEncoding[<|SeqLen->L, ModelDim->D|>]`

## Encoder

- `TransformerEncoder[<|SeqLen->L, ModelDim->D, NumHeads->H, HiddenDim->M, Activation->Gelu|>]`
- `TransformerEncoderStack[<|Layers->N, SeqLen->L, ModelDim->D, NumHeads->H, HiddenDim->M|>]`

Example (stack of 2 encoders):

```
enc = TransformerEncoderStack[<|
  Layers->2,
  SeqLen->8,
  ModelDim->64,
  NumHeads->8,
  HiddenDim->256
|>];
```

Compute memory from a source sequence (flattened `8x64`):

```
source = Tensor[{{... 8 tokens of 64-dim ...}}];
mem = Predict[enc, source];  (* shape: {8,64} *)
```

## Decoder

- `TransformerDecoder[<|SeqLen->L, ModelDim->D, NumHeads->H, HiddenDim->M, Activation->Gelu, Causal->True|>]`
- `TransformerDecoderStack[<|Layers->N, SeqLen->L, ModelDim->D, NumHeads->H, HiddenDim->M, Causal->True|>]`

The decoder performs self-attention (optionally causal), cross-attention over the provided `Memory` (encoder output), and an FFN, with residual connections and LayerNorm.

Example (single decoder block with memory):

```
dec = Sequential[{TransformerDecoder[<|
  SeqLen->8,
  ModelDim->64,
  NumHeads->8,
  HiddenDim->256,
  Causal->True,
  Memory->mem
|>]}];
```

Run on a target sequence (flattened `8x64`):

```
target = Tensor[{{... 8 tokens of 64-dim ...}}];
out = Predict[dec, target];  (* shape: {8,64} *)
```

### Cross-Attention (standalone)

You can use cross-attention directly by supplying a `Memory` matrix and optional `MemoryMask` (or `SourceMask`).

```
layer = CrossAttention[<|
  SeqLen->8,
  ModelDim->64,
  NumHeads->8,
  Memory->mem,
  MemoryMask->Table[1, {8}]  (* or a {8x8} matrix *)
|>];
net = Sequential[{layer}];
out = Predict[net, target];
```

## Full Encoder–Decoder Stack

```
enc = TransformerEncoderStack[<|Layers->4, SeqLen->8, ModelDim->64, NumHeads->8, HiddenDim->256|>];
mem = Predict[enc, source];

dec = TransformerDecoderStack[<|Layers->4, SeqLen->8, ModelDim->64, NumHeads->8, HiddenDim->256, Causal->True, Memory->mem|>];
out = Predict[dec, target];
```

### Convenience: Encoder+Decoder builder

Use `TransformerEncoderDecoder` to build both stacks with shared parameters. It returns an association with `Encoder` and `Decoder` networks; compute memory with the encoder, then feed it to the decoder.

```
e2d = TransformerEncoderDecoder[<|
  EncLayers->4, DecLayers->4,
  SeqLen->8, ModelDim->64, NumHeads->8, HiddenDim->256, Causal->True
|>];
enc = e2d["Encoder"];  dec = e2d["Decoder"];
mem = Predict[enc, source];
(* Optionally build a MemoryMask here *)
out = Predict[dec /. <|"Memory"->mem|>, target];
```

## Masks

Self-attention mask options (decoder):
- `Causal->True`: Applies an upper-triangular mask to prevent attending to future positions.
- `Mask->vec`: Vector length `SeqLen`; 1=keep, 0=mask for each key position.
- `Mask->mat`: Matrix `SeqLen x SeqLen`; 1=keep, 0=mask, applied per (query,key).
- `SelfMask->...`: Like `Mask` but explicitly overrides `Causal`/`Mask` if both present.

Cross-attention mask options:
- `MemoryMask->vec`: Vector length `memLen` (encoder time steps); 1=keep, 0=mask.
- `MemoryMask->mat`: Matrix `SeqLen x memLen`.
- `SourceMask` is accepted as an alias for `MemoryMask`.

Notes:
- Current implementation is single-batch.
- Shapes are inferred from `SeqLen` and `ModelDim`, or from the input tensor if not provided.
- Provide `Memory` as a tensor or list-of-lists shaped `memLen x ModelDim`.
