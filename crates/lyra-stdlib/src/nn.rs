use crate::register_if;
#[cfg(feature = "tools")]
use crate::tool_spec;
#[cfg(feature = "tools")]
use crate::tools::add_specs;
use lyra_core::value::Value;
use lyra_runtime::attrs::Attributes;
use lyra_runtime::Evaluator;
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};
use std::env;

type NativeFn = fn(&mut Evaluator, Vec<Value>) -> Value;

#[derive(Clone, Default)]
struct NetState {
    kind: String,         // "Chain" | "Graph"
    layers: Vec<Value>,   // raw layer specs (opaque Values)
    graph: Option<Value>, // edges/ports (opaque)
    #[allow(dead_code)]
    opts: HashMap<String, Value>, // construction options
    encoder: Option<Value>,
    decoder: Option<Value>,
    initialized: bool,
    trained_epochs: usize,
    method: String, // optimizer (e.g., Adam)
    batch_size: usize,
}

static NN_REG: OnceLock<Mutex<HashMap<i64, NetState>>> = OnceLock::new();
static NEXT_ID: OnceLock<std::sync::atomic::AtomicI64> = OnceLock::new();
static MAX_OPS: OnceLock<Option<usize>> = OnceLock::new();

fn reg() -> &'static Mutex<HashMap<i64, NetState>> {
    NN_REG.get_or_init(|| Mutex::new(HashMap::new()))
}
fn next_id() -> i64 {
    let a = NEXT_ID.get_or_init(|| std::sync::atomic::AtomicI64::new(1));
    a.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
}

fn net_handle(id: i64) -> Value {
    Value::assoc(vec![("__type", Value::String("Net".into())), ("id", Value::Integer(id))])
}
fn get_net_id(v: &Value) -> Option<i64> {
    if let Value::Assoc(m) = v {
        if matches!(m.get("__type"), Some(Value::String(s)) if s=="Net") {
            if let Some(Value::Integer(id)) = m.get("id") {
                return Some(*id);
            }
        }
    }
    None
}

fn max_ops_limit() -> Option<usize> {
    // Read LYRA_NN_MAX_OPS once; 0 or missing disables guard
    *MAX_OPS.get_or_init(|| {
        match env::var("LYRA_NN_MAX_OPS") {
            Ok(s) => {
                let st = s.trim().to_ascii_lowercase();
                if st.is_empty() || st == "0" || st == "off" || st == "none" || st == "unlimited" {
                    None
                } else {
                    st.parse::<usize>().ok().filter(|v| *v > 0)
                }
            }
            Err(_) => None,
        }
    })
}

fn error_assoc(msg: &str) -> Value {
    Value::Assoc(HashMap::from([(String::from("Error"), Value::String(msg.into()))]))
}

pub fn register_nn(ev: &mut Evaluator) {
    ev.register("NetChain", net_chain as NativeFn, Attributes::empty());
    ev.register("NetGraph", net_graph as NativeFn, Attributes::empty());
    ev.register("NetInitialize", net_initialize as NativeFn, Attributes::empty());
    ev.register("NetTrain", net_train as NativeFn, Attributes::empty());
    // Canonical training alias
    ev.register("Fit", fit as NativeFn, Attributes::empty());
    ev.register("NetApply", net_apply as NativeFn, Attributes::empty());
    ev.register("NetProperty", net_property as NativeFn, Attributes::empty());
    ev.register("NetSummary", net_summary as NativeFn, Attributes::empty());
    ev.register("NetEncoder", net_encoder as NativeFn, Attributes::empty());
    ev.register("NetDecoder", net_decoder as NativeFn, Attributes::empty());
    // Layers (constructors return opaque layer specs)
    ev.register("LinearLayer", linear_layer as NativeFn, Attributes::empty());
    ev.register("ActivationLayer", activation_layer as NativeFn, Attributes::empty());
    ev.register("DropoutLayer", dropout_layer as NativeFn, Attributes::empty());
    ev.register("FlattenLayer", flatten_layer as NativeFn, Attributes::empty());
    ev.register("SoftmaxLayer", softmax_layer as NativeFn, Attributes::empty());
    ev.register("ConvolutionLayer", convolution_layer as NativeFn, Attributes::empty());
    ev.register("PoolingLayer", pooling_layer as NativeFn, Attributes::empty());
    ev.register("BatchNormLayer", batchnorm_layer as NativeFn, Attributes::empty());
    ev.register("ReshapeLayer", reshape_layer as NativeFn, Attributes::empty());
    ev.register("TransposeLayer", transpose_layer as NativeFn, Attributes::empty());
    ev.register("ConcatLayer", concat_layer as NativeFn, Attributes::empty());
    ev.register("AddLayer", add_layer as NativeFn, Attributes::empty());
    ev.register("MulLayer", mul_layer as NativeFn, Attributes::empty());
    ev.register("EmbeddingLayer", embedding_layer as NativeFn, Attributes::empty());
    ev.register("LayerNormLayer", layernorm_layer as NativeFn, Attributes::empty());

    // Canonical heads (naming conventions)
    ev.register("Network", network as NativeFn, Attributes::empty());
    ev.register("Sequential", sequential as NativeFn, Attributes::empty());
    ev.register("GraphNetwork", graph_network as NativeFn, Attributes::empty());
    ev.register("Initializer", initializer as NativeFn, Attributes::empty());

    // Layer constructors (noun heads)
    ev.register("Dense", dense as NativeFn, Attributes::empty());
    ev.register("Convolution1D", convolution1d as NativeFn, Attributes::empty());
    ev.register("Convolution2D", convolution2d as NativeFn, Attributes::empty());
    ev.register("DepthwiseConv2D", depthwise_conv2d as NativeFn, Attributes::empty());
    ev.register("ConvTranspose2D", conv_transpose2d as NativeFn, Attributes::empty());
    ev.register("SeparableConv2D", separable_conv2d as NativeFn, Attributes::empty());
    ev.register("Pooling", pooling as NativeFn, Attributes::empty());
    ev.register("Pooling2D", pooling2d as NativeFn, Attributes::empty());
    ev.register("GlobalAvgPool2D", global_avg_pool2d as NativeFn, Attributes::empty());
    ev.register("GroupNorm", group_norm as NativeFn, Attributes::empty());
    ev.register("Residual", residual as NativeFn, Attributes::empty());
    ev.register("Upsample2D", upsample2d as NativeFn, Attributes::empty());
    ev.register("ResidualBlock", residual_block as NativeFn, Attributes::empty());
    ev.register("BatchNorm", batchnorm as NativeFn, Attributes::empty());
    ev.register("LayerNorm", layernorm as NativeFn, Attributes::empty());
    ev.register("Flatten", flatten as NativeFn, Attributes::empty());
    ev.register("Reshape", reshape as NativeFn, Attributes::empty());
    // Transformer layers
    ev.register("MultiHeadAttention", multi_head_attention as NativeFn, Attributes::empty());
    ev.register("PositionalEncoding", positional_encoding as NativeFn, Attributes::empty());
    ev.register("TransformerEncoder", transformer_encoder as NativeFn, Attributes::empty());
    ev.register("TransformerEncoderStack", transformer_encoder_stack as NativeFn, Attributes::empty());
    ev.register("TransformerDecoder", transformer_decoder as NativeFn, Attributes::empty());
    ev.register("TransformerDecoderStack", transformer_decoder_stack as NativeFn, Attributes::empty());
    ev.register("TransformerEncoderDecoder", transformer_encoder_decoder as NativeFn, Attributes::empty());
    // New layers
    ev.register("RMSNorm", rms_norm as NativeFn, Attributes::empty());
    ev.register("FFN", ffn as NativeFn, Attributes::empty());
    ev.register("CausalSelfAttention", causal_self_attention as NativeFn, Attributes::empty());
    ev.register("CrossAttention", cross_attention as NativeFn, Attributes::empty());
    ev.register("PositionalEmbedding", positional_embedding as NativeFn, Attributes::empty());
    ev.register("PatchEmbedding2D", patch_embedding2d as NativeFn, Attributes::empty());
    // Transpose tensor op handled by dispatcher; expose internal layer alias only
    ev.register("__TransposeLayer", __transpose_layer as NativeFn, Attributes::empty());
    // Avoid clobbering Dataset Concat; keep internal layer alias only
    ev.register("__ConcatLayer", __concat_layer as NativeFn, Attributes::empty());
    ev.register("__AddLayer", add as NativeFn, Attributes::empty());
    ev.register("__MulLayer", mul as NativeFn, Attributes::empty());
    ev.register("Embedding", embedding as NativeFn, Attributes::empty());
    // Internal: activation via zero-arg Relu[]/Tanh[]/...
    ev.register("__ActivationLayer", activation_layer as NativeFn, Attributes::empty());

    #[cfg(feature = "tools")]
    add_specs(vec![
        // Canonical heads
        tool_spec!("Sequential", summary: "Construct a sequential network from layers", params: ["layers","opts?"], tags: ["nn","network"], examples: [
            Value::String("Sequential[{Dense[<|Output->4|>], Relu[]}]".into()),
            Value::String("(* MLP *) Sequential[{Flatten[], Dense[<|Output->64|>], Relu[], Dense[<|Output->10|>], Softmax[]}]".into()),
            Value::String("(* Conv block *) Sequential[{Convolution2D[<|Output->8, KernelSize->3, Padding->1, InputChannels->1, Height->28, Width->28|>], Relu[], Pooling2D[\"Max\", 2, <|InputChannels->8, Height->28, Width->28|>]}]".into()),
            Value::String("(* End-to-end conv → dense *) net := Sequential[{Convolution2D[<|Output->8, KernelSize->3, Padding->1, InputChannels->1, Height->28, Width->28|>], Relu[], Pooling2D[\"Max\", 2], Flatten[], Dense[<|Output->10|>], Softmax[]}]; Shape[Predict[net, Tensor[(* 1x28x28 image tensor here *)]]]  ==> {10}".into())
        ]),
        tool_spec!("Network", summary: "Construct a network from assoc (Kind->Chain|Graph)", params: ["spec"], tags: ["nn","network"], examples: [
            Value::String("Network[<|Kind->\"Chain\", Layers->{Dense[<|Output->2|>], Softmax[]}|>]".into())
        ]),
        tool_spec!("GraphNetwork", summary: "Construct a graph network from nodes/edges", params: ["nodes","edges","opts?"], tags: ["nn","network"]),
        tool_spec!("Initializer", summary: "Initializer spec for layer parameters", params: ["opts?"], tags: ["nn","init"], examples: [
            Value::String("Initializer[<|Type->\"Xavier\"|>]".into())
        ]),
        // Layers (noun heads)
        tool_spec!("Dense", summary: "Linear (fully-connected) layer", params: ["opts?"], tags: ["nn","layer"]),
        tool_spec!("Convolution1D", summary: "1D convolution layer", params: ["opts?"], tags: ["nn","layer"]),
        tool_spec!("Convolution2D", summary: "2D convolution layer (uses InputChannels/Height/Width for forward)", params: ["opts?"], tags: ["nn","layer"], examples: [
            Value::String("Sequential[{Convolution2D[<|Output->1, KernelSize->{2,2}, InputChannels->1, Height->2, Width->2, W->{{{{1,1},{1,1}}}}, b->{0}|>}]]".into())
        ]),
        tool_spec!("DepthwiseConv2D", summary: "Depthwise 2D convolution (per-channel)", params: ["opts?"], tags: ["nn","layer"]),
        tool_spec!("ConvTranspose2D", summary: "Transposed 2D convolution (deconv)", params: ["opts?"], tags: ["nn","layer"], examples: [
            Value::String("Sequential[{ConvTranspose2D[<|Output->1, KernelSize->2, Stride->2, InputChannels->1, Height->1, Width->1|>]}]".into()),
            Value::String("(* Upsample feature map 16x16 -> 32x32 *) Sequential[{ConvTranspose2D[<|Output->8, KernelSize->4, Stride->2, Padding->1, InputChannels->16, Height->16, Width->16|>], Relu[]}]".into())
        ]),
        tool_spec!("SeparableConv2D", summary: "Depthwise + 1x1 pointwise convolution", params: ["opts?"], tags: ["nn","layer"], examples: [
            Value::String("Sequential[{SeparableConv2D[<|Output->2, KernelSize->3, InputChannels->1, Height->8, Width->8|>]}]".into())
        ]),
        tool_spec!("Pooling", summary: "Pooling layer (Max/Avg)", params: ["kind","size","opts?"], tags: ["nn","layer"]),
        tool_spec!("Pooling2D", summary: "2D pooling layer (Max/Avg; requires InputChannels/Height/Width)", params: ["kind","size","opts?"], tags: ["nn","layer"], examples: [
            Value::String("Sequential[{Pooling2D[\"Max\", 2, <|InputChannels->1, Height->2, Width->2|>]}]".into())
        ]),
        tool_spec!("GlobalAvgPool2D", summary: "Global average pooling per channel over HxW", params: ["opts?"], tags: ["nn","layer"], examples: [
            Value::String("Sequential[{GlobalAvgPool2D[<|InputChannels->16, Height->7, Width->7|>], Dense[<|Output->10|>], Softmax[]}]".into())
        ]),
        tool_spec!("BatchNorm", summary: "Batch normalization layer", params: ["opts?"], tags: ["nn","layer"]),
        tool_spec!("Dropout", summary: "Dropout layer (training only; stub)", params: ["opts?"], tags: ["nn","layer"], examples: [
            Value::String("Sequential[{Dense[<|Output->64|>], Dropout[<|P->0.1|>], Relu[]}]".into())
        ]),
        tool_spec!("Fit", summary: "Train a network (alias of NetTrain)", params: ["net","data","opts?"], tags: ["nn","train"], examples: [
            Value::String("Fit[net, data, <|Epochs->1, BatchSize->32|>]".into())
        ]),
        tool_spec!("LayerNorm", summary: "Layer normalization layer", params: ["opts?"], tags: ["nn","layer"]),
        tool_spec!("GroupNorm", summary: "Group normalization over channels (NumGroups)", params: ["opts?"], tags: ["nn","layer"], examples: [
            Value::String("Sequential[{Convolution2D[<|Output->32, KernelSize->3, Padding->1, InputChannels->3, Height->32, Width->32|>], GroupNorm[<|NumGroups->4, InputChannels->32, Height->32, Width->32|>], Relu[]}]".into())
        ]),
        tool_spec!("Residual", summary: "Residual wrapper with inner layers (adds skip)", params: ["layers"], tags: ["nn","layer"]),
        tool_spec!("Upsample2D", summary: "Upsample HxW (Nearest/Bilinear)", params: ["opts?"], tags: ["nn","layer"], examples: [
            Value::String("Sequential[{Upsample2D[<|Scale->2, Mode->\"Nearest\", InputChannels->1, Height->2, Width->2|>]}]".into()),
            Value::String("Sequential[{Upsample2D[<|Scale->2, Mode->\"Bilinear\", InputChannels->1, Height->2, Width->2|>]}]".into())
        ]),
        tool_spec!("ResidualBlock", summary: "Two convs + skip (MVP no norm)", params: ["opts?"], tags: ["nn","layer"], examples: [
            Value::String("Sequential[{Convolution2D[<|Output->8, KernelSize->3, Padding->1, InputChannels->3, Height->32, Width->32|>], ResidualBlock[<|Output->8, KernelSize->3, Padding->1, Activation->\"Relu\"|>]}]".into())
        ]),
        tool_spec!("Dropout", summary: "Dropout with rate p", params: ["p"], tags: ["nn","layer"]),
        tool_spec!("Flatten", summary: "Flatten to 1D", params: [], tags: ["nn","layer"]),
        tool_spec!("Embedding", summary: "Embedding lookup layer", params: ["vocab","dim"], tags: ["nn","layer"]),
        tool_spec!("Reshape", summary: "Reshape layer", params: ["shape"], tags: ["nn","layer"]),
        // Transformer layers
        tool_spec!("MultiHeadAttention", summary: "Self-attention with NumHeads (single-batch)", params: ["opts?"], tags: ["nn","layer","transformer"], examples: [
            Value::String("Sequential[{MultiHeadAttention[<|SeqLen->4, ModelDim->32, NumHeads->4|>]}]".into()),
            Value::String("Sequential[{MultiHeadAttention[<|SeqLen->4, ModelDim->32, NumHeads->4, Mask->\"Causal\"|>]}]".into())
        ]),
        tool_spec!("PositionalEncoding", summary: "Sinusoidal positional encoding (adds to input)", params: ["opts?"], tags: ["nn","layer","transformer"], examples: [
            Value::String("Sequential[{PositionalEncoding[<|SeqLen->4, ModelDim->32|>]}]".into())
        ]),
        tool_spec!("TransformerEncoder", summary: "Encoder block: MHA + FFN with residuals and layer norms (single-batch)", params: ["opts?"], tags: ["nn","layer","transformer"], examples: [
            Value::String("Sequential[{TransformerEncoder[<|SeqLen->8, ModelDim->64, NumHeads->8, HiddenDim->256|>]}]".into()),
            Value::String("Sequential[{TransformerEncoder[<|SeqLen->8, ModelDim->64, NumHeads->8, HiddenDim->256, Mask->\"Causal\"|>]}]".into())
        ]),
        tool_spec!("TransformerEncoderStack", summary: "Stack N encoder blocks (returns Sequential network)", params: ["opts?"], tags: ["nn","network","transformer"], examples: [
            Value::String("TransformerEncoderStack[<|Layers->2, SeqLen->8, ModelDim->64, NumHeads->8, HiddenDim->256|>]".into())
        ]),
        tool_spec!("TransformerDecoder", summary: "Decoder block: self-attn + cross-attn + FFN (single-batch)", params: ["opts?"], tags: ["nn","layer","transformer"], examples: [
            Value::String("Sequential[{TransformerDecoder[<|SeqLen->8, ModelDim->64, NumHeads->8, HiddenDim->256, Causal->True|>]}]".into()),
            Value::String("(* With fixed memory/context *) Sequential[{TransformerDecoder[<|SeqLen->4, ModelDim->32, NumHeads->4, HiddenDim->128, Memory->{{1,0,0,0},{0,1,0,0}}|>]}]".into())
        ]),
        tool_spec!("TransformerDecoderStack", summary: "Stack N decoder blocks (returns Sequential network)", params: ["opts?"], tags: ["nn","network","transformer"], examples: [
            Value::String("TransformerDecoderStack[<|Layers->2, SeqLen->8, ModelDim->64, NumHeads->8, HiddenDim->256, Causal->True|>]".into())
        ]),
        tool_spec!("TransformerEncoderDecoder", summary: "Convenience: builds Encoder/Decoder stacks", params: ["opts?"], tags: ["nn","network","transformer"], examples: [
            Value::String("TransformerEncoderDecoder[<|EncLayers->4, DecLayers->4, SeqLen->8, ModelDim->64, NumHeads->8, HiddenDim->256, Causal->True|>]".into())
        ]),
        tool_spec!("RMSNorm", summary: "RMS normalization (seq x dim)", params: ["opts?"], tags: ["nn","layer","transformer"], examples: [
            Value::String("Sequential[{RMSNorm[<|SeqLen->8, ModelDim->64|>]}]".into())
        ]),
        tool_spec!("FFN", summary: "Position-wise feed-forward (supports SwiGLU/GEGLU)", params: ["opts?"], tags: ["nn","layer","transformer"], examples: [
            Value::String("Sequential[{FFN[<|SeqLen->8, ModelDim->64, HiddenDim->256, Variant->\"SwiGLU\"|>]}]".into())
        ]),
        tool_spec!("CausalSelfAttention", summary: "Self-attention with causal mask", params: ["opts?"], tags: ["nn","layer","transformer"], examples: [
            Value::String("Sequential[{CausalSelfAttention[<|SeqLen->8, ModelDim->64, NumHeads->8|>]}]".into())
        ]),
        tool_spec!("CrossAttention", summary: "Cross-attention over Memory (seq x dim)", params: ["opts?"], tags: ["nn","layer","transformer"], examples: [
            Value::String("Sequential[{CrossAttention[<|SeqLen->8, ModelDim->64, NumHeads->8, Memory->{{...}}|>]}]".into())
        ]),
        tool_spec!("PositionalEmbedding", summary: "Learnable positional embeddings (adds to input)", params: ["opts?"], tags: ["nn","layer","transformer"], examples: [
            Value::String("Sequential[{PositionalEmbedding[<|SeqLen->8, ModelDim->64|>]}]".into())
        ]),
        tool_spec!("PatchEmbedding2D", summary: "2D to tokens via patch conv", params: ["opts?"], tags: ["nn","layer","vision","transformer"], examples: [
            Value::String("Sequential[{PatchEmbedding2D[<|PatchSize->{4,4}, ModelDim->64, InputChannels->3, Height->32, Width->32|>]}]".into())
        ]),
        tool_spec!("Transpose", summary: "Transpose tensor or construct layer (zero-arg)", params: ["x?","perm?"], tags: ["tensor","nn","layer"]),
        // Internal layer aliases (not usually documented): __TransposeLayer, __ConcatLayer, __AddLayer, __MulLayer
    ]);
}

// --------- helpers ---------

fn parse_opts(ev: &mut Evaluator, args: &[Value]) -> (Vec<Value>, HashMap<String, Value>) {
    if args.is_empty() {
        return (vec![], HashMap::new());
    }
    let mut pos: Vec<Value> = Vec::new();
    let mut opts: HashMap<String, Value> = HashMap::new();
    for (i, a) in args.iter().enumerate() {
        if i == args.len() - 1 {
            let evd = ev.eval(a.clone());
            if let Value::Assoc(m) = evd {
                opts = m;
                continue;
            }
        }
        pos.push(a.clone());
    }
    (pos, opts)
}

fn get_option<'a>(opts: &'a HashMap<String, Value>, key: &str) -> Option<&'a Value> {
    if let Some(v) = opts.get(key) { return Some(v); }
    let mut chars = key.chars();
    if let Some(first) = chars.next() {
        let mut alt = String::new();
        alt.push(first.to_ascii_lowercase());
        alt.push_str(chars.as_str());
        if let Some(v) = opts.get(&alt) { return Some(v); }
    }
    None
}

fn layer_spec(kind: &str, params: HashMap<String, Value>) -> Value {
    Value::assoc(vec![
        ("__type", Value::String("Layer".into())),
        ("LayerType", Value::String(kind.into())),
        ("Params", Value::Assoc(params)),
    ])
}

fn to_vec_f64(v: &Value) -> Option<Vec<f64>> {
    match v {
        Value::Real(x) => Some(vec![*x]),
        Value::Integer(n) => Some(vec![*n as f64]),
        Value::List(xs) => {
            let mut out = Vec::with_capacity(xs.len());
            for x in xs {
                match x {
                    Value::Real(r) => out.push(*r),
                    Value::Integer(n) => out.push(*n as f64),
                    _ => return None,
                }
            }
            Some(out)
        }
        _ => None,
    }
}

fn flatten_to_vec_f64(v: &Value, out: &mut Vec<f64>) -> bool {
    match v {
        Value::Real(x) => {
            out.push(*x);
            true
        }
        Value::Integer(n) => {
            out.push(*n as f64);
            true
        }
        Value::List(xs) => {
            for x in xs {
                if !flatten_to_vec_f64(x, out) {
                    return false;
                }
            }
            true
        }
        _ => false,
    }
}

fn from_vec_like(input_like: &Value, xs: &[f64]) -> Value {
    let is_scalar = matches!(input_like, Value::Real(_) | Value::Integer(_));
    if is_scalar && xs.len() == 1 {
        let x = xs[0];
        if x.fract() == 0.0 {
            Value::Integer(x as i64)
        } else {
            Value::Real(x)
        }
    } else {
        Value::List(xs.iter().map(|x| Value::Real(*x)).collect())
    }
}

fn lcg(seed: &mut u64) -> f64 {
    // Deterministic pseudo-random in [-0.1, 0.1]
    *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    let v = ((*seed >> 33) as f64) / ((1u64 << 31) as f64);
    (v - 1.0) * 0.1
}

fn assoc_get<'a>(m: &'a HashMap<String, Value>, k: &str) -> Option<&'a Value> {
    if let Some(v) = m.get(k) { return Some(v); }
    // Try lowerCamelCase alias (e.g., InputChannels -> inputChannels)
    let mut chars = k.chars();
    if let Some(first) = chars.next() {
        let mut alt = String::new();
        alt.push(first.to_ascii_lowercase());
        alt.push_str(chars.as_str());
        if let Some(v) = m.get(&alt) { return Some(v); }
    }
    None
}
fn as_i64(v: &Value) -> Option<i64> {
    match v {
        Value::Integer(n) => Some(*n),
        Value::Real(x) => Some(*x as i64),
        _ => None,
    }
}
fn as_bool(v: &Value) -> Option<bool> {
    match v {
        Value::Boolean(b) => Some(*b),
        _ => None,
    }
}

fn as_f64(v: &Value) -> Option<f64> {
    match v {
        Value::Real(x) => Some(*x),
        Value::Integer(n) => Some(*n as f64),
        _ => None,
    }
}

fn ensure_linear_params(id: i64, layer_idx: usize, layer: &Value, in_dim: usize) -> (Value, usize) {
    // Returns (updated_layer, out_dim)
    if let Value::Assoc(m) = layer {
        if let Some(Value::Assoc(params)) = m.get("Params") {
            let out_dim = assoc_get(params, "Output")
                .and_then(|v| match v {
                    Value::Integer(n) => Some((*n).max(0) as usize),
                    Value::Real(x) => Some((*x as i64).max(0) as usize),
                    _ => None,
                })
                .unwrap_or(in_dim.max(1));
            let has_w = assoc_get(params, "W").is_some();
            let has_b = assoc_get(params, "b").is_some();
            if has_w && has_b {
                return (layer.clone(), out_dim);
            }
            // build new params with W,b
            let mut p2 = params.clone();
            let mut seed = (id as u64) ^ ((layer_idx as u64) << 13) ^ 0x9E3779B97F4A7C15;
            let mut rows: Vec<Value> = Vec::with_capacity(out_dim);
            for _ in 0..out_dim {
                let mut row: Vec<Value> = Vec::with_capacity(in_dim.max(1));
                for _ in 0..in_dim.max(1) {
                    row.push(Value::Real(lcg(&mut seed)));
                }
                rows.push(Value::List(row));
            }
            let mut b: Vec<Value> = Vec::with_capacity(out_dim);
            for _ in 0..out_dim {
                b.push(Value::Real(0.0));
            }
            p2.insert("W".into(), Value::List(rows));
            p2.insert("b".into(), Value::List(b));
            let mut m2 = m.clone();
            m2.insert("Params".into(), Value::Assoc(p2));
            return (Value::Assoc(m2), out_dim);
        }
    }
    (layer.clone(), in_dim)
}

fn apply_linear(params: &HashMap<String, Value>, x: &[f64]) -> Option<Vec<f64>> {
    let w = assoc_get(params, "W")?;
    let b = assoc_get(params, "b");
    let (w_rows, in_dim) = match w {
        Value::List(rows) => {
            let r = rows.len();
            let c = rows
                .get(0)
                .and_then(|v| if let Value::List(xs) = v { Some(xs.len()) } else { None })
                .unwrap_or(0);
            (r, c)
        }
        _ => return None,
    };
    if in_dim == 0 {
        return None;
    }
    let mut y = vec![0.0f64; w_rows];
    match w {
        Value::List(rows) => {
            for (i, row) in rows.iter().enumerate() {
                let mut acc = 0.0f64;
                if let Value::List(xs) = row {
                    for (j, v) in xs.iter().enumerate().take(x.len()) {
                        let wij = match v {
                            Value::Real(r) => *r,
                            Value::Integer(n) => *n as f64,
                            _ => 0.0,
                        };
                        acc += wij * x.get(j).cloned().unwrap_or(0.0);
                    }
                }
                y[i] = acc;
            }
        }
        _ => {}
    }
    let use_bias = assoc_get(params, "Bias").and_then(|v| as_bool(v)).unwrap_or(true);
    if use_bias {
        if let Some(Value::List(bs)) = b {
            for (i, v) in bs.iter().enumerate().take(y.len()) {
                let bi = match v {
                    Value::Real(r) => *r,
                    Value::Integer(n) => *n as f64,
                    _ => 0.0,
                };
                y[i] += bi;
            }
        }
    }
    Some(y)
}

fn apply_activation(kind: &str, x: &mut [f64]) {
    match kind.to_lowercase().as_str() {
        "relu" => {
            for v in x.iter_mut() {
                if *v < 0.0 {
                    *v = 0.0;
                }
            }
        }
        "sigmoid" => {
            for v in x.iter_mut() {
                *v = 1.0 / (1.0 + (-*v).exp());
            }
        }
        "tanh" => {
            for v in x.iter_mut() {
                *v = v.tanh();
            }
        }
        "gelu" => {
            for v in x.iter_mut() {
                let u = *v;
                // tanh approximation
                let k = (0.79788456 * (u + 0.044715 * u * u * u)).tanh();
                *v = 0.5 * u * (1.0 + k);
            }
        }
        "softmax" => {
            // in-place: compute stable softmax
            let maxv = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let mut sum = 0.0;
            for v in x.iter_mut() {
                *v = (*v - maxv).exp();
                sum += *v;
            }
            if sum != 0.0 {
                for v in x.iter_mut() {
                    *v /= sum;
                }
            }
        }
        _ => {}
    }
}

fn ensure_conv1d_params(
    id: i64,
    layer_idx: usize,
    layer: &Value,
    _in_dim: usize,
) -> (Value, usize, usize) {
    // returns (updated_layer, out_channels, kernel_size)
    if let Value::Assoc(m) = layer {
        if let Some(Value::Assoc(params)) = m.get("Params") {
            let out_ch =
                assoc_get(params, "Output").and_then(|v| as_i64(v)).unwrap_or(1).max(1) as usize;
            let k = assoc_get(params, "KernelSize").and_then(|v| as_i64(v)).unwrap_or(3).max(1)
                as usize;
            let has_w = assoc_get(params, "W").is_some();
            let has_b = assoc_get(params, "b").is_some();
            if has_w && has_b {
                return (layer.clone(), out_ch, k);
            }
            let mut p2 = params.clone();
            let mut seed = (id as u64) ^ ((layer_idx as u64) << 9) ^ 0x517CC1B7;
            // one kernel per out channel
            let mut kernels: Vec<Value> = Vec::with_capacity(out_ch);
            for _ in 0..out_ch {
                let mut ker: Vec<Value> = Vec::with_capacity(k);
                for _ in 0..k {
                    ker.push(Value::Real(lcg(&mut seed)));
                }
                kernels.push(Value::List(ker));
            }
            let mut b: Vec<Value> = Vec::with_capacity(out_ch);
            for _ in 0..out_ch {
                b.push(Value::Real(0.0));
            }
            p2.insert("W".into(), Value::List(kernels));
            p2.insert("b".into(), Value::List(b));
            let mut m2 = m.clone();
            m2.insert("Params".into(), Value::Assoc(p2));
            return (Value::Assoc(m2), out_ch, k);
        }
    }
    (layer.clone(), 1, 3)
}

fn apply_conv1d(params: &HashMap<String, Value>, x: &[f64]) -> Option<Vec<f64>> {
    let w = assoc_get(params, "W")?;
    let b = assoc_get(params, "b")?;
    let out_ch = match w {
        Value::List(rows) => rows.len(),
        _ => 0,
    };
    let k = match w {
        Value::List(rows) => rows
            .get(0)
            .and_then(|v| if let Value::List(xs) = v { Some(xs.len()) } else { None })
            .unwrap_or(0),
        _ => 0,
    };
    let stride = assoc_get(params, "Stride").and_then(|v| as_i64(v)).unwrap_or(1).max(1) as usize;
    let pad = assoc_get(params, "Padding").and_then(|v| as_i64(v)).unwrap_or(0).max(0) as usize;
    let n = x.len();
    let out_len = if n + 2 * pad >= k { ((n + 2 * pad - k) / stride) + 1 } else { 0 };
    let mut y = vec![0.0f64; out_len * out_ch];
    if out_len == 0 {
        return Some(y);
    }
    for oc in 0..out_ch {
        // bias
        let b_oc = if let Value::List(bs) = b {
            bs.get(oc)
                .and_then(|v| match v {
                    Value::Real(r) => Some(*r),
                    Value::Integer(n) => Some(*n as f64),
                    _ => None,
                })
                .unwrap_or(0.0)
        } else {
            0.0
        };
        // kernel
        let ker: Vec<f64> = match w {
            Value::List(rows) => {
                if let Some(Value::List(xs)) = rows.get(oc) {
                    xs.iter()
                        .map(|v| match v {
                            Value::Real(r) => *r,
                            Value::Integer(n) => *n as f64,
                            _ => 0.0,
                        })
                        .collect()
                } else {
                    vec![0.0; k]
                }
            }
            _ => vec![0.0; k],
        };
        for i in 0..out_len {
            let mut acc = 0.0f64;
            for j in 0..k {
                let xi = i * stride + j;
                let xv = if xi < pad {
                    0.0
                } else {
                    let ii = xi - pad;
                    if ii < n {
                        x[ii]
                    } else {
                        0.0
                    }
                };
                acc += ker[j] * xv;
            }
            y[oc * out_len + i] = acc + b_oc;
        }
    }
    Some(y)
}

fn apply_pool1d(params: &HashMap<String, Value>, x: &[f64]) -> Option<Vec<f64>> {
    let size = assoc_get(params, "Size").and_then(|v| as_i64(v)).unwrap_or(2).max(1) as usize;
    let stride =
        assoc_get(params, "Stride").and_then(|v| as_i64(v)).unwrap_or(size as i64).max(1) as usize;
    let kind = assoc_get(params, "PoolType")
        .and_then(|v| match v {
            Value::String(s) | Value::Symbol(s) => Some(s.clone()),
            _ => None,
        })
        .unwrap_or("Max".into());
    let n = x.len();
    if n < size {
        return Some(vec![]);
    }
    let out_len = ((n - size) / stride) + 1;
    let mut y = vec![0.0f64; out_len];
    for i in 0..out_len {
        let start = i * stride;
        let end = start + size;
        let window = &x[start..end];
        y[i] = if kind.eq_ignore_ascii_case("avg") || kind.eq_ignore_ascii_case("average") {
            window.iter().sum::<f64>() / (size as f64)
        } else {
            window.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
        };
    }
    Some(y)
}

fn apply_batchnorm(params: &HashMap<String, Value>, x: &mut [f64]) {
    let eps = assoc_get(params, "Epsilon")
        .and_then(|v| match v { Value::Real(r) => Some(*r), Value::Integer(n) => Some(*n as f64), _ => None })
        .unwrap_or(1e-5);
    let gamma = assoc_get(params, "Gamma");
    let beta = assoc_get(params, "Beta");
    let cin = assoc_get(params, "InputChannels").and_then(|v| as_i64(v)).unwrap_or(0).max(0) as usize;
    let h = assoc_get(params, "Height").and_then(|v| as_i64(v)).unwrap_or(0).max(0) as usize;
    let w = assoc_get(params, "Width").and_then(|v| as_i64(v)).unwrap_or(0).max(0) as usize;
    if cin > 0 && h > 0 && w > 0 && x.len() == cin * h * w {
        // Per-channel normalize across spatial dims
        for c in 0..cin {
            let base = c * h * w;
            let mut mean = 0.0;
            for i in 0..(h * w) { mean += x[base + i]; }
            mean /= (h * w) as f64;
            let mut var = 0.0;
            for i in 0..(h * w) { let d = x[base + i] - mean; var += d * d; }
            var /= (h * w) as f64;
            let inv = 1.0 / (var + eps).sqrt();
            let g = match gamma {
                Some(Value::List(gs)) => gs.get(c).and_then(|v| as_f64(v)).unwrap_or(1.0),
                Some(Value::Real(r)) => *r,
                Some(Value::Integer(n)) => *n as f64,
                _ => 1.0,
            };
            let be = match beta {
                Some(Value::List(bs)) => bs.get(c).and_then(|v| as_f64(v)).unwrap_or(0.0),
                Some(Value::Real(r)) => *r,
                Some(Value::Integer(n)) => *n as f64,
                _ => 0.0,
            };
            for i in 0..(h * w) {
                let y = (x[base + i] - mean) * inv;
                x[base + i] = y * g + be;
            }
        }
        return;
    }
    // 1D fallback
    let mean = if x.is_empty() { 0.0 } else { x.iter().sum::<f64>() / (x.len() as f64) };
    let var = if x.is_empty() { 1.0 } else { let m = mean; x.iter().map(|v| { let d = *v - m; d * d }).sum::<f64>() / (x.len() as f64) };
    let inv = 1.0 / (var + eps).sqrt();
    for i in 0..x.len() {
        let y = (x[i] - mean) * inv;
        let g = match gamma {
            Some(Value::List(gs)) => gs.get(i).and_then(|v| as_f64(v)).unwrap_or(1.0),
            Some(Value::Real(r)) => *r,
            Some(Value::Integer(n)) => *n as f64,
            _ => 1.0,
        };
        let be = match beta {
            Some(Value::List(bs)) => bs.get(i).and_then(|v| as_f64(v)).unwrap_or(0.0),
            Some(Value::Real(r)) => *r,
            Some(Value::Integer(n)) => *n as f64,
            _ => 0.0,
        };
        x[i] = y * g + be;
    }
}

fn apply_layernorm(params: &HashMap<String, Value>, x: &mut [f64]) {
    let eps = assoc_get(params, "Epsilon")
        .and_then(|v| match v { Value::Real(r) => Some(*r), Value::Integer(n) => Some(*n as f64), _ => None })
        .unwrap_or(1e-5);
    let gamma = assoc_get(params, "Gamma");
    let beta = assoc_get(params, "Beta");
    let cin = assoc_get(params, "InputChannels").and_then(|v| as_i64(v)).unwrap_or(0).max(0) as usize;
    let h = assoc_get(params, "Height").and_then(|v| as_i64(v)).unwrap_or(0).max(0) as usize;
    let w = assoc_get(params, "Width").and_then(|v| as_i64(v)).unwrap_or(0).max(0) as usize;
    if cin > 0 && h > 0 && w > 0 && x.len() == cin * h * w {
        // Normalize across channels per spatial location
        for i in 0..(h * w) {
            let mut mean = 0.0;
            for c in 0..cin { mean += x[c * h * w + i]; }
            mean /= cin as f64;
            let mut var = 0.0;
            for c in 0..cin { let d = x[c * h * w + i] - mean; var += d * d; }
            var /= cin as f64;
            let inv = 1.0 / (var + eps).sqrt();
            for c in 0..cin {
                let idx = c * h * w + i;
                let y = (x[idx] - mean) * inv;
                let g = match gamma {
                    Some(Value::List(gs)) => gs.get(c).and_then(|v| as_f64(v)).unwrap_or(1.0),
                    Some(Value::Real(r)) => *r,
                    Some(Value::Integer(n)) => *n as f64,
                    _ => 1.0,
                };
                let be = match beta {
                    Some(Value::List(bs)) => bs.get(c).and_then(|v| as_f64(v)).unwrap_or(0.0),
                    Some(Value::Real(r)) => *r,
                    Some(Value::Integer(n)) => *n as f64,
                    _ => 0.0,
                };
                x[idx] = y * g + be;
            }
        }
        return;
    }
    // 1D fallback: reuse BN path
    apply_batchnorm(params, x)
}

fn parse_hw(param: Option<&Value>) -> Option<(usize, usize)> {
    match param {
        Some(Value::Integer(k)) => { let n = (*k).max(1) as usize; Some((n, n)) }
        Some(Value::Real(x)) => { let n = (*x as i64).max(1) as usize; Some((n, n)) }
        Some(Value::List(xs)) => {
            if xs.len() == 2 {
                let a = as_i64(&xs[0]).unwrap_or(1).max(1) as usize;
                let b = as_i64(&xs[1]).unwrap_or(1).max(1) as usize;
                Some((a, b))
            } else { None }
        }
        _ => None,
    }
}

fn apply_conv2d(params: &HashMap<String, Value>, x: &[f64]) -> Option<Vec<f64>> {
    // Expect InputChannels, Height, Width to reshape x; otherwise bail
    let cin = assoc_get(params, "InputChannels").and_then(|v| as_i64(v)).unwrap_or(0).max(0) as usize;
    let h = assoc_get(params, "Height").and_then(|v| as_i64(v)).unwrap_or(0).max(0) as usize;
    let w = assoc_get(params, "Width").and_then(|v| as_i64(v)).unwrap_or(0).max(0) as usize;
    if cin == 0 || h == 0 || w == 0 { return None; }
    if x.len() != cin * h * w { return None; }
    let (kh, kw) = parse_hw(assoc_get(params, "KernelSize")).unwrap_or((3,3));
    let (sh, sw) = parse_hw(assoc_get(params, "Stride")).unwrap_or((1,1));
    let (ph, pw) = parse_hw(assoc_get(params, "Padding")).unwrap_or((0,0));
    let cout = assoc_get(params, "Output").and_then(|v| as_i64(v)).unwrap_or(1).max(1) as usize;
    // Weights: W[out][in][kh][kw]; Bias: b[out]
    let wv = assoc_get(params, "W");
    let bv = assoc_get(params, "b");
    // reshape x into [cin][h][w]
    let mut input = vec![0.0f64; cin*h*w];
    input.copy_from_slice(x);
    let oh = if h + 2*ph >= kh { ((h + 2*ph - kh) / sh) + 1 } else { 0 };
    let ow = if w + 2*pw >= kw { ((w + 2*pw - kw) / sw) + 1 } else { 0 };
    let mut out = vec![0.0f64; cout * oh * ow];
    for oc in 0..cout {
        let b = match bv {
            Some(Value::List(bs)) => bs.get(oc).and_then(|v| as_f64(v)).unwrap_or(0.0),
            _ => 0.0,
        };
        for oh_i in 0..oh {
            for ow_i in 0..ow {
                let mut acc = 0.0f64;
                for ic in 0..cin {
                    // get kernel for (oc,ic)
                    let ker: Vec<f64> = if let Some(Value::List(oc_list)) = wv {
                        if let Some(Value::List(ic_list)) = oc_list.get(oc) {
                            if let Some(Value::List(rows)) = ic_list.get(ic) {
                                let mut k = Vec::with_capacity(kh * kw);
                                for r in 0..kh {
                                    if let Some(Value::List(row)) = rows.get(r) {
                                        for c in 0..kw {
                                            k.push(as_f64(row.get(c).unwrap_or(&Value::Integer(0))).unwrap_or(0.0));
                                        }
                                    } else {
                                        for _ in 0..kw {
                                            k.push(0.0);
                                        }
                                    }
                                }
                                k
                            } else { vec![0.0; kh * kw] }
                        } else { vec![0.0; kh * kw] }
                    } else { vec![0.0; kh * kw] };
                    for r in 0..kh {
                        for c in 0..kw {
                            let in_r = (oh_i * sh + r) as i64 - ph as i64;
                            let in_c = (ow_i * sw + c) as i64 - pw as i64;
                            if in_r >= 0 && in_c >= 0 && (in_r as usize) < h && (in_c as usize) < w {
                                let idx = ic * h * w + (in_r as usize) * w + (in_c as usize);
                                let kidx = r * kw + c;
                                acc += input[idx] * ker.get(kidx).cloned().unwrap_or(0.0);
                            }
                        }
                    }
                }
                out[oc * oh * ow + oh_i * ow + ow_i] = acc + b;
            }
        }
    }
    Some(out)
}

fn apply_pool2d(params: &HashMap<String, Value>, x: &[f64]) -> Option<Vec<f64>> {
    let cin = assoc_get(params, "InputChannels").and_then(|v| as_i64(v)).unwrap_or(0).max(0) as usize;
    let h = assoc_get(params, "Height").and_then(|v| as_i64(v)).unwrap_or(0).max(0) as usize;
    let w = assoc_get(params, "Width").and_then(|v| as_i64(v)).unwrap_or(0).max(0) as usize;
    if cin == 0 || h == 0 || w == 0 { return None; }
    if x.len() != cin * h * w { return None; }
    let (kh, kw) = parse_hw(assoc_get(params, "KernelSize")).unwrap_or((2,2));
    let (sh, sw) = parse_hw(assoc_get(params, "Stride")).unwrap_or((kh,kw));
    let kind = assoc_get(params, "PoolType").and_then(|v| match v { Value::String(s) | Value::Symbol(s) => Some(s.clone()), _ => None }).unwrap_or("Max".into());
    let oh = if h >= kh { ((h - kh) / sh) + 1 } else { 0 };
    let ow = if w >= kw { ((w - kw) / sw) + 1 } else { 0 };
    let mut out = vec![0.0f64; cin * oh * ow];
    for ic in 0..cin {
        for oh_i in 0..oh { for ow_i in 0..ow {
            let mut acc = if kind.eq_ignore_ascii_case("avg") { 0.0 } else { f64::NEG_INFINITY };
            for r in 0..kh { for c in 0..kw {
                let in_r = oh_i*sh + r; let in_c = ow_i*sw + c;
                let idx = ic*h*w + in_r*w + in_c;
                let v = x[idx];
                if kind.eq_ignore_ascii_case("avg") { acc += v; } else { if v > acc { acc = v; } }
            }}
            let out_idx = ic*oh*ow + oh_i*ow + ow_i;
            out[out_idx] = if kind.eq_ignore_ascii_case("avg") { acc / ((kh*kw) as f64) } else { acc };
        }}
    }
    Some(out)
}

fn apply_depthwise_conv2d(params: &HashMap<String, Value>, x: &[f64]) -> Option<Vec<f64>> {
    let cin = assoc_get(params, "InputChannels").and_then(|v| as_i64(v)).unwrap_or(0).max(0) as usize;
    let h = assoc_get(params, "Height").and_then(|v| as_i64(v)).unwrap_or(0).max(0) as usize;
    let w = assoc_get(params, "Width").and_then(|v| as_i64(v)).unwrap_or(0).max(0) as usize;
    if cin == 0 || h == 0 || w == 0 || x.len() != cin*h*w { return None; }
    let (kh, kw) = parse_hw(assoc_get(params, "KernelSize")).unwrap_or((3,3));
    let (sh, sw) = parse_hw(assoc_get(params, "Stride")).unwrap_or((1,1));
    let (ph, pw) = parse_hw(assoc_get(params, "Padding")).unwrap_or((0,0));
    // W[cin][kh][kw], b[cin]
    let wv = assoc_get(params, "W");
    let bv = assoc_get(params, "b");
    let oh = if h + 2*ph >= kh { ((h + 2*ph - kh) / sh) + 1 } else { 0 };
    let ow = if w + 2*pw >= kw { ((w + 2*pw - kw) / sw) + 1 } else { 0 };
    let mut out = vec![0.0f64; cin * oh * ow];
    for c in 0..cin {
        let b = match bv { Some(Value::List(bs)) => bs.get(c).and_then(|v| as_f64(v)).unwrap_or(0.0), _ => 0.0 };
        // kernel for channel c
        let ker: Vec<f64> = match wv {
            Some(Value::List(ch_list)) => ch_list.get(c).and_then(|v| match v {
                Value::List(rows) => {
                    let mut k = Vec::with_capacity(kh*kw);
                    for r in 0..kh {
                        if let Some(Value::List(row)) = rows.get(r) {
                            for c2 in 0..kw { k.push(as_f64(row.get(c2).unwrap_or(&Value::Integer(0))).unwrap_or(0.0)); }
                        } else { for _ in 0..kw { k.push(0.0); } }
                    }
                    Some(k)
                }
                _ => None,
            }),
            _ => None,
        }.unwrap_or(vec![0.0; kh*kw]);
        for oh_i in 0..oh { for ow_i in 0..ow {
            let mut acc = 0.0;
            for r in 0..kh { for c2 in 0..kw {
                let in_r = (oh_i*sh + r) as i64 - ph as i64;
                let in_c = (ow_i*sw + c2) as i64 - pw as i64;
                if in_r >= 0 && in_c >= 0 && (in_r as usize) < h && (in_c as usize) < w {
                    let idx = c*h*w + (in_r as usize)*w + (in_c as usize);
                    acc += x[idx] * ker[r*kw + c2];
                }
            }}
            out[c*oh*ow + oh_i*ow + ow_i] = acc + b;
        }}
    }
    Some(out)
}

fn apply_conv_transpose2d(params: &HashMap<String, Value>, x: &[f64]) -> Option<Vec<f64>> {
    let cin = assoc_get(params, "InputChannels").and_then(|v| as_i64(v)).unwrap_or(0).max(0) as usize;
    let h = assoc_get(params, "Height").and_then(|v| as_i64(v)).unwrap_or(0).max(0) as usize;
    let w = assoc_get(params, "Width").and_then(|v| as_i64(v)).unwrap_or(0).max(0) as usize;
    if cin == 0 || h == 0 || w == 0 || x.len() != cin*h*w { return None; }
    let (kh, kw) = parse_hw(assoc_get(params, "KernelSize")).unwrap_or((3,3));
    let (sh, sw) = parse_hw(assoc_get(params, "Stride")).unwrap_or((1,1));
    let (ph, pw) = parse_hw(assoc_get(params, "Padding")).unwrap_or((0,0));
    let cout = assoc_get(params, "Output").and_then(|v| as_i64(v)).unwrap_or(1).max(1) as usize;
    // Output dims: (h-1)*sh - 2*ph + kh, (w-1)*sw - 2*pw + kw
    let oh = (h - 1) * sh + kh - 2*ph;
    let ow = (w - 1) * sw + kw - 2*pw;
    if oh == 0 || ow == 0 { return Some(vec![0.0; cout * 0]); }
    let mut out = vec![0.0f64; cout * oh * ow];
    // W[out][in][kh][kw]
    let wv = assoc_get(params, "W");
    let bv = assoc_get(params, "b");
    for ic in 0..cin { for ih in 0..h { for iw in 0..w {
        let xval = x[ic*h*w + ih*w + iw];
        for oc in 0..cout {
            let ker: Vec<f64> = match wv {
                Some(Value::List(oc_list)) => oc_list.get(oc).and_then(|v| match v { Value::List(ic_list) => ic_list.get(ic).cloned(), _ => None }).and_then(|v| match v {
                    Value::List(rows) => {
                        let mut k = Vec::with_capacity(kh*kw);
                        for r in 0..kh { if let Some(Value::List(row)) = rows.get(r) { for c in 0..kw { k.push(as_f64(row.get(c).unwrap_or(&Value::Integer(0))).unwrap_or(0.0)); } } }
                        Some(k)
                    }
                    _ => None,
                }),
                _ => None,
            }.unwrap_or(vec![0.0; kh*kw]);
            for r in 0..kh { for c in 0..kw {
                let oh_i = ih*sh + r - ph; let ow_i = iw*sw + c - pw;
                if oh_i < oh && ow_i < ow {
                    let idx = oc*oh*ow + oh_i*ow + ow_i;
                    out[idx] += xval * ker[r*kw + c];
                }
            }}
        }
    }}}
    // add bias
    if let Some(Value::List(bs)) = bv { for oc in 0..cout { let b = bs.get(oc).and_then(|v| as_f64(v)).unwrap_or(0.0); for i in 0..(oh*ow) { out[oc*oh*ow + i] += b; } } }
    Some(out)
}

fn apply_separable_conv2d(params: &HashMap<String, Value>, x: &[f64]) -> Option<Vec<f64>> {
    // Depthwise then pointwise (1x1) mixing
    let dw_out = apply_depthwise_conv2d(params, x)?; // shape (cin, oh, ow) flattened
    let cin = assoc_get(params, "InputChannels").and_then(|v| as_i64(v)).unwrap_or(0).max(0) as usize;
    let (kh, kw) = parse_hw(assoc_get(params, "KernelSize")).unwrap_or((3,3));
    let h = assoc_get(params, "Height").and_then(|v| as_i64(v)).unwrap_or(0).max(0) as usize;
    let w = assoc_get(params, "Width").and_then(|v| as_i64(v)).unwrap_or(0).max(0) as usize;
    let (sh, sw) = parse_hw(assoc_get(params, "Stride")).unwrap_or((1,1));
    let (ph, pw) = parse_hw(assoc_get(params, "Padding")).unwrap_or((0,0));
    let oh = if h + 2*ph >= kh { ((h + 2*ph - kh) / sh) + 1 } else { 0 };
    let ow = if w + 2*pw >= kw { ((w + 2*pw - kw) / sw) + 1 } else { 0 };
    let cout = assoc_get(params, "Output").and_then(|v| as_i64(v)).unwrap_or(cin as i64).max(1) as usize;
    // WPW[cout][cin], b[cout]
    let wpw = assoc_get(params, "WPW");
    let b = assoc_get(params, "b");
    let mut out = vec![0.0f64; cout * oh * ow];
    for oc in 0..cout {
        let bias = match b { Some(Value::List(bs)) => bs.get(oc).and_then(|v| as_f64(v)).unwrap_or(0.0), _ => 0.0 };
        for i in 0..(oh*ow) {
            let mut acc = 0.0;
            for ic in 0..cin {
                let w_ic = match wpw { Some(Value::List(ws)) => ws.get(oc).and_then(|v| match v { Value::List(row) => row.get(ic).and_then(|vv| as_f64(vv)), _ => None }).unwrap_or(0.0), _ => 0.0 };
                acc += dw_out[ic*oh*ow + i] * w_ic;
            }
            out[oc*oh*ow + i] = acc + bias;
        }
    }
    Some(out)
}

fn build_shape_nested(xs: &[f64], shape: &[usize]) -> Value {
    if shape.is_empty() {
        return Value::List(vec![]);
    }
    fn rec(vs: &[f64], shape: &[usize], idx: &mut usize) -> Value {
        if shape.len() == 1 {
            let n = shape[0];
            let mut out: Vec<Value> = Vec::with_capacity(n);
            for _ in 0..n {
                let x = vs.get(*idx).cloned().unwrap_or(0.0);
                *idx += 1;
                out.push(Value::Real(x));
            }
            Value::List(out)
        } else {
            let n = shape[0];
            let mut out: Vec<Value> = Vec::with_capacity(n);
            for _ in 0..n {
                out.push(rec(vs, &shape[1..], idx));
            }
            Value::List(out)
        }
    }
    let mut i = 0usize;
    rec(xs, shape, &mut i)
}

fn infer_chw(x: &Value) -> Option<(usize, usize, usize)> {
    match x {
        Value::List(chs) if !chs.is_empty() => {
            let cin = chs.len();
            if let Value::List(rows) = &chs[0] {
                if !rows.is_empty() {
                    let h = rows.len();
                    if let Value::List(cols) = &rows[0] {
                        if !cols.is_empty() {
                            let w = cols.len();
                            return Some((cin, h, w));
                        }
                    }
                }
            }
            None
        }
        _ => None,
    }
}

fn infer_seqdim(x: &Value) -> Option<(usize, usize)> {
    match x {
        Value::List(rows) if !rows.is_empty() => {
            if let Value::List(cols) = &rows[0] {
                Some((rows.len(), cols.len()))
            } else { None }
        }
        _ => None,
    }
}

// --------- constructors ---------

fn net_chain(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    let (pos, opts) = parse_opts(ev, &args);
    let layers = if pos.is_empty() {
        vec![]
    } else {
        match ev.eval(pos[0].clone()) {
            Value::List(vs) => vs,
            v => vec![v],
        }
    };
    let id = next_id();
    reg().lock().unwrap().insert(
        id,
        NetState {
            kind: "Chain".into(),
            layers,
            graph: None,
            opts,
            encoder: None,
            decoder: None,
            initialized: false,
            trained_epochs: 0,
            method: "Adam".into(),
            batch_size: 32,
        },
    );
    net_handle(id)
}

fn net_graph(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // NetGraph[assoc, edges, opts]
    let (_pos, opts) = parse_opts(ev, &args);
    let id = next_id();
    let graph = if args.len() >= 2 { Some(ev.eval(args[1].clone())) } else { None };
    let layers = if args.len() >= 1 {
        match ev.eval(args[0].clone()) {
            Value::Assoc(m) => m
                .into_iter()
                .map(|(k, v)| Value::expr(Value::symbol("Rule"), vec![Value::String(k), v]))
                .collect(),
            _ => vec![],
        }
    } else {
        vec![]
    };
    reg().lock().unwrap().insert(
        id,
        NetState {
            kind: "Graph".into(),
            layers,
            graph,
            opts,
            encoder: None,
            decoder: None,
            initialized: false,
            trained_epochs: 0,
            method: "Adam".into(),
            batch_size: 32,
        },
    );
    net_handle(id)
}

fn net_initialize(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() {
        return Value::Expr { head: Box::new(Value::Symbol("NetInitialize".into())), args };
    }
    let net = args[0].clone();
    let opts = if args.len() >= 2 { args[1].clone() } else { Value::Assoc(HashMap::new()) };
    if let Some(id) = get_net_id(&net) {
        if let Some(mut st) = reg().lock().unwrap().remove(&id) {
            // Parse Initializer and optional InputDim
            let (init_kind, input_dim) = match opts {
                Value::Assoc(m) => {
                    let init_kind = m.get("Initializer").and_then(|v| match v {
                        Value::Assoc(mm) => mm.get("Type").and_then(|vv| match vv { Value::String(s) | Value::Symbol(s) => Some(s.clone()), _ => None }),
                        Value::String(s) | Value::Symbol(s) => Some(s.clone()),
                        _ => None,
                    }).unwrap_or("Xavier".into());
                    let input_dim = m.get("InputDim").and_then(|v| match v { Value::Integer(n) => Some((*n).max(1) as usize), Value::Real(x) => Some((*x as i64).max(1) as usize), _ => None });
                    (init_kind, input_dim)
                }
                _ => ("Xavier".into(), None),
            };
            // Initialize parameters layer-by-layer; track current width when possible
            let mut curr_dim = input_dim.unwrap_or(0usize);
            let mut new_layers: Vec<Value> = Vec::with_capacity(st.layers.len());
            for (idx, layer) in st.layers.iter().enumerate() {
                let mut current = layer.clone();
                if let Value::Assoc(m) = layer {
                    let ltype = m.get("LayerType").and_then(|v| match v { Value::String(s) | Value::Symbol(s) => Some(s.clone()), _ => None }).unwrap_or_default();
                    let params_in = m.get("Params").and_then(|v| if let Value::Assoc(mm) = v { Some(mm.clone()) } else { None }).unwrap_or_default();
                    let mut params = params_in.clone();
                    match ltype.as_str() {
                        "Linear" => {
                            // Requires Output and input width
                            let out = params.get("Output").and_then(|v| as_i64(v)).unwrap_or(0).max(0) as usize;
                            let inw = if curr_dim > 0 { curr_dim } else { out.max(1) };
                            let range = match init_kind.to_lowercase().as_str() { "he" => (6.0 / (inw as f64)).sqrt(), _ => (6.0 / ((inw + out).max(1) as f64)).sqrt() };
                            let mut seed = (id as u64) ^ ((idx as u64) << 21) ^ 0xA3C59AC3;
                            let mut rows: Vec<Value> = Vec::with_capacity(out);
                            for _ in 0..out {
                                let mut row: Vec<Value> = Vec::with_capacity(inw);
                                for _ in 0..inw { row.push(Value::Real(lcg(&mut seed) * range)); }
                                rows.push(Value::List(row));
                            }
                            let b: Vec<Value> = (0..out).map(|_| match init_kind.to_lowercase().as_str() { "zeros" => Value::Real(0.0), "ones" => Value::Real(1.0), _ => Value::Real(0.0) }).collect();
                            params.insert("W".into(), Value::List(rows));
                            params.insert("b".into(), Value::List(b));
                            curr_dim = out;
                        }
                        "Convolution" => {
                            // 1D conv: init per out channel kernel with KernelSize
                            let out_ch = params.get("Output").and_then(|v| as_i64(v)).unwrap_or(1).max(1) as usize;
                            let k = params.get("KernelSize").and_then(|v| as_i64(v)).unwrap_or(3).max(1) as usize;
                            let mut seed = (id as u64) ^ ((idx as u64) << 17) ^ 0x517CC1B7;
                            let mut kernels: Vec<Value> = Vec::with_capacity(out_ch);
                            for _ in 0..out_ch {
                                let mut ker: Vec<Value> = Vec::with_capacity(k);
                                for _ in 0..k { ker.push(Value::Real(lcg(&mut seed))); }
                                kernels.push(Value::List(ker));
                            }
                            let b: Vec<Value> = (0..out_ch).map(|_| Value::Real(0.0)).collect();
                            params.insert("W".into(), Value::List(kernels));
                            params.insert("b".into(), Value::List(b));
                            curr_dim = out_ch; // output length changes with input at runtime
                        }
                        "Conv2D" => {
                            let out_ch = params.get("Output").and_then(|v| as_i64(v)).unwrap_or(1).max(1) as usize;
                            let cin = params.get("InputChannels").and_then(|v| as_i64(v)).unwrap_or(1).max(1) as usize;
                            let (kh, kw) = parse_hw(assoc_get(&params, "KernelSize")).unwrap_or((3,3));
                            let mut seed = (id as u64) ^ ((idx as u64) << 11) ^ 0xA3C59AC3;
                            let mut oc_list: Vec<Value> = Vec::with_capacity(out_ch);
                            for _ in 0..out_ch {
                                let mut ic_list: Vec<Value> = Vec::with_capacity(cin);
                                for _ in 0..cin {
                                    let mut rows: Vec<Value> = Vec::with_capacity(kh);
                                    for _ in 0..kh {
                                        let mut row: Vec<Value> = Vec::with_capacity(kw);
                                        for _ in 0..kw { row.push(Value::Real(lcg(&mut seed))); }
                                        rows.push(Value::List(row));
                                    }
                                    ic_list.push(Value::List(rows));
                                }
                                oc_list.push(Value::List(ic_list));
                            }
                            let b: Vec<Value> = (0..out_ch).map(|_| Value::Real(0.0)).collect();
                            params.insert("W".into(), Value::List(oc_list));
                            params.insert("b".into(), Value::List(b));
                        }
                        "ConvTranspose2D" => {
                            // Initialize like Conv2D: W[out][in][kh][kw] and b[out]
                            let out_ch = params.get("Output").and_then(|v| as_i64(v)).unwrap_or(1).max(1) as usize;
                            let cin = params.get("InputChannels").and_then(|v| as_i64(v)).unwrap_or(1).max(1) as usize;
                            let (kh, kw) = parse_hw(assoc_get(&params, "KernelSize")).unwrap_or((3,3));
                            let mut seed = (id as u64) ^ ((idx as u64) << 9) ^ 0x9E3779B97F4A7C15;
                            let mut oc_list: Vec<Value> = Vec::with_capacity(out_ch);
                            for _ in 0..out_ch {
                                let mut ic_list: Vec<Value> = Vec::with_capacity(cin);
                                for _ in 0..cin {
                                    let mut rows: Vec<Value> = Vec::with_capacity(kh);
                                    for _ in 0..kh {
                                        let mut row: Vec<Value> = Vec::with_capacity(kw);
                                        for _ in 0..kw { row.push(Value::Real(lcg(&mut seed))); }
                                        rows.push(Value::List(row));
                                    }
                                    ic_list.push(Value::List(rows));
                                }
                                oc_list.push(Value::List(ic_list));
                            }
                            let b: Vec<Value> = (0..out_ch).map(|_| Value::Real(0.0)).collect();
                            params.insert("W".into(), Value::List(oc_list));
                            params.insert("b".into(), Value::List(b));
                        }
                        "SeparableConv2D" => {
                            // Initialize depthwise W per input channel and pointwise WPW[cout][cin], plus bias b[cout]
                            let cin = params.get("InputChannels").and_then(|v| as_i64(v)).unwrap_or(1).max(1) as usize;
                            let cout = params.get("Output").and_then(|v| as_i64(v)).unwrap_or(cin as i64).max(1) as usize;
                            let (kh, kw) = parse_hw(assoc_get(&params, "KernelSize")).unwrap_or((3,3));
                            let mut seed = (id as u64) ^ ((idx as u64) << 7) ^ 0xB5297A4D;
                            // Depthwise kernels W[cin][kh][kw]
                            let mut ch_list: Vec<Value> = Vec::with_capacity(cin);
                            for _ in 0..cin {
                                let mut rows: Vec<Value> = Vec::with_capacity(kh);
                                for _ in 0..kh {
                                    let mut row: Vec<Value> = Vec::with_capacity(kw);
                                    for _ in 0..kw { row.push(Value::Real(lcg(&mut seed))); }
                                    rows.push(Value::List(row));
                                }
                                ch_list.push(Value::List(rows));
                            }
                            params.insert("W".into(), Value::List(ch_list));
                            // Pointwise weights WPW[cout][cin]
                            let mut wpw_rows: Vec<Value> = Vec::with_capacity(cout);
                            for _ in 0..cout {
                                let mut row: Vec<Value> = Vec::with_capacity(cin);
                                for _ in 0..cin { row.push(Value::Real(lcg(&mut seed))); }
                                wpw_rows.push(Value::List(row));
                            }
                            params.insert("WPW".into(), Value::List(wpw_rows));
                            // Bias
                            let b: Vec<Value> = (0..cout).map(|_| Value::Real(0.0)).collect();
                            params.insert("b".into(), Value::List(b));
                        }
                        "MultiHeadAttention" => {
                            // Initialize Wq,Wk,Wv,Wo (dim x dim) and bq,bk,bv,bo (dim)
                            let dim = assoc_get(&params, "ModelDim").and_then(|v| as_i64(v)).unwrap_or(0).max(0) as usize;
                            if dim > 0 {
                                let mut seed = (id as u64) ^ ((idx as u64) << 3) ^ 0xA2BFE8A1;
                                let init_mat = |seed: &mut u64| -> Value {
                                    let mut rows: Vec<Value> = Vec::with_capacity(dim);
                                    for _ in 0..dim { let mut row: Vec<Value> = Vec::with_capacity(dim); for _ in 0..dim { row.push(Value::Real(lcg(seed))); } rows.push(Value::List(row)); }
                                    Value::List(rows)
                                };
                                let zeros = |n: usize| -> Value { Value::List((0..n).map(|_| Value::Real(0.0)).collect()) };
                                params.entry("Wq".into()).or_insert_with(|| init_mat(&mut seed));
                                params.entry("Wk".into()).or_insert_with(|| init_mat(&mut seed));
                                params.entry("Wv".into()).or_insert_with(|| init_mat(&mut seed));
                                params.entry("Wo".into()).or_insert_with(|| init_mat(&mut seed));
                                params.entry("bq".into()).or_insert_with(|| zeros(dim));
                                params.entry("bk".into()).or_insert_with(|| zeros(dim));
                                params.entry("bv".into()).or_insert_with(|| zeros(dim));
                                params.entry("bo".into()).or_insert_with(|| zeros(dim));
                            }
                        }
                        "TransformerEncoder" => {
                            // Initialize MHA and FFN if missing
                            let dim = assoc_get(&params, "ModelDim").and_then(|v| as_i64(v)).unwrap_or(0).max(0) as usize;
                            let hidden = params.get("HiddenDim").and_then(|v| as_i64(v)).unwrap_or((dim as i64)*4).max(1) as usize;
                            if dim>0 {
                                let mut seed = (id as u64) ^ ((idx as u64) << 2) ^ 0x243F6A88;
                                let init_mat = |rows: usize, cols: usize, seed: &mut u64| -> Value {
                                    let mut out: Vec<Value> = Vec::with_capacity(rows);
                                    for _ in 0..rows { let mut row: Vec<Value> = Vec::with_capacity(cols); for _ in 0..cols { row.push(Value::Real(lcg(seed))); } out.push(Value::List(row)); }
                                    Value::List(out)
                                };
                                let zeros = |n: usize| -> Value { Value::List((0..n).map(|_| Value::Real(0.0)).collect()) };
                                // MHA weights
                                params.entry("Wq".into()).or_insert_with(|| init_mat(dim, dim, &mut seed));
                                params.entry("Wk".into()).or_insert_with(|| init_mat(dim, dim, &mut seed));
                                params.entry("Wv".into()).or_insert_with(|| init_mat(dim, dim, &mut seed));
                                params.entry("Wo".into()).or_insert_with(|| init_mat(dim, dim, &mut seed));
                                params.entry("bq".into()).or_insert_with(|| zeros(dim));
                                params.entry("bk".into()).or_insert_with(|| zeros(dim));
                                params.entry("bv".into()).or_insert_with(|| zeros(dim));
                                params.entry("bo".into()).or_insert_with(|| zeros(dim));
                                // FFN weights
                                params.entry("W1".into()).or_insert_with(|| init_mat(hidden, dim, &mut seed));
                                params.entry("b1".into()).or_insert_with(|| zeros(hidden));
                                params.entry("W2".into()).or_insert_with(|| init_mat(dim, hidden, &mut seed));
                                params.entry("b2".into()).or_insert_with(|| zeros(dim));
                                params.entry("Activation".into()).or_insert(Value::String("Gelu".into()));
                                // Optional gated FFN
                                if let Some(v) = assoc_get(&params, "FFNVariant") {
                                    if matches!(v, Value::String(s) if !s.is_empty()) || matches!(v, Value::Symbol(_)) {
                                        params.entry("Wg".into()).or_insert_with(|| init_mat(hidden, dim, &mut seed));
                                        params.entry("bg".into()).or_insert_with(|| zeros(hidden));
                                    }
                                }
                            }
                        }
                        "RMSNorm" => {
                            // Initialize Gamma/Beta (optional) for sequence dim ModelDim
                            let dim = assoc_get(&params, "ModelDim").and_then(|v| as_i64(v)).unwrap_or(0).max(0) as usize;
                            if dim>0 {
                                let gamma: Vec<Value> = (0..dim).map(|_| Value::Real(1.0)).collect();
                                let beta: Vec<Value> = (0..dim).map(|_| Value::Real(0.0)).collect();
                                params.entry("Gamma".into()).or_insert(Value::List(gamma));
                                params.entry("Beta".into()).or_insert(Value::List(beta));
                                params.entry("Epsilon".into()).or_insert(Value::Real(1e-5));
                            }
                        }
                        "FFN" => {
                            let dim = assoc_get(&params, "ModelDim").and_then(|v| as_i64(v)).unwrap_or(0).max(0) as usize;
                            let hidden = params.get("HiddenDim").and_then(|v| as_i64(v)).unwrap_or((dim as i64)*4).max(1) as usize;
                            if dim>0 {
                                let mut seed = (id as u64) ^ ((idx as u64) << 6) ^ 0x0F1BBCDC;
                                let init_mat = |rows: usize, cols: usize, seed: &mut u64| -> Value {
                                    let mut out: Vec<Value> = Vec::with_capacity(rows);
                                    for _ in 0..rows { let mut row: Vec<Value> = Vec::with_capacity(cols); for _ in 0..cols { row.push(Value::Real(lcg(seed))); } out.push(Value::List(row)); }
                                    Value::List(out)
                                };
                                let zeros = |n: usize| -> Value { Value::List((0..n).map(|_| Value::Real(0.0)).collect()) };
                                params.entry("W1".into()).or_insert_with(|| init_mat(hidden, dim, &mut seed));
                                params.entry("b1".into()).or_insert_with(|| zeros(hidden));
                                params.entry("Wg".into()).or_insert_with(|| init_mat(hidden, dim, &mut seed));
                                params.entry("bg".into()).or_insert_with(|| zeros(hidden));
                                params.entry("W2".into()).or_insert_with(|| init_mat(dim, hidden, &mut seed));
                                params.entry("b2".into()).or_insert_with(|| zeros(dim));
                                params.entry("Variant".into()).or_insert(Value::String("SwiGLU".into()));
                            }
                        }
                        "CrossAttention" => {
                            let dim = params.get("ModelDim").and_then(|v| as_i64(v)).unwrap_or(0).max(0) as usize;
                            if dim > 0 {
                                let mut seed = (id as u64) ^ ((idx as u64) << 4) ^ 0x13579BDFu64;
                                let init_mat = |seed: &mut u64| -> Value {
                                    let mut rows: Vec<Value> = Vec::with_capacity(dim);
                                    for _ in 0..dim { let mut row: Vec<Value> = Vec::with_capacity(dim); for _ in 0..dim { row.push(Value::Real(lcg(seed))); } rows.push(Value::List(row)); }
                                    Value::List(rows)
                                };
                                let zeros = |n: usize| -> Value { Value::List((0..n).map(|_| Value::Real(0.0)).collect()) };
                                params.entry("Wq".into()).or_insert_with(|| init_mat(&mut seed));
                                params.entry("Wk".into()).or_insert_with(|| init_mat(&mut seed));
                                params.entry("Wv".into()).or_insert_with(|| init_mat(&mut seed));
                                params.entry("Wo".into()).or_insert_with(|| init_mat(&mut seed));
                                params.entry("bq".into()).or_insert_with(|| zeros(dim));
                                params.entry("bk".into()).or_insert_with(|| zeros(dim));
                                params.entry("bv".into()).or_insert_with(|| zeros(dim));
                                params.entry("bo".into()).or_insert_with(|| zeros(dim));
                            }
                        }
                        "PositionalEmbedding" => {
                            let (seq, dim) = match (assoc_get(&params, "SeqLen").and_then(|v| as_i64(v)), assoc_get(&params, "ModelDim").and_then(|v| as_i64(v))) { (Some(s), Some(d)) if s>0 && d>0 => (s as usize, d as usize), _ => (0,0) };
                            if seq>0 && dim>0 {
                                let mut seed = (id as u64) ^ ((idx as u64) << 10) ^ 0xCAFEBABEu64;
                                let mut rows: Vec<Value> = Vec::with_capacity(seq);
                                for _ in 0..seq { let mut row: Vec<Value> = Vec::with_capacity(dim); for _ in 0..dim { row.push(Value::Real(lcg(&mut seed))); } rows.push(Value::List(row)); }
                                params.entry("P".into()).or_insert(Value::List(rows));
                            }
                        }
                        "PatchEmbedding2D" => {
                            // W[ModelDim][Cin][ph][pw], b[ModelDim]
                            let d = assoc_get(&params, "ModelDim").and_then(|v| as_i64(v)).unwrap_or(1).max(1) as usize;
                            let cin = assoc_get(&params, "InputChannels").and_then(|v| as_i64(v)).unwrap_or(1).max(1) as usize;
                            let (ph, pw) = parse_hw(assoc_get(&params, "PatchSize")).unwrap_or((16,16));
                            let mut seed = (id as u64) ^ ((idx as u64) << 12) ^ 0xDEADBEEFu64;
                            let mut oc_list: Vec<Value> = Vec::with_capacity(d);
                            for _ in 0..d {
                                let mut ic_list: Vec<Value> = Vec::with_capacity(cin);
                                for _ in 0..cin {
                                    let mut rows: Vec<Value> = Vec::with_capacity(ph);
                                    for _ in 0..ph { let mut row: Vec<Value> = Vec::with_capacity(pw); for _ in 0..pw { row.push(Value::Real(lcg(&mut seed))); } rows.push(Value::List(row)); }
                                    ic_list.push(Value::List(rows));
                                }
                                oc_list.push(Value::List(ic_list));
                            }
                            let b: Vec<Value> = (0..d).map(|_| Value::Real(0.0)).collect();
                            params.entry("W".into()).or_insert(Value::List(oc_list));
                            params.entry("b".into()).or_insert(Value::List(b));
                        }
                        "TransformerDecoder" => {
                            // Initialize self-attn (1), cross-attn (2), and FFN if missing
                            let dim = params.get("ModelDim").and_then(|v| as_i64(v)).unwrap_or(0).max(0) as usize;
                            let hidden = params.get("HiddenDim").and_then(|v| as_i64(v)).unwrap_or((dim as i64)*4).max(1) as usize;
                            if dim>0 {
                                let mut seed = (id as u64) ^ ((idx as u64) << 1) ^ 0x13198A2E;
                                let init_mat = |rows: usize, cols: usize, seed: &mut u64| -> Value {
                                    let mut out: Vec<Value> = Vec::with_capacity(rows);
                                    for _ in 0..rows { let mut row: Vec<Value> = Vec::with_capacity(cols); for _ in 0..cols { row.push(Value::Real(lcg(seed))); } out.push(Value::List(row)); }
                                    Value::List(out)
                                };
                                let zeros = |n: usize| -> Value { Value::List((0..n).map(|_| Value::Real(0.0)).collect()) };
                                // Self-attention weights (suffix 1)
                                params.entry("Wq1".into()).or_insert_with(|| init_mat(dim, dim, &mut seed));
                                params.entry("Wk1".into()).or_insert_with(|| init_mat(dim, dim, &mut seed));
                                params.entry("Wv1".into()).or_insert_with(|| init_mat(dim, dim, &mut seed));
                                params.entry("Wo1".into()).or_insert_with(|| init_mat(dim, dim, &mut seed));
                                params.entry("bq1".into()).or_insert_with(|| zeros(dim));
                                params.entry("bk1".into()).or_insert_with(|| zeros(dim));
                                params.entry("bv1".into()).or_insert_with(|| zeros(dim));
                                params.entry("bo1".into()).or_insert_with(|| zeros(dim));
                                // Cross-attention weights (suffix 2)
                                params.entry("Wq2".into()).or_insert_with(|| init_mat(dim, dim, &mut seed));
                                params.entry("Wk2".into()).or_insert_with(|| init_mat(dim, dim, &mut seed));
                                params.entry("Wv2".into()).or_insert_with(|| init_mat(dim, dim, &mut seed));
                                params.entry("Wo2".into()).or_insert_with(|| init_mat(dim, dim, &mut seed));
                                params.entry("bq2".into()).or_insert_with(|| zeros(dim));
                                params.entry("bk2".into()).or_insert_with(|| zeros(dim));
                                params.entry("bv2".into()).or_insert_with(|| zeros(dim));
                                params.entry("bo2".into()).or_insert_with(|| zeros(dim));
                                // FFN
                                params.entry("W1".into()).or_insert_with(|| init_mat(hidden, dim, &mut seed));
                                params.entry("b1".into()).or_insert_with(|| zeros(hidden));
                                if let Some(v) = assoc_get(&params, "FFNVariant") {
                                    if matches!(v, Value::String(s) if !s.is_empty()) || matches!(v, Value::Symbol(_)) {
                                        params.entry("Wg".into()).or_insert_with(|| init_mat(hidden, dim, &mut seed));
                                        params.entry("bg".into()).or_insert_with(|| zeros(hidden));
                                    }
                                }
                                params.entry("W2".into()).or_insert_with(|| init_mat(dim, hidden, &mut seed));
                                params.entry("b2".into()).or_insert_with(|| zeros(dim));
                                params.entry("Activation".into()).or_insert(Value::String("Gelu".into()));
                            }
                        }
                        "BatchNorm" | "LayerNorm" => {
                            let width = if curr_dim > 0 { curr_dim } else { 1 };
                            let gamma: Vec<Value> = (0..width).map(|_| Value::Real(1.0)).collect();
                            let beta: Vec<Value> = (0..width).map(|_| Value::Real(0.0)).collect();
                            params.insert("Gamma".into(), Value::List(gamma));
                            params.insert("Beta".into(), Value::List(beta));
                        }
                        _ => {}
                    }
                    let mut m2 = m.clone();
                    m2.insert("Params".into(), Value::Assoc(params));
                    current = Value::Assoc(m2);
                }
                new_layers.push(current);
            }
            st.layers = new_layers;
            st.initialized = true;
            reg().lock().unwrap().insert(id, st);
        }
        return net;
    }
    net
}

// --------- train/apply ---------

fn net_train(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() {
        return Value::Expr { head: Box::new(Value::Symbol("NetTrain".into())), args };
    }
    let (pos, opts) = parse_opts(ev, &args);
    let net = ev.eval(pos.get(0).cloned().unwrap_or(Value::Symbol("Null".into())));
    let _data = pos.get(1).cloned().unwrap_or(Value::List(vec![]));
    let epochs = get_option(&opts, "Epochs")
        .and_then(|v| match v {
            Value::Integer(n) => Some(*n as usize),
            Value::Real(x) => Some(*x as usize),
            _ => None,
        })
        .unwrap_or(1);
    let method = get_option(&opts, "Method")
        .and_then(|v| match v {
            Value::String(s) | Value::Symbol(s) => Some(s.clone()),
            _ => None,
        })
        .unwrap_or("Adam".into());
    let batch = get_option(&opts, "BatchSize")
        .and_then(|v| match v {
            Value::Integer(n) => Some(*n as usize),
            Value::Real(x) => Some(*x as usize),
            _ => None,
        })
        .unwrap_or(32);
    if let Some(id) = get_net_id(&net) {
        if let Some(mut st) = reg().lock().unwrap().remove(&id) {
            st.trained_epochs += epochs;
            st.method = method;
            st.batch_size = batch;
            reg().lock().unwrap().insert(id, st);
        }
    }
    // Return callable model: PureFunction[x |-> NetApply[net, x, opts]]
    let body = Value::expr(
        Value::symbol("NetApply"),
        vec![net.clone(), Value::slot(None), Value::slot(Some(2))],
    );
    Value::PureFunction { params: None, body: Box::new(body) }
}

// Canonical alias: Fit delegates to NetTrain for now
fn fit(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // Fit[net, data, opts?] -> NetTrain[...]
    Value::Expr { head: Box::new(Value::Symbol("NetTrain".into())), args }
}

// Dropout: Tier1 stub (pass-through). Honors P option signature but no-op.
// removed: Dropout stub (use existing Dropout->DropoutLayer mapping)

fn net_apply(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() {
        return Value::Expr { head: Box::new(Value::Symbol("NetApply".into())), args };
    }
    if args.len() < 2 {
        return Value::Symbol("Null".into());
    }
    let net_v = ev.eval(args[0].clone());
    let x_in = ev.eval(args[1].clone());
    let _opts =
        if args.len() >= 3 { ev.eval(args[2].clone()) } else { Value::Assoc(HashMap::new()) };
    // Accept lists and tensors; infer 2D shape when possible
    let mut curr_chw: Option<(usize, usize, usize)> = infer_chw(&x_in);
    let mut curr_sd: Option<(usize, usize)> = infer_seqdim(&x_in);
    let mut x_vec: Vec<f64> = match &x_in {
        Value::PackedArray { shape: _shape, data } => data.clone(),
        _ => match to_vec_f64(&x_in) {
            Some(v) => v,
            None => {
                let mut out = Vec::new();
                if flatten_to_vec_f64(&x_in, &mut out) { out } else { return x_in } }
        },
    };
    let id = match get_net_id(&net_v) {
        Some(id) => id,
        None => return x_in,
    };
    let mut state = match reg().lock().unwrap().remove(&id) {
        Some(s) => s,
        None => return x_in,
    };
    // Only NetChain forward is supported for MVP
    if state.kind == "Chain" {
        // Iterate layers; lazy init for Linear when needed
        let mut new_layers: Vec<Value> = Vec::with_capacity(state.layers.len());
        let mut in_dim = x_vec.len();
        let mut pending_shape: Option<Vec<usize>> = None;
        let mut curr_chw: Option<(usize, usize, usize)> = infer_chw(&x_in);
        let mut curr_sd: Option<(usize, usize)> = infer_seqdim(&x_in);
        let mut force_1d_tensor: bool = false;
        for (idx, layer) in state.layers.iter().enumerate() {
            let mut current = layer.clone();
            if let Value::Assoc(m) = &layer {
                let ltype = m
                    .get("LayerType")
                    .and_then(|v| match v {
                        Value::String(s) => Some(s.as_str().to_string()),
                        Value::Symbol(s) => Some(s.clone()),
                        _ => None,
                    })
                    .unwrap_or_default();
                let mut params = m
                    .get("Params")
                    .and_then(|v| if let Value::Assoc(p) = v { Some(p.clone()) } else { None })
                    .unwrap_or_default();
                match ltype.as_str() {
                    "Linear" => {
                        let (updated, _out_dim) = ensure_linear_params(id, idx, layer, in_dim);
                        current = updated;
                        // apply
                        if let Value::Assoc(m2) = &current {
                            if let Some(Value::Assoc(p2)) = m2.get("Params") {
                                if let Some(y) = apply_linear(p2, &x_vec) {
                                    x_vec = y;
                                    in_dim = x_vec.len();
                                }
                            }
                        }
                    }
                    "Convolution" => {
                        let (updated, _out_ch, _k) = ensure_conv1d_params(id, idx, layer, in_dim);
                        current = updated;
                        if let Value::Assoc(m2) = &current {
                            if let Some(Value::Assoc(p2)) = m2.get("Params") {
                                if let Some(y) = apply_conv1d(p2, &x_vec) {
                                    x_vec = y;
                                    in_dim = x_vec.len();
                                }
                            }
                        }
                    }
                    "Pooling" => {
                        if let Some(y) = apply_pool1d(&params, &x_vec) {
                            x_vec = y;
                            in_dim = x_vec.len();
                        }
                    }
                    "Conv2D" => {
                        if params.get("InputChannels").is_none() || params.get("Height").is_none() || params.get("Width").is_none() {
                            if let Some((cin0,h0,w0)) = curr_chw { params.insert("InputChannels".into(), Value::Integer(cin0 as i64)); params.insert("Height".into(), Value::Integer(h0 as i64)); params.insert("Width".into(), Value::Integer(w0 as i64)); }
                        }
                        // Optional guard against excessive ops
                        if let (Some(cin0), Some(h0), Some(w0)) = (params.get("InputChannels").and_then(|v| as_i64(v)), params.get("Height").and_then(|v| as_i64(v)), params.get("Width").and_then(|v| as_i64(v))) {
                            if let Some(limit) = max_ops_limit() {
                                let (kh, kw) = parse_hw(assoc_get(&params, "KernelSize")).unwrap_or((3,3));
                                let (sh, sw) = parse_hw(assoc_get(&params, "Stride")).unwrap_or((1,1));
                                let (ph, pw) = parse_hw(assoc_get(&params, "Padding")).unwrap_or((0,0));
                                let cout = params.get("Output").and_then(|v| as_i64(v)).unwrap_or(1).max(1) as usize;
                                let h = h0 as usize; let w = w0 as usize;
                                let oh = if h + 2*ph >= kh { ((h + 2*ph - kh) / sh.max(1)) + 1 } else { 0 };
                                let ow = if w + 2*pw >= kw { ((w + 2*pw - kw) / sw.max(1)) + 1 } else { 0 };
                                let ops = cout.saturating_mul(oh).saturating_mul(ow).saturating_mul(cin0.max(0) as usize).saturating_mul(kh).saturating_mul(kw);
                                if ops > limit { reg().lock().unwrap().insert(id, state); return error_assoc("Conv2D exceeds LYRA_NN_MAX_OPS"); }
                            }
                        }
                        if let Some(y) = apply_conv2d(&params, &x_vec) {
                            // compute output dims
                            if let (Some(cin0), Some(h0), Some(w0)) = (params.get("InputChannels").and_then(|v| as_i64(v)), params.get("Height").and_then(|v| as_i64(v)), params.get("Width").and_then(|v| as_i64(v))) {
                                let (kh, kw) = parse_hw(assoc_get(&params, "KernelSize")).unwrap_or((3,3));
                                let (sh, sw) = parse_hw(assoc_get(&params, "Stride")).unwrap_or((1,1));
                                let (ph, pw) = parse_hw(assoc_get(&params, "Padding")).unwrap_or((0,0));
                                let cout = params.get("Output").and_then(|v| as_i64(v)).unwrap_or(1).max(1) as usize;
                                let h = h0 as usize; let w = w0 as usize;
                                let oh = if h + 2*ph >= kh { ((h + 2*ph - kh) / sh.max(1)) + 1 } else { 0 };
                                let ow = if w + 2*pw >= kw { ((w + 2*pw - kw) / sw.max(1)) + 1 } else { 0 };
                                curr_chw = Some((cout, oh, ow)); let _ = cin0; // silence unused
                            }
                            x_vec = y;
                            in_dim = x_vec.len();
                            // persist filled params into layer
                            let mut m2 = m.clone(); m2.insert("Params".into(), Value::Assoc(params.clone())); current = Value::Assoc(m2);
                        }
                    }
                    "SeparableConv2D" => {
                        if params.get("InputChannels").is_none() || params.get("Height").is_none() || params.get("Width").is_none() {
                            if let Some((cin0,h0,w0)) = curr_chw { params.insert("InputChannels".into(), Value::Integer(cin0 as i64)); params.insert("Height".into(), Value::Integer(h0 as i64)); params.insert("Width".into(), Value::Integer(w0 as i64)); }
                        }
                        if let (Some(cin0), Some(h0), Some(w0)) = (params.get("InputChannels").and_then(|v| as_i64(v)), params.get("Height").and_then(|v| as_i64(v)), params.get("Width").and_then(|v| as_i64(v))) {
                            if let Some(limit) = max_ops_limit() {
                                let (kh, kw) = parse_hw(assoc_get(&params, "KernelSize")).unwrap_or((3,3));
                                let (sh, sw) = parse_hw(assoc_get(&params, "Stride")).unwrap_or((1,1));
                                let (ph, pw) = parse_hw(assoc_get(&params, "Padding")).unwrap_or((0,0));
                                let cout = params.get("Output").and_then(|v| as_i64(v)).unwrap_or(1).max(1) as usize;
                                let h = h0 as usize; let w = w0 as usize;
                                let oh = if h + 2*ph >= kh { ((h + 2*ph - kh) / sh.max(1)) + 1 } else { 0 };
                                let ow = if w + 2*pw >= kw { ((w + 2*pw - kw) / sw.max(1)) + 1 } else { 0 };
                                let depthwise_ops = (cin0.max(0) as usize).saturating_mul(oh).saturating_mul(ow).saturating_mul(kh).saturating_mul(kw);
                                let pointwise_ops = cout.saturating_mul(oh).saturating_mul(ow).saturating_mul(cin0.max(0) as usize);
                                let ops = depthwise_ops.saturating_add(pointwise_ops);
                                if ops > limit { reg().lock().unwrap().insert(id, state); return error_assoc("SeparableConv2D exceeds LYRA_NN_MAX_OPS"); }
                            }
                        }
                        if let Some(y) = apply_separable_conv2d(&params, &x_vec) {
                            if let (Some(_cin0), Some(h0), Some(w0)) = (params.get("InputChannels").and_then(|v| as_i64(v)), params.get("Height").and_then(|v| as_i64(v)), params.get("Width").and_then(|v| as_i64(v))) {
                                let (kh, kw) = parse_hw(assoc_get(&params, "KernelSize")).unwrap_or((3,3));
                                let (sh, sw) = parse_hw(assoc_get(&params, "Stride")).unwrap_or((1,1));
                                let (ph, pw) = parse_hw(assoc_get(&params, "Padding")).unwrap_or((0,0));
                                let cout = params.get("Output").and_then(|v| as_i64(v)).unwrap_or(1).max(1) as usize;
                                let h = h0 as usize; let w = w0 as usize;
                                let oh = if h + 2*ph >= kh { ((h + 2*ph - kh) / sh.max(1)) + 1 } else { 0 };
                                let ow = if w + 2*pw >= kw { ((w + 2*pw - kw) / sw.max(1)) + 1 } else { 0 };
                                curr_chw = Some((cout, oh, ow));
                            }
                            x_vec = y; in_dim = x_vec.len();
                            let mut m2 = m.clone(); m2.insert("Params".into(), Value::Assoc(params.clone())); current = Value::Assoc(m2);
                        }
                    }
                    "GroupNorm" => {
                        if params.get("InputChannels").is_none() || params.get("Height").is_none() || params.get("Width").is_none() {
                            if let Some((cin0,h0,w0)) = curr_chw { params.insert("InputChannels".into(), Value::Integer(cin0 as i64)); params.insert("Height".into(), Value::Integer(h0 as i64)); params.insert("Width".into(), Value::Integer(w0 as i64)); }
                        }
                        // apply group norm
                        if let Some((cin0,h0,w0)) = curr_chw {
                            let cin = cin0 as usize; let h = h0 as usize; let w = w0 as usize;
                            let eps = assoc_get(&params, "Epsilon").and_then(|v| as_f64(v)).unwrap_or(1e-5);
                            let ng = assoc_get(&params, "NumGroups").and_then(|v| as_i64(v)).unwrap_or(1).max(1) as usize;
                            let gsz = (cin + ng - 1) / ng;
                            for g in 0..ng {
                                let start = g * gsz; if start >= cin { break; }
                                let end = ((g+1)*gsz).min(cin);
                                let mut mean = 0.0; let mut cnt = 0usize;
                                for c in start..end { for i in 0..(h*w) { mean += x_vec[c*h*w + i]; cnt += 1; } }
                                if cnt==0 { continue; }
                                mean /= cnt as f64;
                                let mut var = 0.0; for c in start..end { for i in 0..(h*w) { let d = x_vec[c*h*w + i]-mean; var += d*d; } }
                                var /= cnt as f64; let inv = 1.0 / (var + eps).sqrt();
                                for c in start..end { for i in 0..(h*w) { let idx = c*h*w + i; x_vec[idx] = (x_vec[idx]-mean)*inv; } }
                            }
                            // apply gamma/beta per channel if provided
                            if let Some(Value::List(gs)) = params.get("Gamma") { for c in 0..cin { let g = gs.get(c).and_then(|v| as_f64(v)).unwrap_or(1.0); for i in 0..(h*w) { x_vec[c*h*w+i]*=g; } } }
                            if let Some(Value::List(bs)) = params.get("Beta") { for c in 0..cin { let b = bs.get(c).and_then(|v| as_f64(v)).unwrap_or(0.0); for i in 0..(h*w) { x_vec[c*h*w+i]+=b; } } }
                        }
                    }
                    "Residual" => {
                        // Run inner layers and add skip connection when lengths match
                        let skip = x_vec.clone(); let mut y = x_vec.clone(); let mut inner_chw = curr_chw;
                        if let Some(Value::List(layers)) = params.get("Layers") {
                            for lay in layers {
                                if let Value::Assoc(m2) = lay {
                                    let lt = m2.get("LayerType").and_then(|v| match v { Value::String(s)|Value::Symbol(s)=>Some(s.clone()), _=>None }).unwrap_or_default();
                                    let p2 = m2.get("Params").and_then(|v| if let Value::Assoc(mm)=v { Some(mm.clone()) } else { None }).unwrap_or_default();
                                    // Handle a subset of layer types
                                    match lt.as_str() {
                                        "Activation" => { let kind=p2.get("Type").and_then(|v| match v { Value::String(s)|Value::Symbol(s)=>Some(s.clone()), _=>None }).unwrap_or("ReLU".into()); apply_activation(&kind, &mut y); }
                                        "BatchNorm" => { apply_batchnorm(&p2, &mut y); }
                                        "LayerNorm" => { apply_layernorm(&p2, &mut y); }
                        _ => { /* ignore complex types in residual MVP */ }
                                    }
                                }
                            }
                        }
                        if y.len() == skip.len() { for i in 0..y.len() { y[i] += skip[i]; } }
                        x_vec = y; in_dim = x_vec.len(); curr_chw = inner_chw;
                    }
                    "Upsample2D" => {
                        if params.get("InputChannels").is_none() || params.get("Height").is_none() || params.get("Width").is_none() {
                            if let Some((cin0,h0,w0)) = curr_chw { params.insert("InputChannels".into(), Value::Integer(cin0 as i64)); params.insert("Height".into(), Value::Integer(h0 as i64)); params.insert("Width".into(), Value::Integer(w0 as i64)); }
                        }
                        if let Some((cin0,h0,w0)) = curr_chw {
                            let cin = cin0 as usize; let h = h0 as usize; let w = w0 as usize;
                            let (sh, sw) = parse_hw(assoc_get(&params, "Scale")).unwrap_or((2,2));
                            let mode = params.get("Mode").and_then(|v| match v { Value::String(s)|Value::Symbol(s)=>Some(s.clone()), _=>None }).unwrap_or("Nearest".into());
                            let oh = h * sh; let ow = w * sw;
                            let mut out = vec![0.0f64; cin*oh*ow];
                            if mode.to_lowercase() == "nearest" {
                                for c in 0..cin { for i in 0..oh { for j in 0..ow {
                                    let ih = i / sh; let jw = j / sw; let src = c*h*w + ih*w + jw; out[c*oh*ow + i*ow + j] = x_vec[src];
                                }}}
                            } else {
                                // Bilinear interpolation per channel
                                let sh_f = sh as f64; let sw_f = sw as f64;
                                for c in 0..cin {
                                    for i in 0..oh {
                                        let y = (i as f64) / sh_f;
                                        let y0 = y.floor() as isize;
                                        let y1 = (y0 + 1) as isize;
                                        let wy = y - (y0 as f64);
                                        let y0c = y0.clamp(0, (h as isize) - 1) as usize;
                                        let y1c = y1.clamp(0, (h as isize) - 1) as usize;
                                        for j in 0..ow {
                                            let x = (j as f64) / sw_f;
                                            let x0 = x.floor() as isize;
                                            let x1 = (x0 + 1) as isize;
                                            let wx = x - (x0 as f64);
                                            let x0c = x0.clamp(0, (w as isize) - 1) as usize;
                                            let x1c = x1.clamp(0, (w as isize) - 1) as usize;
                                            let idx = c*h*w;
                                            let v00 = x_vec[idx + y0c*w + x0c];
                                            let v01 = x_vec[idx + y0c*w + x1c];
                                            let v10 = x_vec[idx + y1c*w + x0c];
                                            let v11 = x_vec[idx + y1c*w + x1c];
                                            let top = v00 * (1.0 - wx) + v01 * wx;
                                            let bot = v10 * (1.0 - wx) + v11 * wx;
                                            let v = top * (1.0 - wy) + bot * wy;
                                            out[c*oh*ow + i*ow + j] = v;
                                        }
                                    }
                                }
                            }
                            x_vec = out; in_dim = x_vec.len(); curr_chw = Some((cin, oh, ow));
                        }
                    }
                    "ResidualBlock" => {
                        if params.get("InputChannels").is_none() || params.get("Height").is_none() || params.get("Width").is_none() {
                            if let Some((cin0,h0,w0)) = curr_chw { params.insert("InputChannels".into(), Value::Integer(cin0 as i64)); params.insert("Height".into(), Value::Integer(h0 as i64)); params.insert("Width".into(), Value::Integer(w0 as i64)); }
                        }
                        if let Some((cin0,h0,w0)) = curr_chw {
                            let cin = cin0 as usize; let h = h0 as usize; let w = w0 as usize;
                            let cout = params.get("Output").and_then(|v| as_i64(v)).unwrap_or(cin0 as i64).max(1) as usize;
                            let (kh, kw) = parse_hw(assoc_get(&params, "KernelSize")).unwrap_or((3,3));
                            let (sh, sw) = parse_hw(assoc_get(&params, "Stride")).unwrap_or((1,1));
                            let (ph, pw) = parse_hw(assoc_get(&params, "Padding")).unwrap_or((1,1));
                            let mut seed = (id as u64) ^ ((idx as u64) << 5) ^ 0xC3A5C85C97CB3127;
                            let ensure_wb = |p: &mut HashMap<String, Value>, wn: &str, bn: &str, in_c: usize, out_c: usize, kh: usize, kw: usize, seed: &mut u64| {
                                if !p.contains_key(wn) {
                                    let mut oc_list: Vec<Value> = Vec::with_capacity(out_c);
                                    for _ in 0..out_c {
                                        let mut ic_list: Vec<Value> = Vec::with_capacity(in_c);
                                        for _ in 0..in_c {
                                            let mut rows: Vec<Value> = Vec::with_capacity(kh);
                                            for _ in 0..kh { let mut row: Vec<Value> = Vec::with_capacity(kw); for _ in 0..kw { row.push(Value::Real(lcg(seed))); } rows.push(Value::List(row)); }
                                            ic_list.push(Value::List(rows));
                                        }
                                        oc_list.push(Value::List(ic_list));
                                    }
                                    p.insert(wn.into(), Value::List(oc_list));
                                }
                                if !p.contains_key(bn) { let b: Vec<Value> = (0..out_c).map(|_| Value::Real(0.0)).collect(); p.insert(bn.into(), Value::List(b)); }
                            };
                            ensure_wb(&mut params, "W1", "b1", cin, cout, kh, kw, &mut seed);
                            ensure_wb(&mut params, "W2", "b2", cout, cout, kh, kw, &mut seed);
                            let p1 = HashMap::from([
                                ("InputChannels".into(), Value::Integer(cin as i64)),
                                ("Height".into(), Value::Integer(h as i64)),
                                ("Width".into(), Value::Integer(w as i64)),
                                ("KernelSize".into(), params.get("KernelSize").cloned().unwrap_or(Value::Integer(3))),
                                ("Stride".into(), params.get("Stride").cloned().unwrap_or(Value::Integer(1))),
                                ("Padding".into(), params.get("Padding").cloned().unwrap_or(Value::Integer(1))),
                                ("W".into(), params.get("W1").cloned().unwrap()),
                                ("b".into(), params.get("b1").cloned().unwrap()),
                            ]);
                            let mut y1 = apply_conv2d(&p1, &x_vec).unwrap_or_else(|| x_vec.clone());
                            let act = params.get("Activation").and_then(|v| match v { Value::String(s)|Value::Symbol(s)=>Some(s.clone()), _=>None }).unwrap_or("Relu".into());
                            apply_activation(&act, &mut y1);
                            let p2_h = ((h + 2*ph as usize - kh)/sh.max(1)) + 1; let p2_w = ((w + 2*pw as usize - kw)/sw.max(1)) + 1;
                            let p2 = HashMap::from([
                                ("InputChannels".into(), Value::Integer(cout as i64)),
                                ("Height".into(), Value::Integer(p2_h as i64)),
                                ("Width".into(), Value::Integer(p2_w as i64)),
                                ("KernelSize".into(), params.get("KernelSize").cloned().unwrap_or(Value::Integer(3))),
                                ("Stride".into(), Value::Integer(1)),
                                ("Padding".into(), params.get("Padding").cloned().unwrap_or(Value::Integer(1))),
                                ("W".into(), params.get("W2").cloned().unwrap()),
                                ("b".into(), params.get("b2").cloned().unwrap()),
                            ]);
                            let mut y2 = apply_conv2d(&p2, &y1).unwrap_or_else(|| y1.clone());
                            let mut skip = x_vec.clone(); let mut skip_c = cin; let mut skip_h = h; let mut skip_w = w;
                            if skip_c != cout || skip_h != p2_h || skip_w != p2_w {
                                if !params.contains_key("WP") {
                                    let mut oc_list: Vec<Value> = Vec::with_capacity(cout);
                                    for _ in 0..cout { let mut ic_list: Vec<Value> = Vec::with_capacity(cin); for _ in 0..cin { ic_list.push(Value::List(vec![Value::List(vec![Value::Real(lcg(&mut seed))])])); } oc_list.push(Value::List(ic_list)); }
                                    params.insert("WP".into(), Value::List(oc_list)); params.insert("bP".into(), Value::List((0..cout).map(|_| Value::Real(0.0)).collect()));
                                }
                                let proj = HashMap::from([
                                    ("InputChannels".into(), Value::Integer(cin as i64)),
                                    ("Height".into(), Value::Integer(h as i64)),
                                    ("Width".into(), Value::Integer(w as i64)),
                                    ("KernelSize".into(), Value::Integer(1)),
                                    ("Stride".into(), params.get("Stride").cloned().unwrap_or(Value::Integer(1))),
                                    ("Padding".into(), Value::Integer(0)),
                                    ("W".into(), params.get("WP").cloned().unwrap()),
                                    ("b".into(), params.get("bP").cloned().unwrap()),
                                ]);
                                skip = apply_conv2d(&proj, &x_vec).unwrap_or_else(|| x_vec.clone());
                                skip_c = cout; skip_h = p2_h; skip_w = p2_w;
                            }
                            if y2.len() == skip.len() { for i in 0..y2.len() { y2[i] += skip[i]; } }
                            apply_activation(&act, &mut y2);
                            x_vec = y2; in_dim = x_vec.len(); curr_chw = Some((cout, p2_h, p2_w));
                            let mut m2 = m.clone(); m2.insert("Params".into(), Value::Assoc(params.clone())); current = Value::Assoc(m2);
                        }
                    }
                    "MultiHeadAttention" => {
                        // Params: SeqLen, ModelDim, NumHeads, Wq,Wk,Wv,Wo (dim x dim), bq,bk,bv,bo (dim)
                        let (seq, dim) = match (params.get("SeqLen").and_then(|v| as_i64(v)), params.get("ModelDim").and_then(|v| as_i64(v))) {
                            (Some(s), Some(d)) if s>0 && d>0 => (s as usize, d as usize),
                            _ => match curr_sd { Some((s,d)) => (s,d), None => (0,0) }
                        };
                        let heads = params.get("NumHeads").and_then(|v| as_i64(v)).unwrap_or(1).max(1) as usize;
                        if seq>0 && dim>0 && x_vec.len()==seq*dim && dim%heads==0 {
                            let head_dim = dim/heads;
                            // Helpers to read matrices
                            let read_mat = |name: &str| -> Vec<f64> {
                                if let Some(Value::List(rows)) = params.get(name) {
                                    let mut out = Vec::with_capacity(dim*dim);
                                    for r in 0..dim { if let Some(Value::List(row)) = rows.get(r) { for c in 0..dim { out.push(as_f64(row.get(c).unwrap_or(&Value::Integer(0))).unwrap_or(0.0)); } } }
                                    return out;
                                }
                                vec![0.0; dim*dim]
                            };
                            let read_vec = |name: &str| -> Vec<f64> {
                                if let Some(Value::List(vs)) = params.get(name) { (0..dim).map(|i| as_f64(vs.get(i).unwrap_or(&Value::Integer(0))).unwrap_or(0.0)).collect() } else { vec![0.0; dim] }
                            };
                            let wq = read_mat("Wq"); let wk = read_mat("Wk"); let wv = read_mat("Wv"); let wo = read_mat("Wo");
                            let bq = read_vec("bq"); let bk = read_vec("bk"); let bv = read_vec("bv"); let bo = read_vec("bo");
                            // X: seq x dim
                            let xmat = |r: usize, c: usize| -> f64 { x_vec[r*dim + c] };
                            // matmul seq x dim times dim x dim -> seq x dim
                            let proj = |w: &Vec<f64>, b: &Vec<f64>| -> Vec<f64> {
                                let mut out = vec![0.0; seq*dim];
                                for s in 0..seq { for j in 0..dim {
                                    let mut acc = 0.0; for k in 0..dim { acc += xmat(s,k) * w[k*dim + j]; }
                                    out[s*dim + j] = acc + b[j];
                                }}
                                out
                            };
                            let q = proj(&wq, &bq); let k = proj(&wk, &bk); let v = proj(&wv, &bv);
                            // Optional mask: either Causal, vector length seq, or matrix seq x seq.
                            let mut att_bias: Option<Vec<f64>> = None;
                            let causal = params.get("Causal").and_then(|v| as_bool(v)).unwrap_or(false)
                                || matches!(params.get("Mask"), Some(Value::String(s)) if s.eq_ignore_ascii_case("Causal"))
                                || matches!(params.get("Mask"), Some(Value::Symbol(s)) if s.eq_ignore_ascii_case("Causal"));
                            if causal {
                                let mut b = vec![0.0f64; seq*seq];
                                for i in 0..seq { for j in 0..seq { if j>i { b[i*seq + j] = -1e9; } } }
                                att_bias = Some(b);
                            } else if let Some(mv) = params.get("Mask") {
                                match mv {
                                    Value::List(rows) if !rows.is_empty() && matches!(rows[0], Value::List(_)) => {
                                        let mut b = vec![0.0f64; seq*seq];
                                        for i in 0..seq { if let Some(Value::List(r)) = rows.get(i) { for j in 0..seq { let v = r.get(j).cloned().unwrap_or(Value::Integer(1)); let x = as_f64(&v).unwrap_or_else(|| if matches!(v, Value::Boolean(false) | Value::Integer(0)) {0.0} else {1.0}); b[i*seq + j] = if x>0.0 { 0.0 } else { -1e9 }; } } }
                                        att_bias = Some(b);
                                    }
                                    Value::List(vs) if vs.len()==seq => {
                                        // Key padding mask: 1=keep,0=mask for each column j
                                        let mut b = vec![0.0f64; seq*seq];
                                        for j in 0..seq { let keep = as_f64(vs.get(j).unwrap_or(&Value::Integer(1))).unwrap_or(1.0) > 0.0; if !keep { for i in 0..seq { b[i*seq + j] = -1e9; } } }
                                        att_bias = Some(b);
                                    }
                                    _ => {}
                                }
                            }
                            // Split heads and compute attention per head
                            let scale = 1.0 / (head_dim as f64).sqrt();
                            let mut out_all = vec![0.0; seq*dim];
                            for h in 0..heads {
                                // slice indices in dim
                                let off = h*head_dim;
                                // compute attention scores: seq x seq
                                let mut att = vec![0.0; seq*seq];
                                for i in 0..seq { for j in 0..seq {
                                    let mut dot = 0.0; for d in 0..head_dim { dot += q[i*dim + off + d] * k[j*dim + off + d]; }
                                    att[i*seq + j] = dot * scale;
                                }}
                                // softmax rows
                                for i in 0..seq {
                                    if let Some(b) = &att_bias { for j in 0..seq { att[i*seq + j] += b[i*seq + j]; } }
                                    let row = &mut att[i*seq..(i+1)*seq];
                                    let m = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                                    let mut sum = 0.0; for x in row.iter_mut() { *x = (*x - m).exp(); sum += *x; }
                                    if sum>0.0 { for x in row.iter_mut() { *x /= sum; } }
                                }
                                // context = att * V_h   (seq x seq) * (seq x head_dim)
                                for i in 0..seq { for d in 0..head_dim {
                                    let mut acc = 0.0; for j in 0..seq { acc += att[i*seq + j] * v[j*dim + off + d]; }
                                    out_all[i*dim + off + d] = acc;
                                }}
                            }
                            // Output projection Wo + bo
                            let mut y = vec![0.0; seq*dim];
                            for s in 0..seq { for j in 0..dim { let mut acc=0.0; for k in 0..dim { acc += out_all[s*dim + k] * wo[k*dim + j]; } y[s*dim + j] = acc + bo[j]; }}
                            x_vec = y; in_dim = x_vec.len(); curr_sd = Some((seq, dim));
                            let mut m2 = m.clone(); m2.insert("Params".into(), Value::Assoc(params.clone())); current = Value::Assoc(m2);
                        }
                    }
                    "PositionalEncoding" => {
                        // Sinusoidal encoding added inplace; requires SeqLen, ModelDim
                        let (seq, dim) = match (params.get("SeqLen").and_then(|v| as_i64(v)), params.get("ModelDim").and_then(|v| as_i64(v))) {
                            (Some(s), Some(d)) if s>0 && d>0 => (s as usize, d as usize),
                            _ => match curr_sd { Some((s,d)) => (s,d), None => (0,0) }
                        };
                        if seq>0 && dim>0 && x_vec.len()==seq*dim {
                            for pos in 0..seq { for i in 0..(dim/2) {
                                let angle = (pos as f64) / (10000f64).powf((2*i) as f64 / (dim as f64));
                                let sinv = angle.sin(); let cosv = angle.cos();
                                x_vec[pos*dim + 2*i] += sinv;
                                if 2*i+1 < dim { x_vec[pos*dim + 2*i + 1] += cosv; }
                            }}
                            in_dim = x_vec.len(); curr_sd = Some((seq, dim));
                        }
                    }
                    "TransformerEncoder" => {
                        // Read dims
                        let (seq, dim) = match (params.get("SeqLen").and_then(|v| as_i64(v)), params.get("ModelDim").and_then(|v| as_i64(v))) {
                            (Some(s), Some(d)) if s>0 && d>0 => (s as usize, d as usize),
                            _ => match curr_sd { Some((s,d)) => (s,d), None => (0,0) }
                        };
                        let heads = params.get("NumHeads").and_then(|v| as_i64(v)).unwrap_or(1).max(1) as usize;
                        if seq>0 && dim>0 && x_vec.len()==seq*dim && dim%heads==0 {
                            let head_dim = dim/heads;
                            // Save residual
                            let x0 = x_vec.clone();
                            // --- MHA --- (reuse logic similar to branch)
                            let read_mat = |name: &str| -> Vec<f64> {
                                if let Some(Value::List(rows)) = params.get(name) {
                                    let mut out = Vec::with_capacity(dim*dim);
                                    for r in 0..dim { if let Some(Value::List(row)) = rows.get(r) { for c in 0..dim { out.push(as_f64(row.get(c).unwrap_or(&Value::Integer(0))).unwrap_or(0.0)); } } }
                                    return out;
                                }
                                vec![0.0; dim*dim]
                            };
                            let read_vec = |name: &str| -> Vec<f64> {
                                if let Some(Value::List(vs)) = params.get(name) { (0..dim).map(|i| as_f64(vs.get(i).unwrap_or(&Value::Integer(0))).unwrap_or(0.0)).collect() } else { vec![0.0; dim] }
                            };
                            let wq = read_mat("Wq"); let wk = read_mat("Wk"); let wv = read_mat("Wv"); let wo = read_mat("Wo");
                            let bq = read_vec("bq"); let bk = read_vec("bk"); let bv = read_vec("bv"); let bo = read_vec("bo");
                            let xmat = |r: usize, c: usize| -> f64 { x0[r*dim + c] };
                            let proj = |w: &Vec<f64>, b: &Vec<f64>| -> Vec<f64> {
                                let mut out = vec![0.0; seq*dim];
                                for s in 0..seq { for j in 0..dim {
                                    let mut acc = 0.0; for k in 0..dim { acc += xmat(s,k) * w[k*dim + j]; }
                                    out[s*dim + j] = acc + b[j];
                                }}
                                out
                            };
                            let q = proj(&wq, &bq); let k = proj(&wk, &bk); let v = proj(&wv, &bv);
                            // Optional mask: Causal, vector, or matrix as in MHA
                            let mut att_bias: Option<Vec<f64>> = None;
                            let causal = params.get("Causal").and_then(|v| as_bool(v)).unwrap_or(false)
                                || matches!(params.get("Mask"), Some(Value::String(s)) if s.eq_ignore_ascii_case("Causal"))
                                || matches!(params.get("Mask"), Some(Value::Symbol(s)) if s.eq_ignore_ascii_case("Causal"));
                            if causal {
                                let mut b = vec![0.0f64; seq*seq];
                                for i in 0..seq { for j in 0..seq { if j>i { b[i*seq + j] = -1e9; } } }
                                att_bias = Some(b);
                            } else if let Some(mv) = params.get("Mask") {
                                match mv {
                                    Value::List(rows) if !rows.is_empty() && matches!(rows[0], Value::List(_)) => {
                                        let mut b = vec![0.0f64; seq*seq];
                                        for i in 0..seq { if let Some(Value::List(r)) = rows.get(i) { for j in 0..seq { let v = r.get(j).cloned().unwrap_or(Value::Integer(1)); let x = as_f64(&v).unwrap_or_else(|| if matches!(v, Value::Boolean(false) | Value::Integer(0)) {0.0} else {1.0}); b[i*seq + j] = if x>0.0 { 0.0 } else { -1e9 }; } } }
                                        att_bias = Some(b);
                                    }
                                    Value::List(vs) if vs.len()==seq => {
                                        let mut b = vec![0.0f64; seq*seq];
                                        for j in 0..seq { let keep = as_f64(vs.get(j).unwrap_or(&Value::Integer(1))).unwrap_or(1.0) > 0.0; if !keep { for i in 0..seq { b[i*seq + j] = -1e9; } } }
                                        att_bias = Some(b);
                                    }
                                    _ => {}
                                }
                            }
                            let scale = 1.0 / (head_dim as f64).sqrt();
                            let mut att_out = vec![0.0; seq*dim];
                            for h in 0..heads {
                                let off = h*head_dim;
                                let mut att = vec![0.0; seq*seq];
                                for i in 0..seq { for j in 0..seq {
                                    let mut dot = 0.0; for d0 in 0..head_dim { dot += q[i*dim + off + d0] * k[j*dim + off + d0]; }
                                    att[i*seq + j] = dot * scale;
                                }}
                                for i in 0..seq {
                                    if let Some(b) = &att_bias { for j in 0..seq { att[i*seq + j] += b[i*seq + j]; } }
                                    let row = &mut att[i*seq..(i+1)*seq]; let m = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                                    let mut sum = 0.0; for x in row.iter_mut() { *x = (*x - m).exp(); sum += *x; }
                                    if sum>0.0 { for x in row.iter_mut() { *x /= sum; } }
                                }
                                for i in 0..seq { for d0 in 0..head_dim {
                                    let mut acc = 0.0; for j in 0..seq { acc += att[i*seq + j] * v[j*dim + off + d0]; }
                                    att_out[i*dim + off + d0] = acc;
                                }}
                            }
                            let mut y = vec![0.0; seq*dim];
                            for s in 0..seq { for j in 0..dim { let mut acc=0.0; for k0 in 0..dim { acc += att_out[s*dim + k0] * wo[k0*dim + j]; } y[s*dim + j] = acc + bo[j]; }}
                            // Residual + LN
                            for i in 0..(seq*dim) { y[i] += x0[i]; }
                            let mut ln_params = HashMap::from([
                                ("InputChannels".into(), Value::Integer(dim as i64)),
                                ("Height".into(), Value::Integer(seq as i64)),
                                ("Width".into(), Value::Integer(1)),
                                ("Epsilon".into(), Value::Real(1e-5)),
                                ("Gamma".into(), Value::List((0..dim).map(|_| Value::Real(1.0)).collect())),
                                ("Beta".into(), Value::List((0..dim).map(|_| Value::Real(0.0)).collect())),
                            ]);
                            apply_layernorm(&ln_params, &mut y);
                            // --- FFN ---: per-token apply linear; supports gated variants via FFNVariant
                            let get_param_map = |name_w: &str, name_b: &str| -> HashMap<String, Value> {
                                let mut m = HashMap::new();
                                if let Some(v) = params.get(name_w) { m.insert("W".into(), v.clone()); }
                                if let Some(v) = params.get(name_b) { m.insert("b".into(), v.clone()); }
                                m
                            };
                            let p1 = get_param_map("W1","b1");
                            let p2 = get_param_map("W2","b2");
                            let act = params.get("Activation").and_then(|v| match v { Value::String(s)|Value::Symbol(s)=>Some(s.clone()), _=>None }).unwrap_or("Gelu".into());
                            let hidden = assoc_get(&params, "W1").and_then(|v| if let Value::List(rows) = v { Some(rows.len()) } else { None }).unwrap_or(dim);
                            let ffn_variant = params.get("FFNVariant").and_then(|v| match v { Value::String(s)|Value::Symbol(s)=>Some(s.clone()), _=>None });
                            let mut y_ffn = vec![0.0; seq*dim];
                            if let Some(var) = ffn_variant.clone() {
                                // Gated path: compute value and gate, multiply, then project
                                let wg = assoc_get(&params, "Wg");
                                let bg = assoc_get(&params, "bg");
                                if wg.is_some() {
                                    // Compute value path h1 and gate g1
                                    for s in 0..seq {
                                        let inp = &y[s*dim..(s+1)*dim];
                                        // h1 = W1*inp + b1
                                        let mut h1 = apply_linear(&p1, inp).unwrap_or_else(|| vec![0.0; hidden]);
                                        // glin = Wg*inp + bg
                                        let mut glin = if let (Some(wgv), Some(bgv)) = (wg, bg) {
                                            let p = HashMap::from([("W".into(), wgv.clone()), ("b".into(), bgv.clone())]);
                                            apply_linear(&p, inp).unwrap_or_else(|| vec![0.0; hidden])
                                        } else { vec![0.0; hidden] };
                                        // Apply gate activation
                                        match var.to_ascii_lowercase().as_str() {
                                            "geglu" => apply_activation("Gelu", &mut glin),
                                            _ => { for v in glin.iter_mut() { let s1 = 1.0 / (1.0 + (-*v).exp()); *v = *v * s1; } },
                                        }
                                        // Multiply and project
                                        for i in 0..hidden { h1[i] *= glin[i]; }
                                        if let Some(o2) = apply_linear(&p2, &h1) {
                                            for j in 0..dim { y_ffn[s*dim + j] = o2.get(j).cloned().unwrap_or(0.0); }
                                        }
                                    }
                                } else {
                                    // Fallback to non-gated
                                    for s in 0..seq {
                                        let inp = &y[s*dim..(s+1)*dim];
                                        if let Some(mut h1) = apply_linear(&p1, inp) { apply_activation(&act, &mut h1); if let Some(o2) = apply_linear(&p2, &h1) { for j in 0..dim { y_ffn[s*dim + j] = o2.get(j).cloned().unwrap_or(0.0); } } }
                                    }
                                }
                            } else {
                                // Non-gated FFN
                                for s in 0..seq {
                                    let inp = &y[s*dim..(s+1)*dim];
                                    if let Some(mut h1) = apply_linear(&p1, inp) { apply_activation(&act, &mut h1); if let Some(o2) = apply_linear(&p2, &h1) { for j in 0..dim { y_ffn[s*dim + j] = o2.get(j).cloned().unwrap_or(0.0); } } }
                                }
                            }
                            // Residual + LN
                            for i in 0..(seq*dim) { y_ffn[i] += y[i]; }
                            apply_layernorm(&ln_params, &mut y_ffn);
                            x_vec = y_ffn; in_dim = x_vec.len(); curr_sd = Some((seq, dim));
                            let mut m2 = m.clone(); m2.insert("Params".into(), Value::Assoc(params.clone())); current = Value::Assoc(m2);
                        }
                    }
                    "TransformerDecoder" => {
                        // Decoder: self-attn (causal optional) + cross-attn over Memory + FFN, with residuals + LN
                        let (seq, dim) = match (params.get("SeqLen").and_then(|v| as_i64(v)), params.get("ModelDim").and_then(|v| as_i64(v))) {
                            (Some(s), Some(d)) if s>0 && d>0 => (s as usize, d as usize),
                            _ => match curr_sd { Some((s,d)) => (s,d), None => (0,0) }
                        };
                        let heads = params.get("NumHeads").and_then(|v| as_i64(v)).unwrap_or(1).max(1) as usize;
                        if seq>0 && dim>0 && x_vec.len()==seq*dim && dim%heads==0 {
                            let head_dim = dim/heads;
                            // Helpers
                            let read_mat = |name: &str| -> Vec<f64> {
                                if let Some(Value::List(rows)) = params.get(name) {
                                    let mut out = Vec::with_capacity(dim*dim);
                                    for r in 0..dim { if let Some(Value::List(row)) = rows.get(r) { for c in 0..dim { out.push(as_f64(row.get(c).unwrap_or(&Value::Integer(0))).unwrap_or(0.0)); } } }
                                    return out;
                                }
                                vec![0.0; dim*dim]
                            };
                            let read_vec = |name: &str| -> Vec<f64> {
                                if let Some(Value::List(vs)) = params.get(name) { (0..dim).map(|i| as_f64(vs.get(i).unwrap_or(&Value::Integer(0))).unwrap_or(0.0)).collect() } else { vec![0.0; dim] }
                            };
                            // --- Self-attention (suffix 1) ---
                            let x0 = x_vec.clone();
                            let wq1 = read_mat("Wq1"); let wk1 = read_mat("Wk1"); let wv1 = read_mat("Wv1"); let wo1 = read_mat("Wo1");
                            let bq1 = read_vec("bq1"); let bk1 = read_vec("bk1"); let bv1 = read_vec("bv1"); let bo1 = read_vec("bo1");
                            let xmat = |r: usize, c: usize| -> f64 { x0[r*dim + c] };
                            let proj_x = |w: &Vec<f64>, b: &Vec<f64>| -> Vec<f64> {
                                let mut out = vec![0.0; seq*dim];
                                for s in 0..seq { for j in 0..dim { let mut acc=0.0; for k in 0..dim { acc += xmat(s,k) * w[k*dim + j]; } out[s*dim + j] = acc + b[j]; }}
                                out
                            };
                            let q1 = proj_x(&wq1, &bq1); let k1 = proj_x(&wk1, &bk1); let v1 = proj_x(&wv1, &bv1);
                            // Mask: prefer SelfMask; else Mask/Causal
                            let mut att_bias1: Option<Vec<f64>> = None;
                            let causal = params.get("Causal").and_then(|v| as_bool(v)).unwrap_or(false)
                                || matches!(params.get("Mask"), Some(Value::String(s)) if s.eq_ignore_ascii_case("Causal"))
                                || matches!(params.get("Mask"), Some(Value::Symbol(s)) if s.eq_ignore_ascii_case("Causal"));
                            if let Some(sm) = params.get("SelfMask") { // explicit mask wins
                                match sm {
                                    Value::List(rows) if !rows.is_empty() && matches!(rows[0], Value::List(_)) => {
                                        let mut b = vec![0.0f64; seq*seq];
                                        for i in 0..seq { if let Some(Value::List(r)) = rows.get(i) { for j in 0..seq { let v = r.get(j).cloned().unwrap_or(Value::Integer(1)); let x = as_f64(&v).unwrap_or_else(|| if matches!(v, Value::Boolean(false) | Value::Integer(0)) {0.0} else {1.0}); b[i*seq + j] = if x>0.0 { 0.0 } else { -1e9 }; } } }
                                        att_bias1 = Some(b);
                                    }
                                    Value::List(vs) if vs.len()==seq => {
                                        let mut b = vec![0.0f64; seq*seq];
                                        for j in 0..seq { let keep = as_f64(vs.get(j).unwrap_or(&Value::Integer(1))).unwrap_or(1.0) > 0.0; if !keep { for i in 0..seq { b[i*seq + j] = -1e9; } } }
                                        att_bias1 = Some(b);
                                    }
                                    _ => {}
                                }
                            } else if causal {
                                let mut b = vec![0.0f64; seq*seq];
                                for i in 0..seq { for j in 0..seq { if j>i { b[i*seq + j] = -1e9; } } }
                                att_bias1 = Some(b);
                            } else if let Some(mv) = params.get("Mask") {
                                match mv {
                                    Value::List(rows) if !rows.is_empty() && matches!(rows[0], Value::List(_)) => {
                                        let mut b = vec![0.0f64; seq*seq];
                                        for i in 0..seq { if let Some(Value::List(r)) = rows.get(i) { for j in 0..seq { let v = r.get(j).cloned().unwrap_or(Value::Integer(1)); let x = as_f64(&v).unwrap_or_else(|| if matches!(v, Value::Boolean(false) | Value::Integer(0)) {0.0} else {1.0}); b[i*seq + j] = if x>0.0 { 0.0 } else { -1e9 }; } } }
                                        att_bias1 = Some(b);
                                    }
                                    Value::List(vs) if vs.len()==seq => {
                                        let mut b = vec![0.0f64; seq*seq];
                                        for j in 0..seq { let keep = as_f64(vs.get(j).unwrap_or(&Value::Integer(1))).unwrap_or(1.0) > 0.0; if !keep { for i in 0..seq { b[i*seq + j] = -1e9; } } }
                                        att_bias1 = Some(b);
                                    }
                                    _ => {}
                                }
                            }
                            let scale = 1.0 / (head_dim as f64).sqrt();
                            let mut att1_out = vec![0.0; seq*dim];
                            for h in 0..heads {
                                let off = h*head_dim;
                                let mut att = vec![0.0; seq*seq];
                                for i in 0..seq { for j in 0..seq { let mut dot=0.0; for d0 in 0..head_dim { dot += q1[i*dim + off + d0] * k1[j*dim + off + d0]; } att[i*seq + j] = dot * scale; }}
                                for i in 0..seq {
                                    if let Some(b) = &att_bias1 { for j in 0..seq { att[i*seq + j] += b[i*seq + j]; } }
                                    let row = &mut att[i*seq..(i+1)*seq]; let m = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                                    let mut sum = 0.0; for x in row.iter_mut() { *x = (*x - m).exp(); sum += *x; }
                                    if sum>0.0 { for x in row.iter_mut() { *x /= sum; } }
                                }
                                for i in 0..seq { for d0 in 0..head_dim { let mut acc=0.0; for j in 0..seq { acc += att[i*seq + j] * v1[j*dim + off + d0]; } att1_out[i*dim + off + d0] = acc; }}
                            }
                            let mut y1 = vec![0.0; seq*dim];
                            for s in 0..seq { for j in 0..dim { let mut acc=0.0; for k0 in 0..dim { acc += att1_out[s*dim + k0] * wo1[k0*dim + j]; } y1[s*dim + j] = acc + bo1[j]; }}
                            for i in 0..(seq*dim) { y1[i] += x0[i]; }
                            let mut ln_params1 = HashMap::from([
                                ("InputChannels".into(), Value::Integer(dim as i64)),
                                ("Height".into(), Value::Integer(seq as i64)),
                                ("Width".into(), Value::Integer(1)),
                                ("Epsilon".into(), Value::Real(1e-5)),
                            ]);
                            apply_layernorm(&ln_params1, &mut y1);
                            // --- Cross-attention (suffix 2) ---
                            let mem_v = params.get("Memory");
                            let mut y2_in = y1.clone();
                            if let Some(mem) = mem_v {
                                if let Some((mseq, mdim)) = infer_seqdim(mem) {
                                    if mdim == dim {
                                        let mut mem_vec: Vec<f64> = Vec::with_capacity(mseq*mdim);
                                        if flatten_to_vec_f64(mem, &mut mem_vec) {
                                            let wq2 = read_mat("Wq2"); let wk2 = read_mat("Wk2"); let wv2 = read_mat("Wv2"); let wo2 = read_mat("Wo2");
                                            let bq2 = read_vec("bq2"); let bk2 = read_vec("bk2"); let bv2 = read_vec("bv2"); let bo2 = read_vec("bo2");
                                            let y1mat = |r: usize, c: usize| -> f64 { y1[r*dim + c] };
                                            let proj_y1 = |w: &Vec<f64>, b: &Vec<f64>| -> Vec<f64> { let mut out=vec![0.0; seq*dim]; for s in 0..seq { for j in 0..dim { let mut acc=0.0; for k in 0..dim { acc += y1mat(s,k) * w[k*dim + j]; } out[s*dim + j] = acc + b[j]; }} out };
                                            let q2 = proj_y1(&wq2, &bq2);
                                            let memmat = |r: usize, c: usize| -> f64 { mem_vec[r*dim + c] };
                                            let proj_mem = |w: &Vec<f64>, b: &Vec<f64>| -> Vec<f64> { let mut out=vec![0.0; mseq*dim]; for s in 0..mseq { for j in 0..dim { let mut acc=0.0; for k in 0..dim { acc += memmat(s,k) * w[k*dim + j]; } out[s*dim + j] = acc + b[j]; }} out };
                                            let k2 = proj_mem(&wk2, &bk2); let v2 = proj_mem(&wv2, &bv2);
                                            // Memory mask support: vector len mseq or matrix seq x mseq
                                            let mut att_bias2: Option<Vec<f64>> = None;
                                            if let Some(mmv) = params.get("MemoryMask").or_else(|| params.get("SourceMask")) {
                                                match mmv {
                                                    Value::List(rows) if !rows.is_empty() && matches!(rows[0], Value::List(_)) => {
                                                        let mut b = vec![0.0f64; seq*mseq];
                                                        for i in 0..seq { if let Some(Value::List(r)) = rows.get(i) { for j in 0..mseq { let v = r.get(j).cloned().unwrap_or(Value::Integer(1)); let x = as_f64(&v).unwrap_or_else(|| if matches!(v, Value::Boolean(false) | Value::Integer(0)) {0.0} else {1.0}); b[i*mseq + j] = if x>0.0 { 0.0 } else { -1e9 }; } } }
                                                        att_bias2 = Some(b);
                                                    }
                                                    Value::List(vs) if vs.len()==mseq => {
                                                        let mut b = vec![0.0f64; seq*mseq];
                                                        for j in 0..mseq { let keep = as_f64(vs.get(j).unwrap_or(&Value::Integer(1))).unwrap_or(1.0) > 0.0; if !keep { for i in 0..seq { b[i*mseq + j] = -1e9; } } }
                                                        att_bias2 = Some(b);
                                                    }
                                                    _ => {}
                                                }
                                            }
                                            let scale2 = 1.0 / (head_dim as f64).sqrt();
                                            let mut att2_out = vec![0.0; seq*dim];
                                            for h in 0..heads {
                                                let off = h*head_dim;
                                                let mut att = vec![0.0; seq*mseq];
                                                for i in 0..seq { for j in 0..mseq { let mut dot=0.0; for d0 in 0..head_dim { dot += q2[i*dim + off + d0] * k2[j*dim + off + d0]; } att[i*mseq + j] = dot * scale2; }}
                                                for i in 0..seq {
                                                    if let Some(b) = &att_bias2 { for j in 0..mseq { att[i*mseq + j] += b[i*mseq + j]; } }
                                                    let row = &mut att[i*mseq..(i+1)*mseq]; let m = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                                                    let mut sum = 0.0; for x in row.iter_mut() { *x = (*x - m).exp(); sum += *x; }
                                                    if sum>0.0 { for x in row.iter_mut() { *x /= sum; } }
                                                }
                                                for i in 0..seq { for d0 in 0..head_dim { let mut acc=0.0; for j in 0..mseq { acc += att[i*mseq + j] * v2[j*dim + off + d0]; } att2_out[i*dim + off + d0] = acc; }}
                                            }
                                            let mut y2 = vec![0.0; seq*dim];
                                            for s in 0..seq { for j in 0..dim { let mut acc=0.0; for k0 in 0..dim { acc += att2_out[s*dim + k0] * wo2[k0*dim + j]; } y2[s*dim + j] = acc + bo2[j]; }}
                                            for i in 0..(seq*dim) { y2[i] += y1[i]; }
                                            ln_params1.insert("InputChannels".into(), Value::Integer(dim as i64));
                                            apply_layernorm(&ln_params1, &mut y2);
                                            y2_in = y2;
                                        }
                                    }
                                }
                            }
                            // --- FFN --- (supports gated via FFNVariant)
                            let act = params.get("Activation").and_then(|v| match v { Value::String(s)|Value::Symbol(s)=>Some(s.clone()), _=>None }).unwrap_or("Gelu".into());
                            let w1 = if let Some(Value::List(rows)) = params.get("W1") { rows.clone() } else { vec![] };
                            let b1 = if let Some(Value::List(v)) = params.get("b1") { v.clone() } else { vec![] };
                            let wg = params.get("Wg");
                            let bg = params.get("bg");
                            let w2 = if let Some(Value::List(rows)) = params.get("W2") { rows.clone() } else { vec![] };
                            let b2 = if let Some(Value::List(v)) = params.get("b2") { v.clone() } else { vec![] };
                            let hidden = w1.len();
                            // y2_in: seq x dim -> hidden
                            let mut ffn1 = vec![0.0; seq*hidden.max(0)];
                            if hidden>0 {
                                // If FFNVariant present and Wg exists, use gated FFN; else normal FFN
                                if let Some(Value::String(var)) = params.get("FFNVariant").or_else(|| params.get("Variant")).cloned() {
                                    if wg.is_some() {
                                        // compute h1 and gate
                                        for s in 0..seq {
                                            for j in 0..hidden {
                                                let mut acc = 0.0; if let Some(Value::List(wrow)) = w1.get(j) { for k in 0..dim { acc += y2_in[s*dim + k] * as_f64(wrow.get(k).unwrap_or(&Value::Integer(0))).unwrap_or(0.0); } }
                                                let bj = as_f64(b1.get(j).unwrap_or(&Value::Integer(0))).unwrap_or(0.0);
                                                ffn1[s*hidden + j] = acc + bj;
                                            }
                                        }
                                        // gate linear
                                        let mut glin = vec![0.0; seq*hidden];
                                        if let (Some(Value::List(_)), Some(_)) = (wg, bg) {
                                            let p = HashMap::from([
                                                ("W".into(), wg.cloned().unwrap()),
                                                ("b".into(), bg.cloned().unwrap()),
                                            ]);
                                            for s in 0..seq {
                                                let inp = &y2_in[s*dim..(s+1)*dim];
                                                let out = apply_linear(&p, inp).unwrap_or_else(|| vec![0.0; hidden]);
                                                for j in 0..hidden { glin[s*hidden + j] = out[j]; }
                                            }
                                        }
                                        match var.to_ascii_lowercase().as_str() {
                                            "geglu" => apply_activation("Gelu", &mut glin),
                                            _ => { for v in glin.iter_mut() { let s1 = 1.0 / (1.0 + (-*v).exp()); *v = *v * s1; } },
                                        }
                                        for i in 0..(seq*hidden) { ffn1[i] *= glin[i]; }
                                    } else {
                                        // fallback non-gated
                                        for s in 0..seq { for j in 0..hidden { let mut acc=0.0; if let Some(Value::List(wrow)) = w1.get(j) { for k in 0..dim { acc += y2_in[s*dim + k]*as_f64(wrow.get(k).unwrap_or(&Value::Integer(0))).unwrap_or(0.0);} } let bj=as_f64(b1.get(j).unwrap_or(&Value::Integer(0))).unwrap_or(0.0); ffn1[s*hidden + j]=acc + bj; } }
                                        apply_activation(&act, &mut ffn1);
                                    }
                                } else {
                                    for s in 0..seq { for j in 0..hidden { let mut acc=0.0; if let Some(Value::List(wrow)) = w1.get(j) { for k in 0..dim { acc += y2_in[s*dim + k]*as_f64(wrow.get(k).unwrap_or(&Value::Integer(0))).unwrap_or(0.0);} } let bj=as_f64(b1.get(j).unwrap_or(&Value::Integer(0))).unwrap_or(0.0); ffn1[s*hidden + j]=acc + bj; } }
                                    apply_activation(&act, &mut ffn1);
                                }
                                // hidden -> dim
                                let mut ffn2 = vec![0.0; seq*dim];
                                for s in 0..seq { for j in 0..dim { let mut acc=0.0; if let Some(Value::List(wrow)) = w2.get(j) { for k in 0..hidden { acc += ffn1[s*hidden + k] * as_f64(wrow.get(k).unwrap_or(&Value::Integer(0))).unwrap_or(0.0); } } let bj = as_f64(b2.get(j).unwrap_or(&Value::Integer(0))).unwrap_or(0.0); ffn2[s*dim + j] = acc + bj; }}
                                for i in 0..(seq*dim) { ffn2[i] += y2_in[i]; }
                                let ln_params2 = HashMap::from([
                                    ("InputChannels".into(), Value::Integer(dim as i64)),
                                    ("Height".into(), Value::Integer(seq as i64)),
                                    ("Width".into(), Value::Integer(1)),
                                    ("Epsilon".into(), Value::Real(1e-5)),
                                ]);
                                apply_layernorm(&ln_params2, &mut ffn2);
                                x_vec = ffn2;
                            } else {
                                x_vec = y2_in;
                            }
                            in_dim = x_vec.len(); curr_sd = Some((seq, dim));
                            let mut m2 = m.clone(); m2.insert("Params".into(), Value::Assoc(params.clone())); current = Value::Assoc(m2);
                        }
                    }
                    "Pool2D" => {
                        if params.get("InputChannels").is_none() || params.get("Height").is_none() || params.get("Width").is_none() {
                            if let Some((cin0,h0,w0)) = curr_chw { params.insert("InputChannels".into(), Value::Integer(cin0 as i64)); params.insert("Height".into(), Value::Integer(h0 as i64)); params.insert("Width".into(), Value::Integer(w0 as i64)); }
                        }
                        if let (Some(cin0), Some(h0), Some(w0)) = (params.get("InputChannels").and_then(|v| as_i64(v)), params.get("Height").and_then(|v| as_i64(v)), params.get("Width").and_then(|v| as_i64(v))) {
                            if let Some(limit) = max_ops_limit() {
                                let (kh, kw) = parse_hw(assoc_get(&params, "KernelSize")).unwrap_or((2,2));
                                let (sh, sw) = parse_hw(assoc_get(&params, "Stride")).unwrap_or((kh,kw));
                                let h = h0 as usize; let w = w0 as usize;
                                let oh = if h >= kh { ((h - kh) / sh.max(1)) + 1 } else { 0 };
                                let ow = if w >= kw { ((w - kw) / sw.max(1)) + 1 } else { 0 };
                                let ops = (cin0.max(0) as usize).saturating_mul(oh).saturating_mul(ow).saturating_mul(kh).saturating_mul(kw);
                                if ops > limit { reg().lock().unwrap().insert(id, state); return error_assoc("Pooling2D exceeds LYRA_NN_MAX_OPS"); }
                            }
                        }
                        if let Some(y) = apply_pool2d(&params, &x_vec) {
                            if let (Some(cin0), Some(h0), Some(w0)) = (params.get("InputChannels").and_then(|v| as_i64(v)), params.get("Height").and_then(|v| as_i64(v)), params.get("Width").and_then(|v| as_i64(v))) {
                                let (kh, kw) = parse_hw(assoc_get(&params, "KernelSize")).unwrap_or((2,2));
                                let (sh, sw) = parse_hw(assoc_get(&params, "Stride")).unwrap_or((kh,kw));
                                let h = h0 as usize; let w = w0 as usize; let cin = cin0 as usize;
                                let oh = if h >= kh { ((h - kh) / sh.max(1)) + 1 } else { 0 };
                                let ow = if w >= kw { ((w - kw) / sw.max(1)) + 1 } else { 0 };
                                curr_chw = Some((cin, oh, ow));
                            }
                            x_vec = y;
                            in_dim = x_vec.len();
                            let mut m2 = m.clone(); m2.insert("Params".into(), Value::Assoc(params.clone())); current = Value::Assoc(m2);
                        }
                    }
                    "DepthwiseConv2D" => {
                        if params.get("InputChannels").is_none() || params.get("Height").is_none() || params.get("Width").is_none() {
                            if let Some((cin0,h0,w0)) = curr_chw { params.insert("InputChannels".into(), Value::Integer(cin0 as i64)); params.insert("Height".into(), Value::Integer(h0 as i64)); params.insert("Width".into(), Value::Integer(w0 as i64)); }
                        }
                        if let (Some(cin0), Some(h0), Some(w0)) = (params.get("InputChannels").and_then(|v| as_i64(v)), params.get("Height").and_then(|v| as_i64(v)), params.get("Width").and_then(|v| as_i64(v))) {
                            if let Some(limit) = max_ops_limit() {
                                let (kh, kw) = parse_hw(assoc_get(&params, "KernelSize")).unwrap_or((3,3));
                                let (sh, sw) = parse_hw(assoc_get(&params, "Stride")).unwrap_or((1,1));
                                let (ph, pw) = parse_hw(assoc_get(&params, "Padding")).unwrap_or((0,0));
                                let h = h0 as usize; let w = w0 as usize;
                                let oh = if h + 2*ph >= kh { ((h + 2*ph - kh) / sh.max(1)) + 1 } else { 0 };
                                let ow = if w + 2*pw >= kw { ((w + 2*pw - kw) / sw.max(1)) + 1 } else { 0 };
                                let ops = (cin0.max(0) as usize).saturating_mul(oh).saturating_mul(ow).saturating_mul(kh).saturating_mul(kw);
                                if ops > limit { reg().lock().unwrap().insert(id, state); return error_assoc("DepthwiseConv2D exceeds LYRA_NN_MAX_OPS"); }
                            }
                        }
                        if let Some(y) = apply_depthwise_conv2d(&params, &x_vec) {
                            if let (Some(cin0), Some(h0), Some(w0)) = (params.get("InputChannels").and_then(|v| as_i64(v)), params.get("Height").and_then(|v| as_i64(v)), params.get("Width").and_then(|v| as_i64(v))) {
                                let (kh, kw) = parse_hw(assoc_get(&params, "KernelSize")).unwrap_or((3,3));
                                let (sh, sw) = parse_hw(assoc_get(&params, "Stride")).unwrap_or((1,1));
                                let (ph, pw) = parse_hw(assoc_get(&params, "Padding")).unwrap_or((0,0));
                                let h = h0 as usize; let w = w0 as usize; let cin = cin0 as usize;
                                let oh = if h + 2*ph >= kh { ((h + 2*ph - kh) / sh.max(1)) + 1 } else { 0 };
                                let ow = if w + 2*pw >= kw { ((w + 2*pw - kw) / sw.max(1)) + 1 } else { 0 };
                                curr_chw = Some((cin, oh, ow));
                            }
                            x_vec = y; in_dim = x_vec.len();
                            let mut m2 = m.clone(); m2.insert("Params".into(), Value::Assoc(params.clone())); current = Value::Assoc(m2);
                        }
                    }
                    "ConvTranspose2D" => {
                        if params.get("InputChannels").is_none() || params.get("Height").is_none() || params.get("Width").is_none() {
                            if let Some((cin0,h0,w0)) = curr_chw { params.insert("InputChannels".into(), Value::Integer(cin0 as i64)); params.insert("Height".into(), Value::Integer(h0 as i64)); params.insert("Width".into(), Value::Integer(w0 as i64)); }
                        }
                        if let (Some(cin0), Some(h0), Some(w0)) = (params.get("InputChannels").and_then(|v| as_i64(v)), params.get("Height").and_then(|v| as_i64(v)), params.get("Width").and_then(|v| as_i64(v))) {
                            if let Some(limit) = max_ops_limit() {
                                let (kh, kw) = parse_hw(assoc_get(&params, "KernelSize")).unwrap_or((3,3));
                                let (sh, sw) = parse_hw(assoc_get(&params, "Stride")).unwrap_or((1,1));
                                let (ph, pw) = parse_hw(assoc_get(&params, "Padding")).unwrap_or((0,0));
                                let cout = params.get("Output").and_then(|v| as_i64(v)).unwrap_or(1).max(1) as usize;
                                let h = h0 as usize; let w = w0 as usize;
                                let oh = (h.saturating_sub(1)).saturating_mul(sh).saturating_add(kh).saturating_sub(2*ph);
                                let ow = (w.saturating_sub(1)).saturating_mul(sw).saturating_add(kw).saturating_sub(2*pw);
                                let ops = cout.saturating_mul(oh).saturating_mul(ow).saturating_mul(cin0.max(0) as usize).saturating_mul(kh).saturating_mul(kw);
                                if ops > limit { reg().lock().unwrap().insert(id, state); return error_assoc("ConvTranspose2D exceeds LYRA_NN_MAX_OPS"); }
                            }
                        }
                        if let Some(y) = apply_conv_transpose2d(&params, &x_vec) {
                            if let (Some(_cin0), Some(h0), Some(w0)) = (params.get("InputChannels").and_then(|v| as_i64(v)), params.get("Height").and_then(|v| as_i64(v)), params.get("Width").and_then(|v| as_i64(v))) {
                                let (kh, kw) = parse_hw(assoc_get(&params, "KernelSize")).unwrap_or((3,3));
                                let (sh, sw) = parse_hw(assoc_get(&params, "Stride")).unwrap_or((1,1));
                                let (ph, pw) = parse_hw(assoc_get(&params, "Padding")).unwrap_or((0,0));
                                let cout = params.get("Output").and_then(|v| as_i64(v)).unwrap_or(1).max(1) as usize;
                                let h = h0 as usize; let w = w0 as usize;
                                let oh = (h - 1) * sh + kh - 2*ph; let ow = (w - 1) * sw + kw - 2*pw;
                                curr_chw = Some((cout, oh, ow));
                            }
                            x_vec = y; in_dim = x_vec.len();
                            let mut m2 = m.clone(); m2.insert("Params".into(), Value::Assoc(params.clone())); current = Value::Assoc(m2);
                        }
                    }
                    "GlobalAvgPool2D" => {
                        if params.get("InputChannels").is_none() || params.get("Height").is_none() || params.get("Width").is_none() {
                            if let Some((cin0,h0,w0)) = curr_chw { params.insert("InputChannels".into(), Value::Integer(cin0 as i64)); params.insert("Height".into(), Value::Integer(h0 as i64)); params.insert("Width".into(), Value::Integer(w0 as i64)); }
                        }
                        if let Some((cin0, h0, w0)) = curr_chw {
                            let cin = cin0 as usize; let h = h0 as usize; let w = w0 as usize;
                            // average per channel
                            let mut outv = vec![0.0f64; cin];
                            for c in 0..cin { let mut s=0.0; for i in 0..(h*w) { s+= x_vec[c*h*w + i]; } outv[c]= if h*w>0 { s/((h*w) as f64) } else {0.0}; }
                            x_vec = outv; in_dim = x_vec.len(); curr_chw = Some((cin,1,1));
                        }
                    }
                    "BatchNorm" => {
                        if params.get("InputChannels").is_none() || params.get("Height").is_none() || params.get("Width").is_none() {
                            if let Some((cin0,h0,w0)) = curr_chw { params.insert("InputChannels".into(), Value::Integer(cin0 as i64)); params.insert("Height".into(), Value::Integer(h0 as i64)); params.insert("Width".into(), Value::Integer(w0 as i64)); }
                        }
                        apply_batchnorm(&params, &mut x_vec);
                        let mut m2 = m.clone(); m2.insert("Params".into(), Value::Assoc(params.clone())); current = Value::Assoc(m2);
                    }
                    "LayerNorm" => {
                        if params.get("InputChannels").is_none() || params.get("Height").is_none() || params.get("Width").is_none() {
                            if let Some((cin0,h0,w0)) = curr_chw { params.insert("InputChannels".into(), Value::Integer(cin0 as i64)); params.insert("Height".into(), Value::Integer(h0 as i64)); params.insert("Width".into(), Value::Integer(w0 as i64)); }
                        }
                        apply_layernorm(&params, &mut x_vec);
                        let mut m2 = m.clone(); m2.insert("Params".into(), Value::Assoc(params.clone())); current = Value::Assoc(m2);
                    }
                    "RMSNorm" => {
                        // Prefer seq x dim; fallback to 1D
                        let (seq, dim) = match (params.get("SeqLen").and_then(|v| as_i64(v)), params.get("ModelDim").and_then(|v| as_i64(v))) {
                            (Some(s), Some(d)) if s>0 && d>0 => (s as usize, d as usize),
                            _ => match curr_sd { Some((s,d)) => (s,d), None => (0,0) }
                        };
                        let eps = assoc_get(&params, "Epsilon").and_then(|v| as_f64(v)).unwrap_or(1e-5);
                        if seq>0 && dim>0 && x_vec.len()==seq*dim {
                            let gamma = assoc_get(&params, "Gamma");
                            let beta = assoc_get(&params, "Beta");
                            for s in 0..seq {
                                let row = &mut x_vec[s*dim..(s+1)*dim];
                                let mut ms = 0.0; for i in 0..dim { let v = row[i]; ms += v*v; }
                                ms /= dim as f64; let inv = 1.0 / (ms + eps).sqrt();
                                for i in 0..dim {
                                    let g = match gamma { Some(Value::List(gs)) => gs.get(i).and_then(|v| as_f64(v)).unwrap_or(1.0), Some(Value::Real(r)) => *r, Some(Value::Integer(n)) => *n as f64, _ => 1.0 };
                                    let b = match beta { Some(Value::List(bs)) => bs.get(i).and_then(|v| as_f64(v)).unwrap_or(0.0), Some(Value::Real(r)) => *r, Some(Value::Integer(n)) => *n as f64, _ => 0.0 };
                                    row[i] = row[i] * inv * g + b;
                                }
                            }
                            curr_sd = Some((seq, dim));
                        } else {
                            // 1D fallback
                            let n = x_vec.len();
                            let mut ms = 0.0; for v in &x_vec { ms += v*v; } ms /= n.max(1) as f64;
                            let inv = 1.0 / (ms + eps).sqrt();
                            let g = assoc_get(&params, "Gamma").and_then(|v| as_f64(v)).unwrap_or(1.0);
                            let b = assoc_get(&params, "Beta").and_then(|v| as_f64(v)).unwrap_or(0.0);
                            for i in 0..n { x_vec[i] = x_vec[i]*inv * g + b; }
                        }
                        let mut m2 = m.clone(); m2.insert("Params".into(), Value::Assoc(params.clone())); current = Value::Assoc(m2);
                    }
                    "FFN" => {
                        // seq x dim; computes gated FFN
                        let (seq, dim) = match (params.get("SeqLen").and_then(|v| as_i64(v)), params.get("ModelDim").and_then(|v| as_i64(v))) {
                            (Some(s), Some(d)) if s>0 && d>0 => (s as usize, d as usize),
                            _ => match curr_sd { Some((s,d)) => (s,d), None => (0,0) }
                        };
                        if seq>0 && dim>0 && x_vec.len()==seq*dim {
                            let w1 = match params.get("W1") { Some(Value::List(r)) => r.clone(), _=>vec![] };
                            let b1 = match params.get("b1") { Some(Value::List(v)) => v.clone(), _=>vec![] };
                            let wg = match params.get("Wg") { Some(Value::List(r)) => r.clone(), _=>vec![] };
                            let bg = match params.get("bg") { Some(Value::List(v)) => v.clone(), _=>vec![] };
                            let w2 = match params.get("W2") { Some(Value::List(r)) => r.clone(), _=>vec![] };
                            let b2 = match params.get("b2") { Some(Value::List(v)) => v.clone(), _=>vec![] };
                            let hidden = w1.len();
                            let variant = params.get("Variant").and_then(|v| match v { Value::String(s)|Value::Symbol(s)=>Some(s.clone()), _=>None }).unwrap_or("SwiGLU".into());
                            let x = x_vec.clone();
                            let mut vlin = vec![0.0; seq*hidden];
                            let mut glin = vec![0.0; seq*hidden];
                            for s in 0..seq {
                                for j in 0..hidden {
                                    // v path
                                    let mut acc1 = 0.0; if let Some(Value::List(wrow)) = w1.get(j) { for k in 0..dim { acc1 += x[s*dim + k] * as_f64(wrow.get(k).unwrap_or(&Value::Integer(0))).unwrap_or(0.0); } }
                                    let b1j = as_f64(b1.get(j).unwrap_or(&Value::Integer(0))).unwrap_or(0.0);
                                    vlin[s*hidden + j] = acc1 + b1j;
                                    // gate path
                                    let mut accg = 0.0; if let Some(Value::List(wrow)) = wg.get(j) { for k in 0..dim { accg += x[s*dim + k] * as_f64(wrow.get(k).unwrap_or(&Value::Integer(0))).unwrap_or(0.0); } }
                                    let bgj = as_f64(bg.get(j).unwrap_or(&Value::Integer(0))).unwrap_or(0.0);
                                    glin[s*hidden + j] = accg + bgj;
                                }
                            }
                            match variant.to_ascii_lowercase().as_str() {
                                "geglu" => { apply_activation("Gelu", &mut glin); },
                                _ => { // swiglu
                                    // SiLU: x * sigmoid(x)
                                    for v in glin.iter_mut() { let s = 1.0 / (1.0 + (-*v).exp()); *v = *v * s; }
                                }
                            }
                            for i in 0..(seq*hidden) { vlin[i] *= glin[i]; }
                            let mut y = vec![0.0; seq*dim];
                            for s in 0..seq { for j in 0..dim { let mut acc=0.0; if let Some(Value::List(wrow)) = w2.get(j) { for k in 0..hidden { acc += vlin[s*hidden + k] * as_f64(wrow.get(k).unwrap_or(&Value::Integer(0))).unwrap_or(0.0); } } let bj = as_f64(b2.get(j).unwrap_or(&Value::Integer(0))).unwrap_or(0.0); y[s*dim + j] = acc + bj; }}
                            x_vec = y; curr_sd = Some((seq, dim)); in_dim = x_vec.len();
                            let mut m2 = m.clone(); m2.insert("Params".into(), Value::Assoc(params.clone())); current = Value::Assoc(m2);
                        }
                    },
                    // CausalSelfAttention constructor aliases to MultiHeadAttention; handled in that branch
                    "CrossAttention" => {
                        // Cross-attention: Q from input (seq x dim), K/V from Memory (mseq x dim)
                        let (seq, dim) = match (params.get("SeqLen").and_then(|v| as_i64(v)), params.get("ModelDim").and_then(|v| as_i64(v))) {
                            (Some(s), Some(d)) if s>0 && d>0 => (s as usize, d as usize),
                            _ => match curr_sd { Some((s,d)) => (s,d), None => (0,0) }
                        };
                        let heads = params.get("NumHeads").and_then(|v| as_i64(v)).unwrap_or(1).max(1) as usize;
                        if seq>0 && dim>0 && x_vec.len()==seq*dim && dim%heads==0 {
                            if let Some(mem) = params.get("Memory") {
                                if let Some((mseq, mdim)) = infer_seqdim(mem) {
                                    if mdim == dim {
                                        let mut mem_vec: Vec<f64> = Vec::with_capacity(mseq*mdim);
                                        if !flatten_to_vec_f64(mem, &mut mem_vec) { mem_vec = vec![0.0; mseq*mdim]; }
                                        // Read params
                                        let read_mat = |name: &str| -> Vec<f64> {
                                            if let Some(Value::List(rows)) = params.get(name) {
                                                let mut out = Vec::with_capacity(dim*dim);
                                                for r in 0..dim {
                                                    if let Some(Value::List(row)) = rows.get(r) {
                                                        for c in 0..dim { out.push(as_f64(row.get(c).unwrap_or(&Value::Integer(0))).unwrap_or(0.0)); }
                                                    }
                                                }
                                                return out;
                                            }
                                            vec![0.0; dim*dim]
                                        };
                                        let read_vec = |name: &str| -> Vec<f64> {
                                            if let Some(Value::List(vs)) = params.get(name) {
                                                (0..dim).map(|i| as_f64(vs.get(i).unwrap_or(&Value::Integer(0))).unwrap_or(0.0)).collect()
                                            } else { vec![0.0; dim] }
                                        };
                                        let wq = read_mat("Wq"); let wk = read_mat("Wk"); let wv = read_mat("Wv"); let wo = read_mat("Wo");
                                        let bq = read_vec("bq"); let bk = read_vec("bk"); let bv = read_vec("bv"); let bo = read_vec("bo");
                                        // Projections
                                        let x0 = x_vec.clone();
                                        let xmat = |r: usize, c: usize| -> f64 { x0[r*dim + c] };
                                        let memat = |r: usize, c: usize| -> f64 { mem_vec[r*dim + c] };
                                        let proj_x = |w: &Vec<f64>, b: &Vec<f64>| -> Vec<f64> {
                                            let mut out = vec![0.0; seq*dim];
                                            for s in 0..seq { for j in 0..dim { let mut acc=0.0; for k in 0..dim { acc += xmat(s,k) * w[k*dim + j]; } out[s*dim + j] = acc + b[j]; }}
                                            out
                                        };
                                        let proj_m = |w: &Vec<f64>, b: &Vec<f64>| -> Vec<f64> {
                                            let mut out = vec![0.0; mseq*dim];
                                            for s in 0..mseq { for j in 0..dim { let mut acc=0.0; for k in 0..dim { acc += memat(s,k) * w[k*dim + j]; } out[s*dim + j] = acc + b[j]; }}
                                            out
                                        };
                                        let q = proj_x(&wq, &bq); let k = proj_m(&wk, &bk); let v = proj_m(&wv, &bv);
                                        // Optional MemoryMask / SourceMask
                                        let mut att_bias: Option<Vec<f64>> = None; // shape: seq x mseq
                                        if let Some(mmv) = params.get("MemoryMask").or_else(|| params.get("SourceMask")) {
                                            match mmv {
                                                Value::List(rows) if !rows.is_empty() && matches!(rows[0], Value::List(_)) => {
                                                    let mut b = vec![0.0f64; seq*mseq];
                                                    for i in 0..seq {
                                                        if let Some(Value::List(r)) = rows.get(i) {
                                                            for j in 0..mseq {
                                                                let v = r.get(j).cloned().unwrap_or(Value::Integer(1));
                                                                let x = as_f64(&v).unwrap_or_else(|| if matches!(v, Value::Boolean(false) | Value::Integer(0)) {0.0} else {1.0});
                                                                b[i*mseq + j] = if x>0.0 { 0.0 } else { -1e9 };
                                                            }
                                                        }
                                                    }
                                                    att_bias = Some(b);
                                                }
                                                Value::List(vs) if vs.len()==mseq => {
                                                    let mut b = vec![0.0f64; seq*mseq];
                                                    for j in 0..mseq {
                                                        let keep = as_f64(vs.get(j).unwrap_or(&Value::Integer(1))).unwrap_or(1.0) > 0.0;
                                                        if !keep { for i in 0..seq { b[i*mseq + j] = -1e9; } }
                                                    }
                                                    att_bias = Some(b);
                                                }
                                                _ => {}
                                            }
                                        }
                                        // Attention
                                        let head_dim = dim/heads; let scale = 1.0 / (head_dim as f64).sqrt();
                                        let mut att_out = vec![0.0; seq*dim];
                                        for h in 0..heads {
                                            let off = h*head_dim;
                                            let mut att = vec![0.0; seq*mseq];
                                            for i in 0..seq {
                                                for j in 0..mseq {
                                                    let mut dot = 0.0; for d0 in 0..head_dim { dot += q[i*dim + off + d0] * k[j*dim + off + d0]; }
                                                    att[i*mseq + j] = dot * scale;
                                                    if let Some(b) = &att_bias { att[i*mseq + j] += b[i*mseq + j]; }
                                                }
                                            }
                                            for i in 0..seq {
                                                let row = &mut att[i*mseq..(i+1)*mseq];
                                                let m = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                                                let mut s = 0.0; for x in row.iter_mut() { *x = (*x - m).exp(); s += *x; }
                                                if s>0.0 { for x in row.iter_mut() { *x /= s; } }
                                            }
                                            for i in 0..seq {
                                                for d0 in 0..head_dim {
                                                    let mut acc = 0.0; for j in 0..mseq { acc += att[i*mseq + j] * v[j*dim + off + d0]; }
                                                    att_out[i*dim + off + d0] = acc;
                                                }
                                            }
                                        }
                                        // Output projection
                                        let mut y = vec![0.0; seq*dim];
                                        for s in 0..seq { for j in 0..dim { let mut acc=0.0; for k0 in 0..dim { acc += att_out[s*dim + k0] * wo[k0*dim + j]; } y[s*dim + j] = acc + bo[j]; }}
                                        x_vec = y; in_dim = x_vec.len(); curr_sd = Some((seq, dim));
                                        let mut m2 = m.clone(); m2.insert("Params".into(), Value::Assoc(params.clone())); current = Value::Assoc(m2);
                                    }
                                }
                            }
                        }
                    },
                    "PositionalEmbedding" => {
                        let (seq, dim) = match (params.get("SeqLen").and_then(|v| as_i64(v)), params.get("ModelDim").and_then(|v| as_i64(v))) {
                            (Some(s), Some(d)) if s>0 && d>0 => (s as usize, d as usize),
                            _ => match curr_sd { Some((s,d)) => (s,d), None => (0,0) }
                        };
                        if seq>0 && dim>0 && x_vec.len()==seq*dim {
                            if let Some(Value::List(rows)) = params.get("P") {
                                for s in 0..seq {
                                    if let Some(Value::List(r)) = rows.get(s) {
                                        for j in 0..dim {
                                            x_vec[s*dim + j] += as_f64(r.get(j).unwrap_or(&Value::Integer(0))).unwrap_or(0.0);
                                        }
                                    }
                                }
                            }
                            curr_sd = Some((seq, dim));
                            in_dim = x_vec.len();
                            let mut m2 = m.clone();
                            m2.insert("Params".into(), Value::Assoc(params.clone()));
                            current = Value::Assoc(m2);
                        }
                    },
                    "PatchEmbedding2D" => {
                        if params.get("InputChannels").is_none() || params.get("Height").is_none() || params.get("Width").is_none() { if let Some((cin0,h0,w0)) = curr_chw { params.insert("InputChannels".into(), Value::Integer(cin0 as i64)); params.insert("Height".into(), Value::Integer(h0 as i64)); params.insert("Width".into(), Value::Integer(w0 as i64)); } }
                        if let (Some(cin0), Some(h0), Some(w0)) = (params.get("InputChannels").and_then(|v| as_i64(v)), params.get("Height").and_then(|v| as_i64(v)), params.get("Width").and_then(|v| as_i64(v))) {
                            let cin = cin0.max(1) as usize; let h = h0.max(1) as usize; let w = w0.max(1) as usize;
                            let d = params.get("ModelDim").and_then(|v| as_i64(v)).unwrap_or(1).max(1) as usize;
                            let (ph, pw) = parse_hw(assoc_get(&params, "PatchSize")).unwrap_or((16,16));
                            let oh = h / ph; let ow = w / pw; if oh>0 && ow>0 {
                                // W[d][cin][ph][pw], b[d]
                                let wts = params.get("W"); let bias = params.get("b");
                                let mut out = vec![0.0; oh*ow*d];
                                // iterate patches
                                for pi in 0..oh { for pj in 0..ow {
                                    let patch_idx = pi*ow + pj;
                                    for oc in 0..d {
                                        let mut acc = 0.0;
                                        for ic in 0..cin { for ii in 0..ph { for jj in 0..pw {
                                            // input index: x_vec in CHW layout
                                            let ih = pi*ph + ii; let jw = pj*pw + jj; let src = ic*h*w + ih*w + jw;
                                            let xval = x_vec.get(src).cloned().unwrap_or(0.0);
                                            // weight
                                            let wval = if let Some(Value::List(oc_list)) = wts { if let Some(Value::List(ic_list)) = oc_list.get(oc) { if let Some(Value::List(rows)) = ic_list.get(ic) { if let Some(Value::List(row)) = rows.get(ii) { as_f64(row.get(jj).unwrap_or(&Value::Integer(0))).unwrap_or(0.0) } else { 0.0 } } else { 0.0 } } else { 0.0 } } else { 0.0 };
                                            acc += xval * wval;
                                        }}}
                                        let b = match bias { Some(Value::List(bs)) => as_f64(bs.get(oc).unwrap_or(&Value::Integer(0))).unwrap_or(0.0), _=>0.0 };
                                        out[patch_idx*d + oc] = acc + b;
                                    }
                                }}
                                x_vec = out; in_dim = x_vec.len(); curr_sd = Some((oh*ow, d)); curr_chw=None;
                                let mut m2=m.clone(); m2.insert("Params".into(), Value::Assoc(params.clone())); current=Value::Assoc(m2);
                            }
                        }
                    }
                    "Activation" => {
                        let kind = params
                            .get("Type")
                            .and_then(|v| match v {
                                Value::String(s) | Value::Symbol(s) => Some(s.clone()),
                                _ => None,
                            })
                            .unwrap_or("ReLU".into());
                        apply_activation(&kind, &mut x_vec);
                    }
                    "Flatten" => {
                        // no-op: we already flattened input when needed
                        curr_chw = None;
                        force_1d_tensor = true;
                    }
                    "Dropout" => {
                        // inference-time no-op
                    }
                    "Reshape" => {
                        // store shape for final output reconstruction
                        if let Some(Value::List(xs)) = params.get("Shape") {
                            let mut shp: Vec<usize> = Vec::new();
                            for v in xs {
                                if let Some(n) = as_i64(v) {
                                    shp.push(n.max(0) as usize);
                                }
                            }
                            if !shp.is_empty() {
                                pending_shape = Some(shp);
                                curr_chw = None;
                            }
                        }
                    }
                    "Embedding" => {
                        // Only support as first layer on integer input
                        if idx == 0 {
                            let token = match &x_in {
                                Value::Integer(n) => Some(*n),
                                _ => None,
                            };
                            if let Some(tok) = token {
                                if let Some(dim) =
                                    as_i64(params.get("Dim").unwrap_or(&Value::Integer(16)))
                                {
                                    let d = dim.max(1) as usize;
                                    let mut seed = (id as u64) ^ 0xDEADBEEF ^ ((tok as u64) << 17);
                                    let mut out = vec![0.0f64; d];
                                    for i in 0..d {
                                        out[i] = lcg(&mut seed);
                                    }
                                    x_vec = out;
                                    in_dim = x_vec.len();
                                }
                            }
                        }
                    }
                    _ => {}
                }
            }
            new_layers.push(current);
        }
        state.layers = new_layers;
        // put back updated state
        reg().lock().unwrap().insert(id, state);
        // Prefer returning Tensor (PackedArray) when we know a shape or input was a tensor
        if let Some((c,h,w)) = curr_chw {
            return Value::PackedArray { shape: vec![c,h,w], data: x_vec };
        }
        if force_1d_tensor {
            return Value::PackedArray { shape: vec![x_vec.len()], data: x_vec };
        }
        if let Some(shape) = pending_shape {
            let prod: usize = shape.iter().product();
            if prod == x_vec.len() {
                return Value::PackedArray { shape, data: x_vec };
            } else {
                return build_shape_nested(&x_vec, &shape);
            }
        }
        if matches!(x_in, Value::PackedArray { .. }) {
            return Value::PackedArray { shape: vec![x_vec.len()], data: x_vec };
        }
        return from_vec_like(&x_in, &x_vec);
    }
    // Unknown net kind: return input
    reg().lock().unwrap().insert(id, state);
    x_in
}

// --------- properties/summary ---------

fn net_property(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() < 2 {
        return Value::Expr { head: Box::new(Value::Symbol("NetProperty".into())), args };
    }
    let net_v = ev.eval(args[0].clone());
    let prop = ev.eval(args[1].clone());
    let id = match get_net_id(&net_v) {
        Some(id) => id,
        None => return Value::Expr { head: Box::new(Value::Symbol("NetProperty".into())), args },
    };
    let st = match reg().lock().unwrap().get(&id) {
        Some(s) => s.clone(),
        None => return Value::Expr { head: Box::new(Value::Symbol("NetProperty".into())), args },
    };
    match prop {
        Value::String(s) | Value::Symbol(s) => match s.as_str() {
            "Properties" => Value::List(
                vec![
                    "Kind",
                    "Layers",
                    "Graph",
                    "Initialized",
                    "Method",
                    "Epochs",
                    "BatchSize",
                    "Encoder",
                    "Decoder",
                    "LayerSummaries",
                ]
                .into_iter()
                .map(|k| Value::String(k.into()))
                .collect(),
            ),
            "Kind" => Value::String(st.kind),
            "Layers" => Value::List(st.layers),
            "Graph" => st.graph.unwrap_or(Value::Symbol("Null".into())),
            "Initialized" => Value::Boolean(st.initialized),
            "Method" => Value::String(st.method),
            "Epochs" => Value::Integer(st.trained_epochs as i64),
            "BatchSize" => Value::Integer(st.batch_size as i64),
            "Encoder" => st.encoder.unwrap_or(Value::Symbol("Automatic".into())),
            "Decoder" => st.decoder.unwrap_or(Value::Symbol("Automatic".into())),
            "LayerSummaries" => {
                let mut out: Vec<Value> = Vec::with_capacity(st.layers.len());
                for l in &st.layers {
                    out.push(layer_summary(l));
                }
                Value::List(out)
            }
            _ => Value::Symbol("Null".into()),
        },
        _ => Value::Symbol("Null".into()),
    }
}

fn net_summary(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() {
        return Value::Expr { head: Box::new(Value::Symbol("NetSummary".into())), args };
    }
    let net_v = ev.eval(args[0].clone());
    let id = match get_net_id(&net_v) {
        Some(id) => id,
        None => return Value::Expr { head: Box::new(Value::Symbol("NetSummary".into())), args },
    };
    let st = match reg().lock().unwrap().get(&id) {
        Some(s) => s.clone(),
        None => return Value::Expr { head: Box::new(Value::Symbol("NetSummary".into())), args },
    };
    Value::Assoc(HashMap::from([
        ("Kind".into(), Value::String(st.kind)),
        ("LayerCount".into(), Value::Integer(st.layers.len() as i64)),
        ("Initialized".into(), Value::Boolean(st.initialized)),
        ("Method".into(), Value::String(st.method)),
        ("Epochs".into(), Value::Integer(st.trained_epochs as i64)),
    ]))
}

fn layer_summary(layer: &Value) -> Value {
    if let Value::Assoc(m) = layer {
        let ltype = m
            .get("LayerType")
            .and_then(|v| match v {
                Value::String(s) | Value::Symbol(s) => Some(s.clone()),
                _ => None,
            })
            .unwrap_or("Unknown".into());
        let params = m
            .get("Params")
            .and_then(|v| if let Value::Assoc(p) = v { Some(p.clone()) } else { None })
            .unwrap_or_default();
        let mut out: HashMap<String, Value> = HashMap::new();
        out.insert("LayerType".into(), Value::String(ltype.clone()));
        match ltype.as_str() {
            "Linear" => {
                if let Some(v) = params.get("Output") {
                    out.insert("Output".into(), v.clone());
                }
                if let Some(v) = params.get("Bias") {
                    out.insert("Bias".into(), v.clone());
                }
            }
            "Activation" => {
                if let Some(v) = params.get("Type") {
                    out.insert("Type".into(), v.clone());
                }
            }
            "Convolution" => {
                for k in ["Output", "KernelSize", "Stride", "Padding"] {
                    if let Some(v) = params.get(k) {
                        out.insert(k.into(), v.clone());
                    }
                }
            }
            "Conv2D" => {
                for k in ["Output", "KernelSize", "Stride", "Padding", "InputChannels", "Height", "Width"] {
                    if let Some(v) = params.get(k) { out.insert(k.into(), v.clone()); }
                }
            }
            "Pooling" => {
                for k in ["PoolType", "Size", "Stride"] {
                    if let Some(v) = params.get(k) {
                        out.insert(k.into(), v.clone());
                    }
                }
            }
            "Pool2D" => {
                for k in ["PoolType", "KernelSize", "Stride", "InputChannels", "Height", "Width"] {
                    if let Some(v) = params.get(k) { out.insert(k.into(), v.clone()); }
                }
            }
            "BatchNorm" | "LayerNorm" => {
                if let Some(v) = params.get("Epsilon") {
                    out.insert("Epsilon".into(), v.clone());
                }
            }
            "Embedding" => {
                for k in ["Vocab", "Dim"] {
                    if let Some(v) = params.get(k) {
                        out.insert(k.into(), v.clone());
                    }
                }
            }
            "Reshape" => {
                if let Some(v) = params.get("Shape") {
                    out.insert("Shape".into(), v.clone());
                }
            }
            _ => {}
        }
        return Value::Assoc(out);
    }
    Value::assoc(vec![("LayerType", Value::String("Unknown".into()))])
}

// --------- encoders/decoders (stubs) ---------

fn net_encoder(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // NetEncoder[spec|auto]
    if args.is_empty() {
        return Value::Expr { head: Box::new(Value::Symbol("NetEncoder".into())), args };
    }
    match &args[0] {
        Value::String(_) | Value::Symbol(_) | Value::Assoc(_) | Value::List(_) => args[0].clone(),
        _ => Value::Symbol("Automatic".into()),
    }
}

fn net_decoder(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // NetDecoder[spec|auto]
    if args.is_empty() {
        return Value::Expr { head: Box::new(Value::Symbol("NetDecoder".into())), args };
    }
    match &args[0] {
        Value::String(_) | Value::Symbol(_) | Value::Assoc(_) | Value::List(_) => args[0].clone(),
        _ => Value::Symbol("Automatic".into()),
    }
}

// --------- layer constructors (descriptive only) ---------

// Naming-convention heads mapping to existing constructors
fn dense(ev: &mut Evaluator, args: Vec<Value>) -> Value { linear_layer(ev, args) }
fn convolution1d(ev: &mut Evaluator, args: Vec<Value>) -> Value { convolution_layer(ev, args) }
fn convolution2d(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // Convolution2D[<|Output->Cout, KernelSize->k|{kh,kw}, Stride->s|{sh,sw}, Padding->p|{ph,pw}, InputChannels->Cin, Height->H, Width->W|>]
    let (pos, opts) = parse_opts(ev, &args);
    let mut params: HashMap<String, Value> = match pos.get(0).map(|v| ev.eval(v.clone())) {
        Some(Value::Assoc(m)) => m,
        _ => opts,
    };
    if !params.contains_key("Output") { params.insert("Output".into(), Value::Integer(1)); }
    if !params.contains_key("KernelSize") { params.insert("KernelSize".into(), Value::Integer(3)); }
    if !params.contains_key("Stride") { params.insert("Stride".into(), Value::Integer(1)); }
    if !params.contains_key("Padding") { params.insert("Padding".into(), Value::Integer(0)); }
    layer_spec("Conv2D", params)
}
fn depthwise_conv2d(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // DepthwiseConv2D[<|KernelSize->k|{kh,kw}, Stride->s|{sh,sw}, Padding->p|{ph,pw}, InputChannels->Cin, Height->H, Width->W|>]
    let (pos, opts) = parse_opts(ev, &args);
    let mut params: HashMap<String, Value> = match pos.get(0).map(|v| ev.eval(v.clone())) {
        Some(Value::Assoc(m)) => m,
        _ => opts,
    };
    if !params.contains_key("KernelSize") { params.insert("KernelSize".into(), Value::Integer(3)); }
    if !params.contains_key("Stride") { params.insert("Stride".into(), Value::Integer(1)); }
    if !params.contains_key("Padding") { params.insert("Padding".into(), Value::Integer(0)); }
    layer_spec("DepthwiseConv2D", params)
}
fn pooling(ev: &mut Evaluator, args: Vec<Value>) -> Value { pooling_layer(ev, args) }
fn pooling2d(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // Pooling2D["Max"|"Avg", size or {kh,kw}, opts]
    let (pos, opts) = parse_opts(ev, &args);
    let mut params = opts;
    if let Some(kind) = pos.get(0) {
        if let Value::String(s) | Value::Symbol(s) = ev.eval(kind.clone()) {
            params.insert("PoolType".into(), Value::String(s));
        }
    }
    if let Some(sz) = pos.get(1) {
        params.insert("KernelSize".into(), ev.eval(sz.clone()));
    }
    if !params.contains_key("PoolType") { params.insert("PoolType".into(), Value::String("Max".into())); }
    if !params.contains_key("KernelSize") { params.insert("KernelSize".into(), Value::Integer(2)); }
    if !params.contains_key("Stride") { params.insert("Stride".into(), Value::Integer(2)); }
    layer_spec("Pool2D", params)
}

fn conv_transpose2d(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // ConvTranspose2D[<|Output->Cout, KernelSize->k|{kh,kw}, Stride->s|{sh,sw}, Padding->p|{ph,pw}, InputChannels->Cin, Height->H, Width->W|>]
    let (pos, opts) = parse_opts(ev, &args);
    let mut params: HashMap<String, Value> = match pos.get(0).map(|v| ev.eval(v.clone())) {
        Some(Value::Assoc(m)) => m,
        _ => opts,
    };
    if !params.contains_key("Output") { params.insert("Output".into(), Value::Integer(1)); }
    if !params.contains_key("KernelSize") { params.insert("KernelSize".into(), Value::Integer(3)); }
    if !params.contains_key("Stride") { params.insert("Stride".into(), Value::Integer(1)); }
    if !params.contains_key("Padding") { params.insert("Padding".into(), Value::Integer(0)); }
    layer_spec("ConvTranspose2D", params)
}

fn global_avg_pool2d(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // GlobalAvgPool2D[] with optional dims provided
    let (_pos, opts) = parse_opts(ev, &args);
    layer_spec("GlobalAvgPool2D", opts)
}

fn separable_conv2d(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // SeparableConv2D[<|Output->Cout, KernelSize->k|{kh,kw}, Stride->s|{sh,sw}, Padding->p|{ph,pw}, InputChannels->Cin, Height->H, Width->W|>]
    let (pos, opts) = parse_opts(ev, &args);
    let mut params: HashMap<String, Value> = match pos.get(0).map(|v| ev.eval(v.clone())) {
        Some(Value::Assoc(m)) => m,
        _ => opts,
    };
    if !params.contains_key("Output") { params.insert("Output".into(), Value::Integer(1)); }
    if !params.contains_key("KernelSize") { params.insert("KernelSize".into(), Value::Integer(3)); }
    if !params.contains_key("Stride") { params.insert("Stride".into(), Value::Integer(1)); }
    if !params.contains_key("Padding") { params.insert("Padding".into(), Value::Integer(0)); }
    layer_spec("SeparableConv2D", params)
}

fn group_norm(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // GroupNorm[<|NumGroups->g, Epsilon->1e-5, Gamma->..., Beta->..., InputChannels->Cin, Height->H, Width->W|>]
    let (pos, opts) = parse_opts(ev, &args);
    let mut params: HashMap<String, Value> = match pos.get(0).map(|v| ev.eval(v.clone())) { Some(Value::Assoc(m)) => m, _ => opts };
    if !params.contains_key("NumGroups") { params.insert("NumGroups".into(), Value::Integer(1)); }
    if !params.contains_key("Epsilon") { params.insert("Epsilon".into(), Value::Real(1e-5)); }
    layer_spec("GroupNorm", params)
}

fn residual(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // Residual[{layers...}] or Residual[layer]
    let (pos, _opts) = parse_opts(ev, &args);
    let layers = if let Some(first) = pos.get(0) {
        match ev.eval(first.clone()) {
            Value::List(vs) => Value::List(vs),
            other => Value::List(vec![other]),
        }
    } else { Value::List(vec![]) };
    let mut m = HashMap::new(); m.insert("Layers".into(), layers);
    layer_spec("Residual", m)
}

fn upsample2d(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // Upsample2D[<|Scale->s|{sh,sw}, Mode->"Nearest"|"Bilinear", InputChannels->Cin, Height->H, Width->W|>]
    let (pos, opts) = parse_opts(ev, &args);
    let mut params: HashMap<String, Value> = match pos.get(0).map(|v| ev.eval(v.clone())) { Some(Value::Assoc(m)) => m, _ => opts };
    if !params.contains_key("Scale") { params.insert("Scale".into(), Value::Integer(2)); }
    if !params.contains_key("Mode") { params.insert("Mode".into(), Value::String("Nearest".into())); }
    layer_spec("Upsample2D", params)
}

fn residual_block(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // ResidualBlock[<|Output->Cout, KernelSize->k|{kh,kw}, Stride->s|{sh,sw}, Padding->p|{ph,pw}, Activation->"Relu"|>]
    let (pos, opts) = parse_opts(ev, &args);
    let mut params: HashMap<String, Value> = match pos.get(0).map(|v| ev.eval(v.clone())) { Some(Value::Assoc(m)) => m, _ => opts };
    if !params.contains_key("Output") { params.insert("Output".into(), Value::Integer(1)); }
    if !params.contains_key("KernelSize") { params.insert("KernelSize".into(), Value::Integer(3)); }
    if !params.contains_key("Stride") { params.insert("Stride".into(), Value::Integer(1)); }
    if !params.contains_key("Padding") { params.insert("Padding".into(), Value::Integer(1)); }
    if !params.contains_key("Activation") { params.insert("Activation".into(), Value::String("Relu".into())); }
    layer_spec("ResidualBlock", params)
}
fn batchnorm(ev: &mut Evaluator, args: Vec<Value>) -> Value { batchnorm_layer(ev, args) }
fn layernorm(ev: &mut Evaluator, args: Vec<Value>) -> Value { layernorm_layer(ev, args) }
fn dropout(ev: &mut Evaluator, args: Vec<Value>) -> Value { dropout_layer(ev, args) }
fn flatten(_ev: &mut Evaluator, _args: Vec<Value>) -> Value { flatten_layer(_ev, _args) }
fn reshape(ev: &mut Evaluator, args: Vec<Value>) -> Value { reshape_layer(ev, args) }
fn __transpose_layer(ev: &mut Evaluator, args: Vec<Value>) -> Value { transpose_layer(ev, args) }
fn __concat_layer(ev: &mut Evaluator, args: Vec<Value>) -> Value { concat_layer(ev, args) }
fn add(_ev: &mut Evaluator, _args: Vec<Value>) -> Value { add_layer(_ev, _args) }
fn mul(_ev: &mut Evaluator, _args: Vec<Value>) -> Value { mul_layer(_ev, _args) }
fn embedding(ev: &mut Evaluator, args: Vec<Value>) -> Value { embedding_layer(ev, args) }

fn multi_head_attention(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // MultiHeadAttention[<|SeqLen->s, ModelDim->d, NumHeads->h|>]
    let (pos, opts) = parse_opts(ev, &args);
    let mut params: HashMap<String, Value> = match pos.get(0).map(|v| ev.eval(v.clone())) { Some(Value::Assoc(m)) => m, _ => opts };
    if !params.contains_key("SeqLen") { params.insert("SeqLen".into(), Value::Integer(0)); }
    if !params.contains_key("ModelDim") { params.insert("ModelDim".into(), Value::Integer(0)); }
    if !params.contains_key("NumHeads") { params.insert("NumHeads".into(), Value::Integer(1)); }
    layer_spec("MultiHeadAttention", params)
}

fn positional_encoding(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // PositionalEncoding[<|SeqLen->s, ModelDim->d|>]
    let (pos, opts) = parse_opts(ev, &args);
    let mut params: HashMap<String, Value> = match pos.get(0).map(|v| ev.eval(v.clone())) { Some(Value::Assoc(m)) => m, _ => opts };
    if !params.contains_key("SeqLen") { params.insert("SeqLen".into(), Value::Integer(0)); }
    if !params.contains_key("ModelDim") { params.insert("ModelDim".into(), Value::Integer(0)); }
    layer_spec("PositionalEncoding", params)
}

fn rms_norm(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    let (pos, opts) = parse_opts(ev, &args);
    let mut params: HashMap<String, Value> = match pos.get(0).map(|v| ev.eval(v.clone())) { Some(Value::Assoc(m)) => m, _ => opts };
    if !params.contains_key("SeqLen") { params.insert("SeqLen".into(), Value::Integer(0)); }
    if !params.contains_key("ModelDim") { params.insert("ModelDim".into(), Value::Integer(0)); }
    if !params.contains_key("Epsilon") { params.insert("Epsilon".into(), Value::Real(1e-5)); }
    layer_spec("RMSNorm", params)
}

fn ffn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    let (pos, opts) = parse_opts(ev, &args);
    let mut params: HashMap<String, Value> = match pos.get(0).map(|v| ev.eval(v.clone())) { Some(Value::Assoc(m)) => m, _ => opts };
    if !params.contains_key("SeqLen") { params.insert("SeqLen".into(), Value::Integer(0)); }
    if !params.contains_key("ModelDim") { params.insert("ModelDim".into(), Value::Integer(0)); }
    if !params.contains_key("HiddenDim") { if let Some(Value::Integer(d)) = params.get("ModelDim") { params.insert("HiddenDim".into(), Value::Integer((*d).max(1)*4)); } else { params.insert("HiddenDim".into(), Value::Integer(0)); } }
    if !params.contains_key("Variant") { params.insert("Variant".into(), Value::String("SwiGLU".into())); }
    layer_spec("FFN", params)
}

fn causal_self_attention(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // Alias to MultiHeadAttention with Causal->True
    let (pos, opts) = parse_opts(ev, &args);
    let mut params: HashMap<String, Value> = match pos.get(0).map(|v| ev.eval(v.clone())) { Some(Value::Assoc(m)) => m, _ => opts };
    if !params.contains_key("SeqLen") { params.insert("SeqLen".into(), Value::Integer(0)); }
    if !params.contains_key("ModelDim") { params.insert("ModelDim".into(), Value::Integer(0)); }
    if !params.contains_key("NumHeads") { params.insert("NumHeads".into(), Value::Integer(1)); }
    params.insert("Causal".into(), Value::Boolean(true));
    layer_spec("MultiHeadAttention", params)
}

fn cross_attention(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    let (pos, opts) = parse_opts(ev, &args);
    let mut params: HashMap<String, Value> = match pos.get(0).map(|v| ev.eval(v.clone())) { Some(Value::Assoc(m)) => m, _ => opts };
    if !params.contains_key("SeqLen") { params.insert("SeqLen".into(), Value::Integer(0)); }
    if !params.contains_key("ModelDim") { params.insert("ModelDim".into(), Value::Integer(0)); }
    if !params.contains_key("NumHeads") { params.insert("NumHeads".into(), Value::Integer(1)); }
    layer_spec("CrossAttention", params)
}

fn positional_embedding(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    let (pos, opts) = parse_opts(ev, &args);
    let mut params: HashMap<String, Value> = match pos.get(0).map(|v| ev.eval(v.clone())) { Some(Value::Assoc(m)) => m, _ => opts };
    if !params.contains_key("SeqLen") { params.insert("SeqLen".into(), Value::Integer(0)); }
    if !params.contains_key("ModelDim") { params.insert("ModelDim".into(), Value::Integer(0)); }
    layer_spec("PositionalEmbedding", params)
}

fn patch_embedding2d(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    let (pos, opts) = parse_opts(ev, &args);
    let mut params: HashMap<String, Value> = match pos.get(0).map(|v| ev.eval(v.clone())) { Some(Value::Assoc(m)) => m, _ => opts };
    if !params.contains_key("PatchSize") { params.insert("PatchSize".into(), Value::List(vec![Value::Integer(16), Value::Integer(16)])); }
    if !params.contains_key("ModelDim") { params.insert("ModelDim".into(), Value::Integer(64)); }
    layer_spec("PatchEmbedding2D", params)
}

fn transformer_encoder(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // TransformerEncoder[<|SeqLen->s, ModelDim->d, NumHeads->h, HiddenDim->m, Activation->Gelu|>]
    let (pos, opts) = parse_opts(ev, &args);
    let mut params: HashMap<String, Value> = match pos.get(0).map(|v| ev.eval(v.clone())) { Some(Value::Assoc(m)) => m, _ => opts };
    if !params.contains_key("SeqLen") { params.insert("SeqLen".into(), Value::Integer(0)); }
    if !params.contains_key("ModelDim") { params.insert("ModelDim".into(), Value::Integer(0)); }
    if !params.contains_key("NumHeads") { params.insert("NumHeads".into(), Value::Integer(1)); }
    if !params.contains_key("HiddenDim") { if let Some(Value::Integer(d)) = params.get("ModelDim") { params.insert("HiddenDim".into(), Value::Integer((*d).max(1)*4)); } else { params.insert("HiddenDim".into(), Value::Integer(0)); } }
    if !params.contains_key("Activation") { params.insert("Activation".into(), Value::String("Gelu".into())); }
    layer_spec("TransformerEncoder", params)
}

fn transformer_encoder_stack(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // TransformerEncoderStack[<|Layers->N, SeqLen->s, ModelDim->d, NumHeads->h, HiddenDim->m, Activation->...|>]
    let (pos, opts) = parse_opts(ev, &args);
    let params: HashMap<String, Value> = match pos.get(0).map(|v| ev.eval(v.clone())) { Some(Value::Assoc(m)) => m, _ => opts };
    let layers = params.get("Layers").or_else(|| params.get("Depth")).and_then(|v| as_i64(v)).unwrap_or(2).max(1) as usize;
    // Build N TransformerEncoder layers with the same configuration; each layer will be initialized independently
    let mut lst: Vec<Value> = Vec::with_capacity(layers);
    // Filter out stack-only keys so inner layers get clean config (but preserve Mask/Causal/etc.)
    let mut inner = params.clone();
    inner.remove("Layers"); inner.remove("Depth");
    for _ in 0..layers {
        lst.push(transformer_encoder(ev, vec![Value::Assoc(inner.clone())]));
    }
    Value::Expr { head: Box::new(Value::Symbol("Sequential".into())), args: vec![Value::List(lst)] }
}

fn transformer_decoder(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // TransformerDecoder[<|SeqLen->s, ModelDim->d, NumHeads->h, HiddenDim->m, Activation->Gelu, Memory->..., MemoryMask->...|>]
    let (pos, opts) = parse_opts(ev, &args);
    let mut params: HashMap<String, Value> = match pos.get(0).map(|v| ev.eval(v.clone())) { Some(Value::Assoc(m)) => m, _ => opts };
    if !params.contains_key("SeqLen") { params.insert("SeqLen".into(), Value::Integer(0)); }
    if !params.contains_key("ModelDim") { params.insert("ModelDim".into(), Value::Integer(0)); }
    if !params.contains_key("NumHeads") { params.insert("NumHeads".into(), Value::Integer(1)); }
    if !params.contains_key("HiddenDim") { if let Some(Value::Integer(d)) = params.get("ModelDim") { params.insert("HiddenDim".into(), Value::Integer((*d).max(1)*4)); } else { params.insert("HiddenDim".into(), Value::Integer(0)); } }
    if !params.contains_key("Activation") { params.insert("Activation".into(), Value::String("Gelu".into())); }
    layer_spec("TransformerDecoder", params)
}

fn transformer_decoder_stack(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // TransformerDecoderStack[<|Layers->N, SeqLen->s, ModelDim->d, NumHeads->h, HiddenDim->m, Activation->...|>]
    let (pos, opts) = parse_opts(ev, &args);
    let params: HashMap<String, Value> = match pos.get(0).map(|v| ev.eval(v.clone())) { Some(Value::Assoc(m)) => m, _ => opts };
    let layers = params.get("Layers").or_else(|| params.get("Depth")).and_then(|v| as_i64(v)).unwrap_or(2).max(1) as usize;
    let mut lst: Vec<Value> = Vec::with_capacity(layers);
    let mut inner = params.clone();
    inner.remove("Layers"); inner.remove("Depth");
    for _ in 0..layers {
        lst.push(transformer_decoder(ev, vec![Value::Assoc(inner.clone())]));
    }
    Value::Expr { head: Box::new(Value::Symbol("Sequential".into())), args: vec![Value::List(lst)] }
}

fn transformer_encoder_decoder(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // TransformerEncoderDecoder[<|EncLayers->Ne, DecLayers->Nd, SeqLen->L, ModelDim->D, NumHeads->H, HiddenDim->M, Activation->..., Causal->True, Mask->..., MemoryMask->...|>]
    // Returns an assoc: <| Encoder -> TransformerEncoderStack[...], Decoder -> TransformerDecoderStack[...] |> (Memory is left to caller)
    let (pos, opts) = parse_opts(ev, &args);
    let mut params: HashMap<String, Value> = match pos.get(0).map(|v| ev.eval(v.clone())) { Some(Value::Assoc(m)) => m, _ => opts };
    let enc_layers = params.get("EncLayers").or_else(|| params.get("EncoderLayers")).or_else(|| params.get("Layers")).and_then(|v| as_i64(v)).unwrap_or(2).max(1) as usize;
    let dec_layers = params.get("DecLayers").or_else(|| params.get("DecoderLayers")).or_else(|| params.get("Layers")).and_then(|v| as_i64(v)).unwrap_or(2).max(1) as usize;
    // Shared core params
    let mut shared = params.clone();
    shared.remove("EncLayers"); shared.remove("EncoderLayers"); shared.remove("DecLayers"); shared.remove("DecoderLayers"); shared.remove("Layers");
    // Build encoder stack
    let mut enc_params = shared.clone();
    enc_params.insert("Layers".into(), Value::Integer(enc_layers as i64));
    let encoder = transformer_encoder_stack(ev, vec![Value::Assoc(enc_params)]);
    // Build decoder stack (with causal/masks if provided). Memory is not set here.
    let mut dec_params = shared.clone();
    dec_params.insert("Layers".into(), Value::Integer(dec_layers as i64));
    let decoder = transformer_decoder_stack(ev, vec![Value::Assoc(dec_params)]);
    Value::Assoc(HashMap::from([
        ("Encoder".into(), encoder),
        ("Decoder".into(), decoder),
    ]))
}

fn linear_layer(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    let (pos, opts) = parse_opts(ev, &args);
    // Accept LinearLayer[out] or LinearLayer[<|"Output"->n, ...|>]
    let mut params: HashMap<String, Value> = HashMap::new();
    if let Some(Value::Assoc(m)) = args.last().and_then(|v| match v {
        Value::Assoc(_) => Some(ev.eval(v.clone())),
        _ => None,
    }) {
        params = m;
    }
    if let Some(first) = pos.get(0) {
        match ev.eval(first.clone()) {
            Value::Integer(n) => {
                params.entry("Output".into()).or_insert(Value::Integer(n));
            }
            Value::Real(x) => {
                params.entry("Output".into()).or_insert(Value::Integer(x as i64));
            }
            _ => {}
        }
    }
    for (k, v) in opts {
        params.insert(k, v);
    }
    if !params.contains_key("Bias") {
        params.insert("Bias".into(), Value::Boolean(true));
    }
    if !params.contains_key("Output") {
        params.insert("Output".into(), Value::Integer(0));
    }
    layer_spec("Linear", params)
}

fn activation_layer(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    let (pos, opts) = parse_opts(ev, &args);
    let mut params: HashMap<String, Value> = HashMap::new();
    if let Some(Value::Assoc(m)) = args.last().and_then(|v| match v {
        Value::Assoc(_) => Some(ev.eval(v.clone())),
        _ => None,
    }) {
        params = m;
    }
    if let Some(first) = pos.get(0) {
        match ev.eval(first.clone()) {
            Value::String(s) | Value::Symbol(s) => {
                params.entry("Type".into()).or_insert(Value::String(s));
            }
            _ => {}
        }
    }
    for (k, v) in opts {
        params.insert(k, v);
    }
    // normalize/validate type
    let at = params
        .get("Type")
        .and_then(|v| match v {
            Value::String(s) | Value::Symbol(s) => Some(s.clone()),
            _ => None,
        })
        .unwrap_or("ReLU".into());
    let at_lc = at.to_lowercase();
    let valid = ["relu", "sigmoid", "tanh", "gelu", "softmax"];
    let at_norm = if valid.contains(&at_lc.as_str()) {
        match at_lc.as_str() {
            "relu" => "ReLU",
            "sigmoid" => "Sigmoid",
            "tanh" => "Tanh",
            "gelu" => "GELU",
            "softmax" => "Softmax",
            _ => "ReLU",
        }
        .to_string()
    } else {
        "ReLU".into()
    };
    params.insert("Type".into(), Value::String(at_norm));
    layer_spec("Activation", params)
}

fn dropout_layer(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    let (pos, opts) = parse_opts(ev, &args);
    let mut params: HashMap<String, Value> = HashMap::new();
    if let Some(first) = pos.get(0) {
        match ev.eval(first.clone()) {
            Value::Real(x) => {
                params.insert("Rate".into(), Value::Real(x));
            }
            Value::Integer(n) => {
                params.insert("Rate".into(), Value::Real(n as f64));
            }
            _ => {}
        }
    }
    for (k, v) in opts {
        params.insert(k, v);
    }
    if !params.contains_key("Rate") {
        params.insert("Rate".into(), Value::Real(0.5));
    }
    layer_spec("Dropout", params)
}

fn flatten_layer(_ev: &mut Evaluator, _args: Vec<Value>) -> Value {
    layer_spec("Flatten", HashMap::new())
}

fn softmax_layer(_ev: &mut Evaluator, _args: Vec<Value>) -> Value {
    layer_spec("Activation", HashMap::from([("Type".into(), Value::String("Softmax".into()))]))
}

fn convolution_layer(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // 1D conv constructor: ConvolutionLayer[outChannels, kernelSize, opts]
    let (pos, opts) = parse_opts(ev, &args);
    let mut params: HashMap<String, Value> = HashMap::new();
    if let Some(Value::Assoc(m)) = args.last().and_then(|v| match v {
        Value::Assoc(_) => Some(ev.eval(v.clone())),
        _ => None,
    }) {
        params = m;
    }
    if let Some(first) = pos.get(0) {
        if let Some(n) = as_i64(&ev.eval(first.clone())) {
            params.entry("Output".into()).or_insert(Value::Integer(n));
        }
    }
    if let Some(second) = pos.get(1) {
        if let Some(n) = as_i64(&ev.eval(second.clone())) {
            params.entry("KernelSize".into()).or_insert(Value::Integer(n));
        }
    }
    for (k, v) in opts {
        params.insert(k, v);
    }
    if !params.contains_key("Output") {
        params.insert("Output".into(), Value::Integer(1));
    }
    if !params.contains_key("KernelSize") {
        params.insert("KernelSize".into(), Value::Integer(3));
    }
    if !params.contains_key("Stride") {
        params.insert("Stride".into(), Value::Integer(1));
    }
    if !params.contains_key("Padding") {
        params.insert("Padding".into(), Value::Integer(0));
    }
    layer_spec("Convolution", params)
}

fn pooling_layer(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // 1D pooling: PoolingLayer["Max"|"Avg", size, opts]
    let (pos, opts) = parse_opts(ev, &args);
    let mut params: HashMap<String, Value> = HashMap::new();
    if let Some(kind) = pos.get(0) {
        if let Value::String(s) | Value::Symbol(s) = ev.eval(kind.clone()) {
            params.insert("PoolType".into(), Value::String(s));
        }
    }
    if let Some(sz) = pos.get(1) {
        if let Some(n) = as_i64(&ev.eval(sz.clone())) {
            params.insert("Size".into(), Value::Integer(n));
        }
    }
    for (k, v) in opts {
        params.insert(k, v);
    }
    if !params.contains_key("PoolType") {
        params.insert("PoolType".into(), Value::String("Max".into()));
    }
    if !params.contains_key("Size") {
        params.insert("Size".into(), Value::Integer(2));
    }
    if !params.contains_key("Stride") {
        params.insert("Stride".into(), Value::Integer(2));
    }
    layer_spec("Pooling", params)
}

fn batchnorm_layer(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    let (_pos, opts) = parse_opts(ev, &args);
    let mut params = opts;
    if !params.contains_key("Epsilon") {
        params.insert("Epsilon".into(), Value::Real(1e-5));
    }
    layer_spec("BatchNorm", params)
}

fn reshape_layer(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    let (pos, opts) = parse_opts(ev, &args);
    let mut params = opts;
    if let Some(shape) = pos.get(0) {
        params.insert("Shape".into(), ev.eval(shape.clone()));
    }
    layer_spec("Reshape", params)
}

fn transpose_layer(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    let (_pos, opts) = parse_opts(ev, &args);
    layer_spec("Transpose", opts)
}
fn concat_layer(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    let (_pos, opts) = parse_opts(ev, &args);
    layer_spec("Concat", opts)
}
fn add_layer(_ev: &mut Evaluator, _args: Vec<Value>) -> Value {
    layer_spec("Add", HashMap::new())
}
fn mul_layer(_ev: &mut Evaluator, _args: Vec<Value>) -> Value {
    layer_spec("Mul", HashMap::new())
}

fn embedding_layer(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // EmbeddingLayer[vocab, dim]
    let (pos, opts) = parse_opts(ev, &args);
    let mut params = opts;
    if let Some(v) = pos.get(0) {
        if let Some(n) = as_i64(&ev.eval(v.clone())) {
            params.insert("Vocab".into(), Value::Integer(n));
        }
    }
    if let Some(v) = pos.get(1) {
        if let Some(n) = as_i64(&ev.eval(v.clone())) {
            params.insert("Dim".into(), Value::Integer(n));
        }
    }
    if !params.contains_key("Vocab") {
        params.insert("Vocab".into(), Value::Integer(1000));
    }
    if !params.contains_key("Dim") {
        params.insert("Dim".into(), Value::Integer(16));
    }
    layer_spec("Embedding", params)
}

fn layernorm_layer(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    let (_pos, opts) = parse_opts(ev, &args);
    let mut params = opts;
    if !params.contains_key("Epsilon") {
        params.insert("Epsilon".into(), Value::Real(1e-5));
    }
    layer_spec("LayerNorm", params)
}

pub fn register_nn_filtered(ev: &mut Evaluator, pred: &dyn Fn(&str) -> bool) {
    register_if(ev, pred, "NetChain", net_chain as NativeFn, Attributes::empty());
    register_if(ev, pred, "NetGraph", net_graph as NativeFn, Attributes::empty());
    register_if(ev, pred, "NetInitialize", net_initialize as NativeFn, Attributes::empty());
    register_if(ev, pred, "NetTrain", net_train as NativeFn, Attributes::empty());
    register_if(ev, pred, "NetApply", net_apply as NativeFn, Attributes::empty());
    register_if(ev, pred, "NetProperty", net_property as NativeFn, Attributes::empty());
    register_if(ev, pred, "NetSummary", net_summary as NativeFn, Attributes::empty());
    register_if(ev, pred, "NetEncoder", net_encoder as NativeFn, Attributes::empty());
    register_if(ev, pred, "NetDecoder", net_decoder as NativeFn, Attributes::empty());

    // Canonical heads (public)
    register_if(ev, pred, "Network", network as NativeFn, Attributes::empty());
    register_if(ev, pred, "Sequential", sequential as NativeFn, Attributes::empty());
    register_if(ev, pred, "GraphNetwork", graph_network as NativeFn, Attributes::empty());
    register_if(ev, pred, "Initializer", initializer as NativeFn, Attributes::empty());

    // Layers (noun heads)
    register_if(ev, pred, "Dense", dense as NativeFn, Attributes::empty());
    register_if(ev, pred, "Convolution1D", convolution1d as NativeFn, Attributes::empty());
    register_if(ev, pred, "Convolution2D", convolution2d as NativeFn, Attributes::empty());
    register_if(ev, pred, "DepthwiseConv2D", depthwise_conv2d as NativeFn, Attributes::empty());
    register_if(ev, pred, "ConvTranspose2D", conv_transpose2d as NativeFn, Attributes::empty());
    register_if(ev, pred, "SeparableConv2D", separable_conv2d as NativeFn, Attributes::empty());
    register_if(ev, pred, "Pooling", pooling as NativeFn, Attributes::empty());
    register_if(ev, pred, "Pooling2D", pooling2d as NativeFn, Attributes::empty());
    register_if(ev, pred, "GlobalAvgPool2D", global_avg_pool2d as NativeFn, Attributes::empty());
    register_if(ev, pred, "GroupNorm", group_norm as NativeFn, Attributes::empty());
    register_if(ev, pred, "Residual", residual as NativeFn, Attributes::empty());
    register_if(ev, pred, "Upsample2D", upsample2d as NativeFn, Attributes::empty());
    register_if(ev, pred, "ResidualBlock", residual_block as NativeFn, Attributes::empty());
    register_if(ev, pred, "BatchNorm", batchnorm as NativeFn, Attributes::empty());
    register_if(ev, pred, "LayerNorm", layernorm as NativeFn, Attributes::empty());
    register_if(ev, pred, "Dropout", dropout as NativeFn, Attributes::empty());
    register_if(ev, pred, "Flatten", flatten as NativeFn, Attributes::empty());
    register_if(ev, pred, "Reshape", reshape as NativeFn, Attributes::empty());
    register_if(ev, pred, "TransformerEncoderDecoder", transformer_encoder_decoder as NativeFn, Attributes::empty());
    // Transpose tensor op handled by dispatcher; expose internal layer alias only
    register_if(ev, pred, "__TransposeLayer", __transpose_layer as NativeFn, Attributes::empty());
    // Avoid clobbering Dataset Concat; keep internal layer alias only
    register_if(ev, pred, "__ConcatLayer", __concat_layer as NativeFn, Attributes::empty());
    register_if(ev, pred, "__AddLayer", add as NativeFn, Attributes::empty());
    register_if(ev, pred, "__MulLayer", mul as NativeFn, Attributes::empty());
    register_if(ev, pred, "Embedding", embedding as NativeFn, Attributes::empty());
    // Internal activation constructor for zero-arg activation heads
    register_if(ev, pred, "__ActivationLayer", activation_layer as NativeFn, Attributes::empty());
    // New layers
    register_if(ev, pred, "RMSNorm", rms_norm as NativeFn, Attributes::empty());
    register_if(ev, pred, "FFN", ffn as NativeFn, Attributes::empty());
    register_if(ev, pred, "CausalSelfAttention", causal_self_attention as NativeFn, Attributes::empty());
    register_if(ev, pred, "CrossAttention", cross_attention as NativeFn, Attributes::empty());
    register_if(ev, pred, "PositionalEmbedding", positional_embedding as NativeFn, Attributes::empty());
    register_if(ev, pred, "PatchEmbedding2D", patch_embedding2d as NativeFn, Attributes::empty());
}

// --------- Naming-convention network constructors ---------
fn network(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // Network[<|Kind->"Chain"|"Graph", Layers->..., ...|>] or Network[layers]
    if args.is_empty() {
        return net_chain(ev, vec![Value::List(vec![])]);
    }
    let a0 = ev.eval(args[0].clone());
    match a0 {
        Value::Assoc(m) => {
            let kind = m
                .get("Kind")
                .and_then(|v| match v { Value::String(s) | Value::Symbol(s) => Some(s.clone()), _ => None })
                .unwrap_or("Chain".into());
            if kind.eq_ignore_ascii_case("Graph") {
                let nodes = m.get("Nodes").cloned().unwrap_or(Value::Assoc(HashMap::new()));
                let edges = m.get("Edges").cloned().unwrap_or(Value::List(vec![]));
                net_graph(ev, vec![nodes, edges, Value::Assoc(m)])
            } else {
                let layers = m.get("Layers").cloned().unwrap_or(Value::List(vec![]));
                net_chain(ev, vec![layers, Value::Assoc(m)])
            }
        }
        Value::List(_) => net_chain(ev, vec![a0]),
        other => net_chain(ev, vec![other]),
    }
}

fn sequential(ev: &mut Evaluator, args: Vec<Value>) -> Value { net_chain(ev, args) }
fn graph_network(ev: &mut Evaluator, args: Vec<Value>) -> Value { net_graph(ev, args) }

fn initializer(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // Initializer[<|Type->"Xavier"|"He"|"Zeros"|"Ones"|>]
    let kind = if let Some(a0) = args.get(0) {
        match ev.eval(a0.clone()) {
            Value::Assoc(m) => m.get("Type").and_then(|v| match v { Value::String(s) | Value::Symbol(s) => Some(s.clone()), _ => None }).unwrap_or("Xavier".into()),
            Value::String(s) | Value::Symbol(s) => s,
            _ => "Xavier".into(),
        }
    } else {
        "Xavier".into()
    };
    Value::Assoc(HashMap::from([(String::from("Type"), Value::String(kind))]))
}
