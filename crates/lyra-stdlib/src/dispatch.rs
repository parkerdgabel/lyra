use lyra_core::value::Value;
use lyra_runtime::attrs::Attributes;
use lyra_runtime::Evaluator;
use std::collections::HashMap;

pub fn register_dispatch(ev: &mut Evaluator) {
    // Register after list and dataset so we override conflicting names with a dispatcher.
    ev.register("Join", join_dispatch as NativeFn, Attributes::empty());
    ev.register("Union", union_dispatch as NativeFn, Attributes::empty());
    ev.register("Intersection", intersection_dispatch as NativeFn, Attributes::empty());
    ev.register("Difference", difference_dispatch as NativeFn, Attributes::empty());
    ev.register("Select", select_dispatch as NativeFn, Attributes::empty());
    ev.register("Drop", drop_dispatch as NativeFn, Attributes::empty());
    ev.register("Delete", delete_dispatch as NativeFn, Attributes::empty());
    ev.register("Write", write_dispatch as NativeFn, Attributes::empty());
    ev.register("Filter", filter_dispatch as NativeFn, Attributes::HOLD_ALL);
    ev.register("Head", head_dispatch as NativeFn, Attributes::empty());
    ev.register("Tail", tail_dispatch as NativeFn, Attributes::empty());
    ev.register("Offset", offset_dispatch as NativeFn, Attributes::empty());
    ev.register("Sort", sort_dispatch as NativeFn, Attributes::empty());
    // SortBy dispatcher to route to assoc or list versions
    ev.register("SortBy", sortby_dispatch as NativeFn, Attributes::HOLD_ALL);
    ev.register("Distinct", distinct_dispatch as NativeFn, Attributes::empty());
    // Generic verbs
    ev.register("Upsert", upsert_dispatch as NativeFn, Attributes::empty());
    ev.register("Insert", insert_dispatch as NativeFn, Attributes::empty());
    ev.register("Remove", remove_dispatch as NativeFn, Attributes::empty());
    ev.register("Add", add_dispatch as NativeFn, Attributes::empty());
    ev.register("Info", info_dispatch as NativeFn, Attributes::empty());
    ev.register("Length", length_dispatch as NativeFn, Attributes::empty());
    ev.register("EmptyQ", emptyq_dispatch as NativeFn, Attributes::empty());
    ev.register("Count", count_dispatch as NativeFn, Attributes::empty());
    ev.register("Search", search_dispatch as NativeFn, Attributes::empty());
    ev.register("Split", split_dispatch as NativeFn, Attributes::empty());
    ev.register("Reset", reset_dispatch as NativeFn, Attributes::empty());
    ev.register("Close", close_dispatch as NativeFn, Attributes::empty());
    ev.register("Columns", columns_dispatch as NativeFn, Attributes::empty());
    ev.register("Contains", contains_dispatch as NativeFn, Attributes::empty());
    ev.register("ContainsQ", contains_dispatch as NativeFn, Attributes::empty());
    ev.register("Get", get_dispatch as NativeFn, Attributes::empty());
    ev.register("SubsetQ", subsetq_dispatch as NativeFn, Attributes::empty());
    ev.register("EqualQ", equalq_dispatch as NativeFn, Attributes::empty());
    // Common aliases/patterns
    ev.register("MemberQ", contains_dispatch as NativeFn, Attributes::empty());
    ev.register("ContainsKeyQ", contains_key_dispatch as NativeFn, Attributes::empty());
    ev.register("HasKeyQ", contains_key_dispatch as NativeFn, Attributes::empty());
    ev.register("Keys", keys_dispatch as NativeFn, Attributes::empty());
    ev.register("Describe", describe_dispatch as NativeFn, Attributes::HOLD_ALL);
    ev.register("Import", import_dispatch as NativeFn, Attributes::empty());
    ev.register("ImportString", import_string_dispatch as NativeFn, Attributes::empty());
    ev.register("ImportBytes", import_bytes_dispatch as NativeFn, Attributes::empty());
    ev.register("Export", export_dispatch as NativeFn, Attributes::empty());
    ev.register("Sniff", sniff_dispatch as NativeFn, Attributes::empty());

    // Tensor/List shared names and NN/ML generic verbs
    ev.register("Transpose", transpose_dispatch as NativeFn, Attributes::empty());
    ev.register("ArgMax", argmax_dispatch as NativeFn, Attributes::HOLD_ALL);
    ev.register("Clip", clip_dispatch as NativeFn, Attributes::empty());
    ev.register("Relu", relu_dispatch as NativeFn, Attributes::empty());
    ev.register("Sigmoid", sigmoid_dispatch as NativeFn, Attributes::empty());
    ev.register("Gelu", gelu_dispatch as NativeFn, Attributes::empty());
    ev.register("Softmax", softmax_dispatch as NativeFn, Attributes::empty());
    ev.register("Power", power_dispatch as NativeFn, Attributes::empty());
    ev.register("Train", train_dispatch as NativeFn, Attributes::empty());
    ev.register("Evaluate", evaluate_dispatch as NativeFn, Attributes::empty());
    ev.register("CrossValidate", crossvalidate_dispatch as NativeFn, Attributes::empty());
    ev.register("Tune", tune_dispatch as NativeFn, Attributes::empty());
    ev.register("Predict", predict_dispatch as NativeFn, Attributes::empty());
    ev.register("Initialize", initialize_dispatch as NativeFn, Attributes::empty());
    ev.register("Property", property_dispatch as NativeFn, Attributes::empty());
    ev.register("Summary", summary_dispatch as NativeFn, Attributes::empty());

    // Tensor-aware elementwise math
    ev.register("Exp", exp_dispatch as NativeFn, Attributes::empty());
    ev.register("Log", log_dispatch as NativeFn, Attributes::empty());
    ev.register("Sqrt", sqrt_dispatch as NativeFn, Attributes::empty());
    ev.register("Sin", sin_dispatch as NativeFn, Attributes::empty());
    ev.register("Cos", cos_dispatch as NativeFn, Attributes::empty());
    ev.register("Tanh", tanh_dispatch as NativeFn, Attributes::empty());

    #[cfg(feature = "tools")]
    {
        use crate::{tool_spec, tools::add_specs};
        let specs = vec![
            tool_spec!(
                "Join",
                summary: "Concatenate lists or join datasets",
                params: ["a","b","opts?"],
                tags: ["generic","list","dataset"],
                input_schema: Value::Assoc(HashMap::from([
                    ("type".into(), Value::String("object".into())),
                    ("properties".into(), Value::Assoc(HashMap::from([
                        ("a".into(), Value::Assoc(HashMap::from([
                            ("type".into(), Value::String("array".into())),
                        ]))),
                        ("b".into(), Value::Assoc(HashMap::from([
                            ("type".into(), Value::String("array".into())),
                        ]))),
                    ]))),
                    ("required".into(), Value::List(vec![Value::String("a".into()), Value::String("b".into())])),
                ])),
                examples: [
                    Value::String("Join[{1,2}, {3,4}]  ==> {1,2,3,4}".into()),
                    Value::String("ds1 := DatasetFromRows[{<|id->1|>}] ; ds2 := DatasetFromRows[{<|id->1|>}]; Join[ds1, ds2, {\"id\"}]".into()),
                ]
            ),
            tool_spec!(
                "SubsetQ",
                summary: "Is a subset of b? (sets, lists)",
                params: ["a","b"],
                tags: ["generic","set","list"],
                examples: [
                    Value::String("SubsetQ[HashSet[{1,2}], HashSet[{1,2,3}]]  ==> True".into()),
                    Value::String("SubsetQ[{1,2}, {1,2,3}]  ==> True".into()),
                ]
            ),
            tool_spec!(
                "EqualQ",
                summary: "Structural equality for sets and handles",
                params: ["a","b"],
                tags: ["generic","set"],
                examples: [
                    Value::String("EqualQ[HashSet[{1,2}], HashSet[{2,1}]]  ==> True".into()),
                ]
            ),
            tool_spec!(
                "Union",
                summary: "Union for lists, sets, or datasets (by columns)",
                params: ["args"],
                tags: ["generic","set","dataset","list"],
                examples: [
                    Value::String("Union[{1,2}, {2,3}]  ==> {1,2,3}".into()),
                    Value::String("s1 := HashSet[{1,2}]; s2 := HashSet[{2,3}]; Union[s1, s2]".into()),
                    Value::String("Union[ {<|a->1|>}, {<|a->1|>} ]  (* rows: dataset union-by-columns *)".into()),
                ]
            ),
            tool_spec!(
                "Insert",
                summary: "Insert into collection or structure (dispatched)",
                params: ["target","value"],
                tags: ["generic","collection"],
                examples: [
                    Value::String("s := HashSet[{1,2}]; Insert[s, 3]".into()),
                    Value::String("q := Queue[]; Insert[q, 5]".into()),
                    Value::String("st := Stack[]; Insert[st, \"x\"]".into()),
                    Value::String("pq := PriorityQueue[]; Insert[pq, <|\"Key\"->1, \"Value\"->\"a\"|>]".into()),
                    Value::String("g := Graph[]; Insert[g, {\"a\",\"b\"}]  (* nodes *)".into()),
                    Value::String("Insert[g, <|Src->\"a\",Dst->\"b\"|>]  (* edge *)".into()),
                ]
            ),
            tool_spec!(
                "Remove",
                summary: "Remove from collection or structure (dispatched)",
                params: ["target","value?"],
                tags: ["generic","collection"],
                examples: [
                    Value::String("Remove[HashSet[{1,2,3}], 2]  ==> s'".into()),
                    Value::String("Remove[Queue[]]  ==> Null (dequeue)".into()),
                    Value::String("Remove[Stack[]]  ==> Null (pop)".into()),
                    Value::String("Remove[PriorityQueue[]]  ==> Null (pop)".into()),
                    Value::String("Remove[Bag[{\"x\"}], \"x\"]".into()),
                    Value::String("Remove[g, {\"a\"}]  (* remove nodes *)".into()),
                    Value::String("Remove[g, {<|Src->\"a\",Dst->\"b\"|>}]  (* remove edges *)".into()),
                ]
            ),
            tool_spec!(
                "Add",
                summary: "Add value to a collection (alias of Insert for some types)",
                params: ["target","value"],
                tags: ["generic","collection"],
                examples: [
                    Value::String("b := Bag[]; Add[b, \"x\"]".into()),
                    Value::String("s := HashSet[]; Add[s, 1]".into()),
                    Value::String("q := Queue[]; Add[q, 7]".into()),
                    Value::String("st := Stack[]; Add[st, 9]".into()),
                    Value::String("pq := PriorityQueue[]; Add[pq, <|\"Key\"->2, \"Value\"->\"b\"|>]".into()),
                ]
            ),
            tool_spec!(
                "Info",
                summary: "Information about a handle (Graph, etc.)",
                params: ["target"],
                tags: ["generic","introspection"],
                examples: [
                    Value::String("Info[Graph[]]  ==> <|nodes->..., edges->...|>".into()),
                    Value::String("Info[DatasetFromRows[{<|a->1|>}]]  ==> <|Type->\"Dataset\", Rows->1, Columns->{\"a\"}|>".into()),
                    Value::String("Info[VectorStore[<|name->\"vs\"|>]]  ==> <|Type->\"VectorStore\", Name->\"vs\", Count->0|>".into()),
                    Value::String("Info[HashSet[{1,2,3}]]  ==> <|Type->\"Set\", Size->3|>".into()),
                    Value::String("Info[Queue[]]  ==> <|Type->\"Queue\", Size->0|>".into()),
                    Value::String("Info[Index[\"/tmp/idx.db\"]]  ==> <|indexPath->..., numDocs->...|>".into()),
                    Value::String("Info[Connect[\"mock://\"]]  ==> <|Type->\"Connection\", ...|>".into()),
                    Value::String("p := Popen[\"sleep\", {\"0.1\"}]; Info[p]  ==> <|Type->\"Process\", ...|>".into()),
                ]
            ),
            tool_spec!(
                "Length",
                summary: "Length of list/string/assoc or size of handle",
                params: ["x"],
                tags: ["generic"],
                examples: [
                    Value::String("Length[{1,2,3}]  ==> 3".into()),
                    Value::String("Length[\"ok\"]  ==> 2".into()),
                    Value::String("Length[Queue[]]  ==> 0".into()),
                    Value::String("Length[DatasetFromRows[{<|a->1|>}]]  ==> 1".into()),
                    Value::String("vs := VectorStore[<|name->\"vs\"|>]; Length[vs]  ==> 0".into()),
                ]
            ),
            tool_spec!(
                "EmptyQ",
                summary: "Is the subject empty? (lists, strings, assocs, handles)",
                params: ["x"],
                tags: ["generic","predicate"],
                examples: [
                    Value::String("EmptyQ[{}]  ==> True".into()),
                    Value::String("EmptyQ[\"\"]  ==> True".into()),
                    Value::String("EmptyQ[Queue[]]  ==> True".into()),
                    Value::String("EmptyQ[DatasetFromRows[{}]]  ==> True".into()),
                    Value::String("vs := VectorStore[<|name->\"vs\"|>]; EmptyQ[vs]  ==> True".into()),
                ]
            ),
            tool_spec!(
                "Count",
                summary: "Count items/elements (lists, assocs, Bag/VectorStore)",
                params: ["x"],
                tags: ["generic","aggregate"],
                examples: [
                    Value::String("Count[{1,2,3}]  ==> 3".into()),
                    Value::String("Count[<|a->1,b->2|>]  ==> 2".into()),
                    Value::String("Count[VectorStore[\"sqlite:///tmp/vs.db\"]]".into()),
                    Value::String("Count[DatasetFromRows[{<|a->1|>,<|a->2|>}]]  ==> 2".into()),
                ]
            ),
            tool_spec!(
                "Search",
                summary: "Search within a store or index (VectorStore, Index)",
                params: ["target","query","opts?"],
                tags: ["generic","search"],
                examples: [
                    Value::String("Search[VectorStore[<|name->\"vs\"|>], {0.1,0.2,0.3}]".into()),
                    Value::String("idx := Index[\"/tmp/idx.db\"]; Search[idx, \"foo\"]".into()),
                ]
            ),
            tool_spec!(
                "Close",
                summary: "Close an open handle (cursor, channel)",
                params: ["handle"],
                tags: ["generic","lifecycle"],
                examples: [
                    Value::String("Close[SQLCursor[conn, \"SELECT 1\"]]".into()),
                    Value::String("p := Popen[\"sleep\", {\"0.1\"}]; Close[p]".into()),
                ]
            ),
            tool_spec!(
                "Columns",
                summary: "Columns/keys for Frame/Dataset/assoc rows",
                params: ["subject"],
                tags: ["generic","schema","frame","dataset","assoc"],
                examples: [
                    Value::String("Columns[ds]".into()),
                    Value::String("Columns[f]".into()),
                    Value::String("Columns[{<|a->1,b->2|>,<|a->3|>}]  ==> {\"a\",\"b\"}".into()),
                    Value::String("Columns[<|a->1,b->2|>]  ==> {\"a\",\"b\"}".into()),
                ]
            ),
            tool_spec!(
                "Contains",
                summary: "Membership test for strings/lists/sets/assocs",
                params: ["container","item"],
                tags: ["generic","predicate"],
                examples: [
                    Value::String("Contains[\"foobar\", \"bar\"]  ==> True".into()),
                    Value::String("Contains[{1,2,3}, 2]  ==> True".into()),
                    Value::String("Contains[<|a->1|>, \"a\"]  ==> True".into()),
                    Value::String("s := HashSet[{1,2}]; Contains[s, 3]  ==> False".into()),
                ]
            ),
            tool_spec!(
                "MemberQ",
                summary: "Alias: membership predicate",
                params: ["container","item"],
                tags: [
                    "generic","predicate"
                ]
            ),
            tool_spec!(
                "ContainsQ",
                summary: "Alias: membership predicate",
                params: ["container","item"],
                tags: ["generic","predicate"],
                examples: [
                    Value::String("ContainsQ[{1,2,3}, 2]  ==> True".into()),
                ]
            ),
            tool_spec!(
                "ContainsKeyQ",
                summary: "Key membership for assoc/rows/Dataset/Frame",
                params: ["subject","key"],
                tags: ["generic","predicate","schema"],
                examples: [
                    Value::String("ContainsKeyQ[<|a->1|>, \"a\"]  ==> True".into()),
                    Value::String("ContainsKeyQ[{<|a->1|>,<|b->2|>}, \"b\"]  ==> True".into()),
                    Value::String("ContainsKeyQ[ds, \"col\"]".into()),
                    Value::String("ContainsKeyQ[f, \"col\"]".into()),
                ]
            ),
            tool_spec!(
                "HasKeyQ",
                summary: "Alias: key membership predicate",
                params: ["subject","key"],
                tags: ["generic","predicate","schema"]
            ),
            tool_spec!(
                "Select",
                summary: "Select/compute fields for Dataset/Frame or assoc rows",
                params: ["subject","spec"],
                tags: ["generic","dataset","frame","assoc"],
                examples: [
                    Value::String("Select[ds, {\"a\", \"b\"}]".into()),
                    Value::String("Select[f, <|\"a2\"->#a*2 &|>]".into()),
                    Value::String("Select[{<|a->1,b->2|>}, {\"a\"}]  ==> {<|a->1|>}".into()),
                ]
            ),
            tool_spec!(
                "Filter",
                summary: "Filter Dataset, Frame, or list with a predicate",
                params: ["subject","pred"],
                tags: ["generic","dataset","frame","list"],
                examples: [
                    Value::String("Filter[ds, #a>1 &]".into()),
                    Value::String("Filter[f, #a>1 &]".into()),
                    Value::String("Filter[{1,2,3,4}, OddQ]  ==> {1,3}".into()),
                ]
            ),
            tool_spec!(
                "Head",
                summary: "Take first n items/rows (list, assoc, Dataset, Frame)",
                params: ["subject","n?"],
                tags: ["generic","inspect"],
                examples: [
                    Value::String("Head[{1,2,3}, 2]  ==> {1,2}".into()),
                    Value::String("Head[ds, 5]".into()),
                    Value::String("Head[f, 5]".into()),
                ]
            ),
            tool_spec!(
                "Tail",
                summary: "Take last n items/rows (list, assoc, Dataset, Frame)",
                params: ["subject","n?"],
                tags: ["generic","inspect"],
                examples: [
                    Value::String("Tail[{1,2,3}, 2]  ==> {2,3}".into()),
                    Value::String("Tail[ds, 5]".into()),
                    Value::String("Tail[f, 5]".into()),
                ]
            ),
            tool_spec!(
                "Offset",
                summary: "Skip first n items/rows (list, Dataset, Frame)",
                params: ["subject","n"],
                tags: ["generic","transform"],
                examples: [
                    Value::String("Offset[{1,2,3}, 1]  ==> {2,3}".into()),
                    Value::String("Offset[ds, 10]".into()),
                    Value::String("Offset[f, 10]".into()),
                ]
            ),
            tool_spec!(
                "Sort",
                summary: "Sort list, list-of-assoc, Dataset or Frame",
                params: ["subject","by?"],
                tags: ["generic","sort"],
                examples: [
                    Value::String("Sort[{3,1,2}]  ==> {1,2,3}".into()),
                    Value::String("Sort[rows, {\"a\"}]".into()),
                    Value::String("Sort[ds, {\"a\"}]".into()),
                    Value::String("Sort[f, {\"a\"}]".into()),
                ]
            ),
            tool_spec!(
                "Distinct",
                summary: "Distinct for lists, list-of-assoc, Dataset, or Frame",
                params: ["subject","cols?"],
                tags: ["generic","distinct"],
                examples: [
                    Value::String("Distinct[{1,1,2}]  ==> {1,2}".into()),
                    Value::String("Distinct[rows, {\"a\"}]".into()),
                    Value::String("Distinct[ds, {\"a\"}]".into()),
                    Value::String("Distinct[f, {\"a\"}]".into()),
                ]
            ),
            tool_spec!(
                "Import",
                summary: "Import data from path/URL into Frame/Dataset/Value",
                params: ["source","opts?"],
                tags: ["generic","import","io","frame","dataset"],
                examples: [
                    Value::String("Import[\"data.csv\"]  (* Frame *)".into()),
                    Value::String("Import[\"data.csv\", <|Target->\"Dataset\"|>]".into()),
                    Value::String("Import[\"data.jsonl\", <|Type->\"JSONL\"|> ]".into()),
                ]
            ),
            tool_spec!(
                "ImportString",
                summary: "Parse in-memory string into Frame/Dataset/Value",
                params: ["content","opts?"],
                tags: ["generic","import","io"],
                examples: [
                    Value::String("ImportString[\"a,b\\n1,2\", <|Type->\"CSV\"|>]".into()),
                    Value::String("ImportString[\"[{\\\"a\\\":1}]\", <|Type->\"JSON\", Target->\"Frame\"|>]".into()),
                ]
            ),
            tool_spec!(
                "ImportBytes",
                summary: "Parse byte buffer using Type (text/json/etc.)",
                params: ["bytes","opts?"],
                tags: ["generic","import","io"],
                examples: [
                    Value::String("ImportBytes[bytes, <|Type->\"Text\"|>]".into()),
                ]
            ),
            tool_spec!(
                "Export",
                summary: "Export Frame/Dataset/Value to destination (csv/json)",
                params: ["value","dest","opts?"],
                tags: ["generic","export","io"],
                examples: [
                    Value::String("Export[f, \"out.csv\"]".into()),
                    Value::String("Export[rows, \"out.json\"]".into()),
                ]
            ),
            tool_spec!(
                "Sniff",
                summary: "Sniff source to suggest Type and options",
                params: ["source"],
                tags: ["generic","import","io"],
                examples: [
                    Value::String("Sniff[\"data.csv\"]  ==> <|Type->\"CSV\", Delimiter->\",\", Header->True|>".into()),
                    Value::String("Sniff[\"file.jsonl\"]  ==> <|Type->\"JSONL\"|>".into()),
                ]
            ),
            tool_spec!(
                "Describe",
                summary: "Describe data/handles or define test suite",
                params: ["subject|name","items?","opts?"],
                tags: ["generic","introspection","stats","testing"],
                examples: [
                    Value::String("Describe[f]  (* FrameDescribe *)".into()),
                    Value::String("Describe[ds]  (* Dataset stats *)".into()),
                    Value::String("Describe[g]   (* GraphInfo *)".into()),
                    Value::String("Describe[\"Math\", {It[\"adds\", 1+1==2]}]  (* test suite *)".into()),
                ]
            ),
            tool_spec!(
                "Keys",
                summary: "Keys/columns for Assoc/rows/Dataset/Frame",
                params: ["subject"],
                tags: ["generic","schema","assoc","dataset","frame"],
                examples: [
                    Value::String("Keys[<|a->1,b->2|>]  ==> {a,b}".into()),
                    Value::String("Keys[{<|a->1|>,<|b->2|>}]  ==> {a,b}".into()),
                    Value::String("Keys[ds] (* Columns *)".into()),
                    Value::String("Keys[f]  (* Columns *)".into()),
                ]
            ),
            tool_spec!(
                "SortBy",
                summary: "Sort list by key or assoc by derived key",
                params: ["f","subject"],
                tags: ["generic","sort"],
                examples: [
                    Value::String("SortBy[StringLength, {\"a\",\"bbb\",\"cc\"}]  ==> {\"a\",\"cc\",\"bbb\"}".into()),
                    Value::String("SortBy[(k,v)->k, <|\"b\"->2, \"a\"->1|>]  ==> <|\"a\"->1, \"b\"->2|>".into()),
                ]
            ),
            tool_spec!(
                "Shape",
                summary: "Shape of a tensor",
                params: ["x"],
                tags: ["tensor"],
                examples: [ Value::String("Shape[Tensor[{{1,2,3},{4,5,6}}]]  ==> {2, 3}".into()) ]
            ),
            tool_spec!(
                "Reshape",
                summary: "Reshape a tensor to given dims (supports -1)",
                params: ["x","dims"],
                tags: ["tensor"],
                examples: [ Value::String("Reshape[Tensor[{{1,2,3},{4,5,6}}], {3,2}]".into()) ]
            ),
            tool_spec!(
                "Relu",
                summary: "ReLU activation: tensor op or zero-arg layer",
                params: ["x?"],
                tags: ["tensor","nn","activation"],
                examples: [
                    Value::String("Relu[{-1,2,-3}]  ==> {0,2,0}".into()),
                    Value::String("Relu[]  (* layer spec *)".into()),
                ]
            ),
            tool_spec!(
                "Softmax",
                summary: "Softmax activation: zero-arg layer (tensor variant TBD)",
                params: ["x?","opts?"],
                tags: ["nn","activation"],
                examples: [ Value::String("Softmax[]  (* layer spec *)".into()) ]
            ),
            tool_spec!(
                "Sigmoid",
                summary: "Sigmoid activation: tensor op or zero-arg layer",
                params: ["x?"],
                tags: ["tensor","nn","activation"],
                examples: [
                    Value::String("Sigmoid[{-1,0,1}]  ==> {~0.2689, 0.5, ~0.7311}".into()),
                    Value::String("Sigmoid[]  (* layer spec *)".into()),
                ]
            ),
            tool_spec!(
                "Gelu",
                summary: "GELU activation (tanh approx): tensor op or zero-arg layer",
                params: ["x?"],
                tags: ["tensor","nn","activation"],
                examples: [ Value::String("Gelu[]  (* layer spec *)".into()) ]
            ),
            tool_spec!(
                "Train",
                summary: "Train a network or ML estimator (dispatched)",
                params: ["obj","data","opts?"],
                tags: ["nn","ml","train"],
                examples: [ Value::String("Train[Sequential[{Dense[<|Output->4|>], Relu[]}], ds, <|Epochs->5|>]".into()) ]
            ),
            tool_spec!(
                "Initialize",
                summary: "Initialize a network or estimator (dispatched)",
                params: ["obj","opts?"],
                tags: ["nn","net"],
                examples: [ Value::String("Initialize[Sequential[{Dense[<|Output->4|>]}]]".into()) ]
            ),
            tool_spec!(
                "Predict",
                summary: "Predict using a network or estimator (dispatched)",
                params: ["obj","x","opts?"],
                tags: ["nn","ml","inference"],
                examples: [
                    Value::String("Predict[Sequential[{Relu[]}], {1,2,3}]".into()),
                    Value::String("(* Tensor input *) net := Sequential[{Convolution2D[<|Output->4, KernelSize->3, Padding->1, InputChannels->1, Height->28, Width->28|>]}]; Shape[Predict[net, Tensor[(* 1x28x28 data *)]]]  ==> {4,28,28}".into())
                ]
            ),
            tool_spec!(
                "Evaluate",
                summary: "Evaluate an ML model on data (dispatched)",
                params: ["model","data","opts?"],
                tags: ["ml","metrics"],
                examples: [ Value::String("Evaluate[model, val, <|Metrics->{Accuracy}|>]".into()) ]
            ),
            tool_spec!(
                "CrossValidate",
                summary: "Cross-validate estimator + data (dispatched)",
                params: ["obj","data","opts?"],
                tags: ["ml","cv"],
                examples: [ Value::String("CrossValidate[Classifier[<|Method->\"Logistic\"|>], train, <|k->5|>]".into()) ]
            ),
            tool_spec!(
                "Tune",
                summary: "Hyperparameter search for estimator (dispatched)",
                params: ["obj","data","opts?"],
                tags: ["ml","tune"],
                examples: [ Value::String("Tune[Regressor[<|Method->\"Linear\"|>], train, <|SearchSpace-><|L2->{0.0,1e-2}|>|>]".into()) ]
            ),
            tool_spec!(
                "Property",
                summary: "Property of a network or ML estimator (dispatched)",
                params: ["obj","key"],
                tags: ["nn","ml","introspect"],
                examples: [ Value::String("Property[Sequential[{Dense[]}], \"Kind\"]".into()) ]
            ),
            tool_spec!(
                "Summary",
                summary: "Summary of a network or estimator (dispatched)",
                params: ["obj"],
                tags: ["nn","introspect"],
                examples: [ Value::String("Summary[Sequential[{Dense[]}]]".into()) ]
            ),
        ];
        add_specs(specs);
    }
}

type NativeFn = fn(&mut Evaluator, Vec<Value>) -> Value;

fn is_dataset_handle(v: &Value) -> bool {
    if let Value::Assoc(m) = v {
        if let Some(Value::String(t)) = m.get("__type") {
            return t == "Dataset";
        }
    }
    false
}

fn is_frame_handle(v: &Value) -> bool {
    if let Value::Assoc(m) = v {
        if let Some(Value::String(t)) = m.get("__type") {
            return t == "Frame";
        }
    }
    false
}

fn is_packed_array(v: &Value) -> bool { matches!(v, Value::PackedArray { .. }) }

fn transpose_dispatch(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() {
        // Layer constructor form: Transpose[]
        return ev.eval(Value::Expr { head: Box::new(Value::Symbol("__TransposeLayer".into())), args });
    }
    let first = ev.eval(args[0].clone());
    if is_packed_array(&first) {
        return ev.eval(Value::Expr { head: Box::new(Value::Symbol("NDTranspose".into())), args });
    }
    ev.eval(Value::Expr { head: Box::new(Value::Symbol("__ListTranspose".into())), args })
}

fn argmax_dispatch(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() { return Value::Expr { head: Box::new(Value::Symbol("ArgMax".into())), args }; }
    let first = ev.eval(args[0].clone());
    if is_packed_array(&first) {
        return ev.eval(Value::Expr { head: Box::new(Value::Symbol("NDArgMax".into())), args });
    }
    ev.eval(Value::Expr { head: Box::new(Value::Symbol("__ListArgMax".into())), args })
}

fn train_dispatch(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() { return Value::Expr { head: Box::new(Value::Symbol("Train".into())), args }; }
    let obj = ev.eval(args[0].clone());
    if let Value::Assoc(m) = &obj {
        if matches!(m.get("__type"), Some(Value::String(t)) if t=="Net") {
            return ev.eval(Value::Expr { head: Box::new(Value::Symbol("NetTrain".into())), args });
        }
        if matches!(m.get("__type"), Some(Value::String(t)) if t=="MLEstimator") {
            // Train[estimator, data, opts?] -> Classify/Predict/Cluster
            let task = m.get("Task").and_then(|v| match v { Value::String(s) | Value::Symbol(s) => Some(s.clone()), _ => None }).unwrap_or("Classification".into());
            let method = m.get("Method").cloned();
            let data = if args.len() >= 2 { ev.eval(args[1].clone()) } else { Value::List(vec![]) };
            let mut opts_map: std::collections::HashMap<String, Value> = std::collections::HashMap::new();
            if args.len() >= 3 {
                if let Value::Assoc(mm) = ev.eval(args[2].clone()) { opts_map = mm; }
            }
            if let Some(mv) = method { opts_map.insert("Method".into(), mv); }
            let opts_v = Value::Assoc(opts_map);
            let head = if task.eq_ignore_ascii_case("Regression") { "Predict" } else if task.eq_ignore_ascii_case("Clustering") { "Cluster" } else { "Classify" };
            return ev.eval(Value::Expr { head: Box::new(Value::Symbol(head.into())), args: vec![data, opts_v] });
        }
    }
    Value::Expr { head: Box::new(Value::Symbol("Train".into())), args }
}

fn initialize_dispatch(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() { return Value::Expr { head: Box::new(Value::Symbol("Initialize".into())), args }; }
    let obj = ev.eval(args[0].clone());
    if let Value::Assoc(m) = &obj {
        if matches!(m.get("__type"), Some(Value::String(t)) if t=="Net") {
            return ev.eval(Value::Expr { head: Box::new(Value::Symbol("NetInitialize".into())), args });
        }
    }
    Value::Expr { head: Box::new(Value::Symbol("Initialize".into())), args }
}

fn property_dispatch(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() < 2 { return Value::Expr { head: Box::new(Value::Symbol("Property".into())), args }; }
    let obj = ev.eval(args[0].clone());
    if let Value::Assoc(m) = &obj {
        if matches!(m.get("__type"), Some(Value::String(t)) if t=="Net") {
            return ev.eval(Value::Expr { head: Box::new(Value::Symbol("NetProperty".into())), args });
        }
        if matches!(m.get("__type"), Some(Value::String(t)) if t=="MLModel") {
            return ev.eval(Value::Expr { head: Box::new(Value::Symbol("MLProperty".into())), args });
        }
    }
    Value::Expr { head: Box::new(Value::Symbol("Property".into())), args }
}

fn summary_dispatch(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() { return Value::Expr { head: Box::new(Value::Symbol("Summary".into())), args }; }
    let obj = ev.eval(args[0].clone());
    if let Value::Assoc(m) = &obj {
        if matches!(m.get("__type"), Some(Value::String(t)) if t=="Net") {
            return ev.eval(Value::Expr { head: Box::new(Value::Symbol("NetSummary".into())), args });
        }
    }
    Value::Expr { head: Box::new(Value::Symbol("Summary".into())), args }
}

fn clip_dispatch(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // Clip[x, {min,max}] or Clip[x, min, max]
    if args.is_empty() { return Value::Expr { head: Box::new(Value::Symbol("Clip".into())), args }; }
    let x = ev.eval(args[0].clone());
    if is_packed_array(&x) {
        match args.as_slice() {
            [_, bounds] => {
                // bounds may be {min,max}
                let b = ev.eval(bounds.clone());
                if let Value::List(ref vs) = b {
                    if vs.len() == 2 {
                        return ev.eval(Value::Expr { head: Box::new(Value::Symbol("NDClip".into())), args: vec![x, vs[0].clone(), vs[1].clone()] });
                    }
                }
                Value::Expr { head: Box::new(Value::Symbol("Clip".into())), args: vec![x, b] }
            }
            [_, minv, maxv] => {
                let minv = ev.eval(minv.clone());
                let maxv = ev.eval(maxv.clone());
                ev.eval(Value::Expr { head: Box::new(Value::Symbol("NDClip".into())), args: vec![x, minv, maxv] })
            }
            _ => Value::Expr { head: Box::new(Value::Symbol("Clip".into())), args: vec![x] },
        }
    } else {
        // Defer to math Clip
        let mut real_args = vec![x];
        real_args.extend(args.into_iter().skip(1).map(|a| ev.eval(a)));
        ev.eval(Value::Expr { head: Box::new(Value::Symbol("__MathClip".into())), args: real_args })
    }
}

fn relu_dispatch(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() {
        // Layer constructor form: Relu[] => __ActivationLayer["Relu"]
        return ev.eval(Value::Expr {
            head: Box::new(Value::Symbol("__ActivationLayer".into())),
            args: vec![Value::String("Relu".into())],
        });
    }
    if args.len() != 1 { return Value::Expr { head: Box::new(Value::Symbol("Relu".into())), args }; }
    let x = ev.eval(args[0].clone());
    if is_packed_array(&x) {
        return ev.eval(Value::Expr { head: Box::new(Value::Symbol("NDRelu".into())), args: vec![x] });
    }
    // Scalar numeric fallback: max(0,x)
    match x {
        Value::Integer(n) => Value::Integer(n.max(0)),
        Value::Real(r) => Value::Real(if r < 0.0 { 0.0 } else { r }),
        _ => Value::Expr { head: Box::new(Value::Symbol("Relu".into())), args: vec![x] },
    }
}

fn power_dispatch(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // Power[a, b] -> NDPow when any tensor present else math Power
    if args.len() != 2 { return ev.eval(Value::Expr { head: Box::new(Value::Symbol("__MathPower".into())), args }); }
    let a = ev.eval(args[0].clone());
    let b = ev.eval(args[1].clone());
    if is_packed_array(&a) || is_packed_array(&b) {
        return ev.eval(Value::Expr { head: Box::new(Value::Symbol("NDPow".into())), args: vec![a, b] });
    }
    ev.eval(Value::Expr { head: Box::new(Value::Symbol("__MathPower".into())), args: vec![a, b] })
}

fn exp_dispatch(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 1 { return ev.eval(Value::Expr { head: Box::new(Value::Symbol("__MathExp".into())), args }); }
    let x = ev.eval(args[0].clone());
    if is_packed_array(&x) { return ev.eval(Value::Expr { head: Box::new(Value::Symbol("NDExp".into())), args: vec![x] }); }
    ev.eval(Value::Expr { head: Box::new(Value::Symbol("__MathExp".into())), args: vec![x] })
}

fn log_dispatch(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 1 { return ev.eval(Value::Expr { head: Box::new(Value::Symbol("__MathLog".into())), args }); }
    let x = ev.eval(args[0].clone());
    if is_packed_array(&x) { return ev.eval(Value::Expr { head: Box::new(Value::Symbol("NDLog".into())), args: vec![x] }); }
    ev.eval(Value::Expr { head: Box::new(Value::Symbol("__MathLog".into())), args: vec![x] })
}

fn sqrt_dispatch(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 1 { return ev.eval(Value::Expr { head: Box::new(Value::Symbol("__MathSqrt".into())), args }); }
    let x = ev.eval(args[0].clone());
    if is_packed_array(&x) { return ev.eval(Value::Expr { head: Box::new(Value::Symbol("NDSqrt".into())), args: vec![x] }); }
    ev.eval(Value::Expr { head: Box::new(Value::Symbol("__MathSqrt".into())), args: vec![x] })
}

fn sin_dispatch(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 1 { return ev.eval(Value::Expr { head: Box::new(Value::Symbol("__MathSin".into())), args }); }
    let x = ev.eval(args[0].clone());
    if is_packed_array(&x) { return ev.eval(Value::Expr { head: Box::new(Value::Symbol("NDSin".into())), args: vec![x] }); }
    ev.eval(Value::Expr { head: Box::new(Value::Symbol("__MathSin".into())), args: vec![x] })
}

fn cos_dispatch(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 1 { return ev.eval(Value::Expr { head: Box::new(Value::Symbol("__MathCos".into())), args }); }
    let x = ev.eval(args[0].clone());
    if is_packed_array(&x) { return ev.eval(Value::Expr { head: Box::new(Value::Symbol("NDCos".into())), args: vec![x] }); }
    ev.eval(Value::Expr { head: Box::new(Value::Symbol("__MathCos".into())), args: vec![x] })
}

fn tanh_dispatch(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() {
        // Layer constructor form: Tanh[]
        return ev.eval(Value::Expr {
            head: Box::new(Value::Symbol("__ActivationLayer".into())),
            args: vec![Value::String("Tanh".into())],
        });
    }
    if args.len() != 1 { return ev.eval(Value::Expr { head: Box::new(Value::Symbol("__MathTanh".into())), args }); }
    let x = ev.eval(args[0].clone());
    if is_packed_array(&x) { return ev.eval(Value::Expr { head: Box::new(Value::Symbol("NDTanh".into())), args: vec![x] }); }
    ev.eval(Value::Expr { head: Box::new(Value::Symbol("__MathTanh".into())), args: vec![x] })
}

fn sigmoid_dispatch(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // Zero-arg: layer constructor; One-arg: tensor/scalar elementwise
    if args.is_empty() {
        return ev.eval(Value::Expr {
            head: Box::new(Value::Symbol("__ActivationLayer".into())),
            args: vec![Value::String("Sigmoid".into())],
        });
    }
    if args.len() != 1 {
        return Value::Expr { head: Box::new(Value::Symbol("Sigmoid".into())), args };
    }
    let x = ev.eval(args[0].clone());
    if is_packed_array(&x) {
        // NDMap[x, (#|->1/(1+Exp[-#]))]
        let body = Value::expr(
            Value::symbol("Divide"),
            vec![
                Value::Integer(1),
                Value::expr(
                    Value::symbol("Plus"),
                    vec![
                        Value::Integer(1),
                        Value::expr(
                            Value::symbol("Exp"),
                            vec![Value::expr(Value::symbol("Times"), vec![Value::Integer(-1), Value::slot(None)])],
                        ),
                    ],
                ),
            ],
        );
        let f = Value::PureFunction { params: None, body: Box::new(body) };
        return ev.eval(Value::Expr { head: Box::new(Value::Symbol("NDMap".into())), args: vec![x, f] });
    }
    match x {
        Value::Integer(n) => {
            let xr = n as f64;
            Value::Real(1.0 / (1.0 + (-xr).exp()))
        }
        Value::Real(r) => Value::Real(1.0 / (1.0 + (-r).exp())),
        _ => Value::Expr { head: Box::new(Value::Symbol("Sigmoid".into())), args: vec![x] },
    }
}

fn gelu_dispatch(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // Zero-arg: layer constructor; One-arg: elementwise GELU approx via tanh
    if args.is_empty() {
        return ev.eval(Value::Expr {
            head: Box::new(Value::Symbol("__ActivationLayer".into())),
            args: vec![Value::String("GELU".into())],
        });
    }
    if args.len() != 1 {
        return Value::Expr { head: Box::new(Value::Symbol("Gelu".into())), args };
    }
    let x = ev.eval(args[0].clone());
    if is_packed_array(&x) {
        // 0.5*x*(1 + tanh(0.79788456*(x + 0.044715*x^3)))
        let tanh_body = Value::expr(
            Value::symbol("Times"),
            vec![
                Value::Real(0.5),
                Value::slot(None),
                Value::expr(
                    Value::symbol("Plus"),
                    vec![
                        Value::Integer(1),
                        Value::expr(
                            Value::symbol("Tanh"),
                            vec![Value::expr(
                                Value::symbol("Times"),
                                vec![
                                    Value::Real(0.79788456),
                                    Value::expr(
                                        Value::symbol("Plus"),
                                        vec![
                                            Value::slot(None),
                                            Value::expr(
                                                Value::symbol("Times"),
                                                vec![
                                                    Value::Real(0.044715),
                                                    Value::expr(Value::symbol("Power"), vec![Value::slot(None), Value::Integer(3)]),
                                                ],
                                            ),
                                        ],
                                    ),
                                ],
                            )],
                        ),
                    ],
                ),
            ],
        );
        let f = Value::PureFunction { params: None, body: Box::new(tanh_body) };
        return ev.eval(Value::Expr { head: Box::new(Value::Symbol("NDMap".into())), args: vec![x, f] });
    }
    match x {
        Value::Integer(n) => {
            let u = n as f64;
            let k = (0.79788456 * (u + 0.044715 * u * u * u)).tanh();
            Value::Real(0.5 * u * (1.0 + k))
        }
        Value::Real(u) => {
            let k = (0.79788456 * (u + 0.044715 * u * u * u)).tanh();
            Value::Real(0.5 * u * (1.0 + k))
        }
        _ => Value::Expr { head: Box::new(Value::Symbol("Gelu".into())), args: vec![x] },
    }
}
fn softmax_dispatch(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // Softmax[] (layer) or Softmax[tensor, Axis->…] in future
    if args.is_empty() {
        return ev.eval(Value::Expr {
            head: Box::new(Value::Symbol("__ActivationLayer".into())),
            args: vec![Value::String("Softmax".into())],
        });
    }
    // Tensor softmax not implemented; leave unevaluated or pass through
    Value::Expr { head: Box::new(Value::Symbol("Softmax".into())), args }
}

fn predict_dispatch(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() { return Value::Expr { head: Box::new(Value::Symbol("Predict".into())), args }; }
    let obj = ev.eval(args[0].clone());
    if let Value::Assoc(m) = &obj {
        if matches!(m.get("__type"), Some(Value::String(t)) if t=="Net") {
            return ev.eval(Value::Expr { head: Box::new(Value::Symbol("NetApply".into())), args });
        }
        if matches!(m.get("__type"), Some(Value::String(t)) if t=="MLModel") {
            return ev.eval(Value::Expr { head: Box::new(Value::Symbol("MLApply".into())), args });
        }
    }
    Value::Expr { head: Box::new(Value::Symbol("Predict".into())), args }
}

fn evaluate_dispatch(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() < 2 { return Value::Expr { head: Box::new(Value::Symbol("Evaluate".into())), args }; }
    let model = ev.eval(args[0].clone());
    if let Value::Assoc(m) = &model {
        if !matches!(m.get("__type"), Some(Value::String(t)) if t=="MLModel") { return Value::Expr { head: Box::new(Value::Symbol("Evaluate".into())), args }; }
    } else { return Value::Expr { head: Box::new(Value::Symbol("Evaluate".into())), args }; }
    // Read task via MLProperty[model, "Task"]
    let task = ev.eval(Value::Expr { head: Box::new(Value::Symbol("MLProperty".into())), args: vec![model.clone(), Value::String("Task".into())] });
    let data = ev.eval(args[1].clone());
    let opts = if args.len() >= 3 { ev.eval(args[2].clone()) } else { Value::Assoc(HashMap::new()) };
    let head = match task { Value::String(s) | Value::Symbol(s) if s.eq_ignore_ascii_case("Regression") => "PredictMeasurements", _ => "ClassifyMeasurements" };
    ev.eval(Value::Expr { head: Box::new(Value::Symbol(head.into())), args: vec![model, data, opts] })
}

fn crossvalidate_dispatch(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // CrossValidate[estimator|spec, data, opts?] -> MLCrossValidate[data, opts]
    if args.len() < 2 { return Value::Expr { head: Box::new(Value::Symbol("CrossValidate".into())), args }; }
    let data = ev.eval(args[1].clone());
    let opts = if args.len() >= 3 { ev.eval(args[2].clone()) } else { Value::Assoc(HashMap::new()) };
    ev.eval(Value::Expr { head: Box::new(Value::Symbol("MLCrossValidate".into())), args: vec![Value::Symbol("Null".into()), data, opts] })
}

fn tune_dispatch(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // Tune[estimator|spec, data, opts?] -> MLTune[data, opts]
    if args.len() < 2 { return Value::Expr { head: Box::new(Value::Symbol("Tune".into())), args }; }
    let data = ev.eval(args[1].clone());
    let opts = if args.len() >= 3 { ev.eval(args[2].clone()) } else { Value::Assoc(HashMap::new()) };
    ev.eval(Value::Expr { head: Box::new(Value::Symbol("MLTune".into())), args: vec![data, opts] })
}

fn describe_dispatch(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // Two+ args: BDD Suite sugar (mirror dev::describe_fn semantics)
    if args.len() >= 2 {
        let name = match &args[0] {
            Value::String(s) | Value::Symbol(s) => s.clone(),
            other => lyra_core::pretty::format_value(other),
        };
        let items_v = ev.eval(args[1].clone());
        let items = match items_v {
            Value::List(vs) => vs,
            other => vec![other],
        };
        let opts = if args.len() >= 3 { ev.eval(args[2].clone()) } else { Value::Assoc(HashMap::new()) };
        return Value::Assoc(HashMap::from([
            ("type".into(), Value::String("suite".into())),
            ("name".into(), Value::String(name)),
            ("items".into(), Value::List(items)),
            ("options".into(), opts),
        ]));
    }
    // One arg: data/handle describe
    if args.len() == 1 {
        let subj = ev.eval(args[0].clone());
        if is_frame_handle(&subj) {
            return ev.eval(Value::expr(Value::symbol("FrameDescribe"), vec![subj]));
        }
        if is_dataset_handle(&subj) {
            return ev.eval(Value::expr(Value::symbol("__DatasetDescribe"), vec![subj]));
        }
        // List-of-assoc rows: dataset-style describe
        if list_is_assoc_rows(&subj) {
            return ev.eval(Value::expr(Value::symbol("__DatasetDescribe"), vec![subj]));
        }
        // Text index association
        if let Some(path) = looks_like_index_assoc(&subj) {
            return ev.eval(Value::expr(Value::symbol("IndexInfo"), vec![Value::String(path)]));
        }
        if is_graph_handle(&subj) { return ev.eval(Value::expr(Value::symbol("GraphInfo"), vec![subj])); }
        if is_vector_store_handle(&subj) { return ev.eval(Value::expr(Value::symbol("Info"), vec![subj])); }
        if let Value::Assoc(m) = &subj { if matches!(m.get("__type"), Some(Value::String(t)) if t=="ContainersRuntime") { return ev.eval(Value::expr(Value::symbol("DescribeContainers"), vec![subj])); } }
        if is_conn_handle(&subj) {
            let info = ev.eval(Value::expr(Value::symbol("ConnectionInfo"), vec![subj.clone()]));
            let tables = ev.eval(Value::expr(Value::symbol("ListTables"), vec![subj]));
            let mut m = HashMap::new();
            m.insert("Info".into(), info);
            m.insert("Tables".into(), tables);
            return Value::Assoc(m);
        }
        if is_process_handle(&subj) { return ev.eval(Value::expr(Value::symbol("ProcessInfo"), vec![subj])); }
        if is_cursor_handle(&subj) { return ev.eval(Value::expr(Value::symbol("CursorInfo"), vec![subj])); }
        // Media/image describe for file paths
        if let Some(p) = str_of(&subj) {
            let exists = ev.eval(Value::expr(Value::symbol("FileExistsQ"), vec![Value::String(p.clone())]));
            if matches!(exists, Value::Boolean(true)) {
                if let Some(ext) = ext_lower(&p) {
                    let img_exts = ["png","jpg","jpeg","gif","bmp","webp"];
                    let media_exts = ["mp4","mov","mkv","avi","webm","mp3","wav","flac","ogg","m4a","aac"];
                    if img_exts.contains(&ext.as_str()) {
                        return ev.eval(Value::expr(Value::symbol("ImageInfo"), vec![Value::String(p)]));
                    }
                    if media_exts.contains(&ext.as_str()) {
                        return ev.eval(Value::expr(Value::symbol("MediaProbe"), vec![Value::String(p)]));
                    }
                    // Default for files: Stat
                    return ev.eval(Value::expr(Value::symbol("Stat"), vec![Value::String(p)]));
                }
                // No extension; still return Stat
                return ev.eval(Value::expr(Value::symbol("Stat"), vec![Value::String(p)]));
            }
        }
        if is_channel_handle(&subj) { return Value::expr(Value::symbol("Describe"), vec![subj]); }
        // Fallback: leave unevaluated
        return Value::expr(Value::symbol("Describe"), vec![subj]);
    }
    Value::Expr { head: Box::new(Value::Symbol("Describe".into())), args }
}

fn columns_dispatch(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 1 { return Value::expr(Value::symbol("Columns"), args); }
    let subj = ev.eval(args[0].clone());
    if is_frame_handle(&subj) {
        return ev.eval(Value::expr(Value::symbol("FrameColumns"), vec![subj]));
    }
    if is_dataset_handle(&subj) {
        return ev.eval(Value::expr(Value::symbol("Columns"), vec![subj]));
    }
    // Assoc: keys
    if let Value::Assoc(m) = &subj {
        let mut ks: Vec<String> = m.keys().cloned().collect();
        ks.sort();
        return Value::List(ks.into_iter().map(Value::String).collect());
    }
    // List of assoc rows: union of keys
    if let Value::List(rows) = &subj {
        let mut set: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
        for r in rows {
            if let Value::Assoc(m) = r { for k in m.keys() { set.insert(k.clone()); } }
        }
        return Value::List(set.into_iter().map(Value::String).collect());
    }
    Value::expr(Value::symbol("Columns"), vec![subj])
}

fn keys_dispatch(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 1 { return Value::expr(Value::symbol("Keys"), args); }
    let subj = ev.eval(args[0].clone());
    if is_frame_handle(&subj) {
        return ev.eval(Value::expr(Value::symbol("FrameColumns"), vec![subj]));
    }
    if is_dataset_handle(&subj) {
        return ev.eval(Value::expr(Value::symbol("Columns"), vec![subj]));
    }
    if let Value::Assoc(m) = &subj {
        let mut ks: Vec<String> = m.keys().cloned().collect();
        ks.sort();
        return Value::List(ks.into_iter().map(Value::String).collect());
    }
    if let Value::List(rows) = &subj {
        let mut set: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
        for r in rows {
            if let Value::Assoc(m) = r { for k in m.keys() { set.insert(k.clone()); } }
        }
        return Value::List(set.into_iter().map(Value::String).collect());
    }
    Value::expr(Value::symbol("Keys"), vec![subj])
}

fn get_dispatch(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [subj, key] | [subj, key, _] => {
            let s = ev.eval(subj.clone());
            if let Value::Assoc(m) = s {
                let k = ev.eval(key.clone());
                let dk = args.get(2).map(|v| ev.eval(v.clone()));
                if let Value::String(ks) | Value::Symbol(ks) = k {
                    if let Some(v) = m.get(&ks) {
                        return v.clone();
                    } else if let Some(defv) = dk { return defv; }
                    else { return Value::Symbol("Null".into()); }
                }
                return Value::Symbol("Null".into());
            }
        }
        _ => {}
    }
    Value::Expr { head: Box::new(Value::Symbol("Get".into())), args }
}

fn contains_dispatch(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 2 { return Value::expr(Value::symbol("Contains"), args); }
    let container = ev.eval(args[0].clone());
    let item = ev.eval(args[1].clone());
    // String contains substring
    if let Value::String(s) = &container {
        let needle = match &item { Value::String(t) | Value::Symbol(t) => t, _ => return Value::expr(Value::symbol("Contains"), vec![container, item]) };
        return Value::Boolean(s.contains(needle));
    }
    // List membership
    if let Value::List(xs) = &container { return Value::Boolean(xs.iter().any(|x| x == &item)); }
    // Assoc has key
    if let Value::Assoc(m) = &container {
        if let Value::String(k) | Value::Symbol(k) = &item { return Value::Boolean(m.contains_key(k)); }
        return Value::Boolean(false);
    }
    // Set membership
    if is_set_handle(&container) { return ev.eval(Value::expr(Value::symbol("SetMemberQ"), vec![container, item])); }
    Value::expr(Value::symbol("Contains"), vec![container, item])
}

fn contains_key_dispatch(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 2 { return Value::expr(Value::symbol("ContainsKeyQ"), args); }
    let subject = ev.eval(args[0].clone());
    let key = ev.eval(args[1].clone());
    // Assoc: check key directly
    if let Value::Assoc(m) = &subject {
        if let Value::String(s) | Value::Symbol(s) = &key { return Value::Boolean(m.contains_key(s)); }
        return Value::Boolean(false);
    }
    // Dataset/Frame/List-of-assoc rows: check Columns
    if is_frame_handle(&subject) || is_dataset_handle(&subject) || list_is_assoc_rows(&subject) {
        let cols_v = if is_frame_handle(&subject) {
            ev.eval(Value::expr(Value::symbol("FrameColumns"), vec![subject.clone()]))
        } else {
            ev.eval(Value::expr(Value::symbol("Columns"), vec![subject.clone()]))
        };
        let names: Vec<String> = match cols_v {
            Value::List(xs) => xs
                .into_iter()
                .filter_map(|v| match v { Value::String(c) | Value::Symbol(c) => Some(c), _ => None })
                .collect(),
            _ => Vec::new(),
        };
        match &key {
            Value::String(s) | Value::Symbol(s) => return Value::Boolean(names.iter().any(|c| c==s)),
            _ => return Value::Boolean(false),
        }
    }
    // Fallback: unevaluated
    Value::expr(Value::symbol("ContainsKeyQ"), vec![subject, key])
}

fn join_dispatch(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [a, b] => {
            let av = ev.eval(a.clone());
            let bv = ev.eval(b.clone());
            match (&av, &bv) {
                (Value::List(la), Value::List(lb)) => {
                    let mut out = la.clone();
                    out.extend(lb.clone());
                    Value::List(out)
                }
                _ => {
                    // If any side is a frame, delegate to FrameJoin with empty keys
                    if is_frame_handle(&av) || is_frame_handle(&bv) {
                        #[cfg(feature = "frame")]
                        {
                            return ev.eval(Value::expr(Value::symbol("FrameJoin"), vec![av, bv]));
                        }
                        #[cfg(not(feature = "frame"))]
                        {
                            return Value::expr(Value::symbol("Join"), vec![av, bv]);
                        }
                    }
                    // If any side is a dataset, delegate to dataset Join
                    if is_dataset_handle(&av) || is_dataset_handle(&bv) {
                        #[cfg(feature = "dataset")]
                        {
                            return crate::dataset::join_ds(ev, vec![av, bv, Value::List(vec![])]);
                        }
                        #[cfg(not(feature = "dataset"))]
                        {
                            return Value::expr(Value::symbol("Join"), vec![av, bv]);
                        }
                    }
                    // Fallback: keep unevaluated
                    Value::expr(Value::symbol("Join"), vec![av, bv])
                }
            }
        }
        // 3+ args: if any side is dataset, delegate; else leave unevaluated or try list semantics if both lists and third is not list.
        [a, b, rest @ ..] => {
            let av = ev.eval(a.clone());
            let bv = ev.eval(b.clone());
            if is_dataset_handle(&av) || is_dataset_handle(&bv) {
                let mut v = vec![av, bv];
                v.extend(rest.iter().cloned().map(|x| ev.eval(x)));
                #[cfg(feature = "dataset")]
                {
                    return crate::dataset::join_ds(ev, v);
                }
                #[cfg(not(feature = "dataset"))]
                {
                    return Value::expr(Value::symbol("Join"), v);
                }
            }
            if is_frame_handle(&av) || is_frame_handle(&bv) {
                let mut v = vec![av, bv];
                v.extend(rest.iter().cloned().map(|x| ev.eval(x)));
                #[cfg(feature = "frame")]
                {
                    return ev.eval(Value::expr(Value::symbol("FrameJoin"), v));
                }
                #[cfg(not(feature = "frame"))]
                {
                    return Value::expr(Value::symbol("Join"), v);
                }
            }
            Value::expr(Value::symbol("Join"), {
                let mut v = vec![av, bv];
                v.extend(rest.iter().cloned().map(|x| ev.eval(x)));
                v
            })
        }
        _ => Value::expr(Value::symbol("Join"), args),
    }
}

fn is_set_handle(v: &Value) -> bool {
    if let Value::Assoc(m) = v {
        if let Some(Value::String(t)) = m.get("__type") {
            return t == "Set";
        }
    }
    false
}

fn is_bag_handle(v: &Value) -> bool {
    if let Value::Assoc(m) = v {
        if let Some(Value::String(t)) = m.get("__type") {
            return t == "Bag";
        }
    }
    false
}

fn is_queue_handle(v: &Value) -> bool {
    if let Value::Assoc(m) = v {
        if let Some(Value::String(t)) = m.get("__type") {
            return t == "Queue";
        }
    }
    false
}

fn is_stack_handle(v: &Value) -> bool {
    if let Value::Assoc(m) = v {
        if let Some(Value::String(t)) = m.get("__type") {
            return t == "Stack";
        }
    }
    false
}

fn is_pq_handle(v: &Value) -> bool {
    if let Value::Assoc(m) = v {
        if let Some(Value::String(t)) = m.get("__type") {
            return t == "PriorityQueue";
        }
    }
    false
}

fn is_graph_handle(v: &Value) -> bool {
    if let Value::Assoc(m) = v {
        if let Some(Value::String(t)) = m.get("__type") {
            return t == "Graph";
        }
    }
    false
}

fn is_channel_handle(v: &Value) -> bool {
    if let Value::Expr { head, .. } = v {
        if let Value::Symbol(s) = &**head {
            return s == "ChannelId";
        }
    }
    false
}

fn is_process_handle(v: &Value) -> bool {
    if let Value::Assoc(m) = v {
        if let Some(Value::String(t)) = m.get("__type") {
            return t == "Process";
        }
    }
    false
}

fn is_cursor_handle(v: &Value) -> bool {
    if let Value::Assoc(m) = v {
        if let Some(Value::String(t)) = m.get("__type") {
            return t == "Cursor";
        }
    }
    false
}

fn is_conn_handle(v: &Value) -> bool {
    if let Value::Assoc(m) = v {
        if let Some(Value::String(t)) = m.get("__type") {
            return t == "Connection";
        }
    }
    false
}

fn is_vector_store_handle(v: &Value) -> bool {
    if let Value::Assoc(m) = v {
        if let Some(Value::String(t)) = m.get("__type") {
            return t == "VectorStore";
        }
    }
    false
}

fn looks_like_index_assoc(v: &Value) -> Option<String> {
    if let Value::Assoc(m) = v {
        if let Some(Value::String(p)) = m.get("indexPath") {
            return Some(p.clone());
        }
    }
    None
}

fn all_lists(vs: &[Value]) -> bool {
    vs.iter().all(|v| matches!(v, Value::List(_)))
}
fn list_is_assoc_rows(v: &Value) -> bool {
    if let Value::List(xs) = v {
        return xs.iter().all(|x| matches!(x, Value::Assoc(_)));
    }
    false
}

fn union_dispatch(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() {
        return Value::expr(Value::symbol("Union"), args);
    }
    let evald: Vec<Value> = args.into_iter().map(|a| ev.eval(a)).collect();
    if evald.iter().any(|v| is_frame_handle(v)) {
        if evald.iter().all(|v| is_frame_handle(v)) {
            #[cfg(feature = "frame")]
            {
                return ev.eval(Value::expr(Value::symbol("FrameUnion"), evald));
            }
            #[cfg(not(feature = "frame"))]
            {
                return Value::expr(Value::symbol("Union"), evald);
            }
        }
        // mixed types; leave unevaluated for now
        return Value::expr(Value::symbol("Union"), evald);
    }
    if evald.iter().any(|v| is_dataset_handle(v)) {
        #[cfg(feature = "dataset")]
        {
            return crate::dataset::union_general(ev, evald);
        }
        #[cfg(not(feature = "dataset"))]
        {
            return Value::expr(Value::symbol("Union"), evald);
        }
    }
    // If all args are lists and appear to be rows (assoc), delegate to dataset union_general for row-wise union-by-columns
    if !evald.is_empty() && all_lists(&evald) && evald.iter().any(|v| list_is_assoc_rows(v)) {
        #[cfg(feature = "dataset")]
        {
            return crate::dataset::union_general(ev, evald);
        }
        #[cfg(not(feature = "dataset"))]
        {
            return Value::expr(Value::symbol("Union"), evald);
        }
    }
    // If any are Set handles, use set-specific impl
    if evald.iter().any(|v| is_set_handle(v)) {
        return ev.eval(Value::expr(Value::symbol("__SetUnion"), evald));
    }
    // Default for lists: order-stable union
    if all_lists(&evald) {
        use std::collections::HashSet;
        let mut seen: HashSet<String> = HashSet::new();
        let mut out: Vec<Value> = Vec::new();
        for v in &evald {
            if let Value::List(xs) = v {
                for x in xs {
                    let k = lyra_runtime::eval::value_order_key(x);
                    if seen.insert(k) { out.push(x.clone()); }
                }
            }
        }
        return Value::List(out);
    }
    // Fallback: leave unevaluated
    Value::expr(Value::symbol("Union"), evald)
}

fn upsert_dispatch(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 2 { return Value::expr(Value::symbol("Upsert"), args); }
    let target = ev.eval(args[0].clone());
    let rows = ev.eval(args[1].clone());
    if is_graph_handle(&target) {
        let is_edge_assoc = |v: &Value| -> bool {
            if let Value::Assoc(m) = v { m.contains_key("src") || m.contains_key("Src") || m.contains_key("dst") || m.contains_key("Dst") } else { false }
        };
        let name = match &rows {
            Value::Assoc(_) if is_edge_assoc(&rows) => "UpsertEdges",
            Value::List(xs) if xs.iter().any(|x| is_edge_assoc(x)) => "UpsertEdges",
            _ => "UpsertNodes",
        };
        return ev.eval(Value::expr(Value::symbol(name), vec![target, rows]));
    }
    if is_vector_store_handle(&target) || matches!(target, Value::String(_) | Value::Symbol(_)) {
        return ev.eval(Value::expr(Value::symbol("VectorUpsert"), vec![target, rows]));
    }
    Value::expr(Value::symbol("Upsert"), vec![target, rows])
}

fn intersection_dispatch(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() { return Value::expr(Value::symbol("Intersection"), args); }
    let evald: Vec<Value> = args.into_iter().map(|a| ev.eval(a)).collect();
    // Sets: defer to set-specific impl
    if evald.iter().any(|v| is_set_handle(v)) {
        return ev.eval(Value::expr(Value::symbol("__SetIntersection"), evald));
    }
    // Lists: support binary intersection
    if evald.len() == 2 { if let (Value::List(la), Value::List(lb)) = (&evald[0], &evald[1]) {
        use std::collections::HashSet; let sb: HashSet<String> = lb.iter().map(|x| lyra_runtime::eval::value_order_key(x)).collect();
        let mut seen: HashSet<String> = HashSet::new(); let mut out = Vec::new();
        for x in la { let k = lyra_runtime::eval::value_order_key(x); if sb.contains(&k) && seen.insert(k.clone()) { out.push(x.clone()); } }
        return Value::List(out);
    }}
    Value::expr(Value::symbol("Intersection"), evald)
}

fn difference_dispatch(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() { return Value::expr(Value::symbol("Difference"), args); }
    let evald: Vec<Value> = args.into_iter().map(|a| ev.eval(a)).collect();
    // Sets: defer to set-specific impl
    if evald.iter().any(|v| is_set_handle(v)) {
        return ev.eval(Value::expr(Value::symbol("__SetDifference"), evald));
    }
    // Lists: support binary difference a \ b
    if evald.len() == 2 { if let (Value::List(la), Value::List(lb)) = (&evald[0], &evald[1]) {
        use std::collections::HashSet; let sb: HashSet<String> = lb.iter().map(|x| lyra_runtime::eval::value_order_key(x)).collect();
        let mut out = Vec::new(); for x in la { if !sb.contains(&lyra_runtime::eval::value_order_key(x)) { out.push(x.clone()); } }
        return Value::List(out);
    }}
    Value::expr(Value::symbol("Difference"), evald)
}

fn select_dispatch(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 2 { return Value::expr(Value::symbol("Select"), args); }
    let subj = ev.eval(args[0].clone());
    let spec = ev.eval(args[1].clone());
    if is_frame_handle(&subj) {
        return ev.eval(Value::expr(Value::symbol("FrameSelect"), vec![subj, spec]));
    }
    // Delegate to dataset's generalized select which also handles assocs and list-of-assocs
    #[cfg(feature = "dataset")]
    {
        return ev.eval(Value::expr(Value::symbol("__DatasetSelect"), vec![subj, spec]));
    }
    #[cfg(not(feature = "dataset"))]
    {
        Value::expr(Value::symbol("Select"), vec![subj, spec])
    }
}

fn drop_dispatch(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 2 { return Value::expr(Value::symbol("Drop"), args); }
    let subj = ev.eval(args[0].clone());
    let n = match ev.eval(args[1].clone()) { Value::Integer(k) => k, other => return Value::expr(Value::symbol("Drop"), vec![subj, other]) };
    if matches!(subj, Value::Assoc(_)) {
        return ev.eval(Value::expr(Value::symbol("__AssocDrop"), vec![subj, Value::Integer(n)]));
    }
    if let Value::List(items) = subj {
        let len = items.len() as i64;
        let k = if n >= 0 { n.min(len).max(0) } else { (-n).min(len).max(0) } as usize;
        let slice: Vec<Value> = if n >= 0 { items.into_iter().skip(k).collect() } else { items.into_iter().take((len as usize).saturating_sub(k)).collect() };
        return Value::List(slice);
    }
    Value::expr(Value::symbol("Drop"), vec![subj, Value::Integer(n)])
}

fn delete_dispatch(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 2 { return Value::expr(Value::symbol("Delete"), args); }
    let subj = ev.eval(args[0].clone());
    let spec = ev.eval(args[1].clone());
    if matches!(subj, Value::Assoc(_)) { return ev.eval(Value::expr(Value::symbol("__AssocDelete"), vec![subj, spec])); }
    if is_vector_store_handle(&subj) || matches!(subj, Value::String(_) | Value::Symbol(_)) {
        return ev.eval(Value::expr(Value::symbol("VectorDelete"), vec![subj, spec]));
    }
    Value::expr(Value::symbol("Delete"), vec![subj, spec])
}

fn write_dispatch(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 3 { return Value::expr(Value::symbol("Write"), args); }
    let subj = ev.eval(args[0].clone());
    let key = ev.eval(args[1].clone());
    let val = ev.eval(args[2].clone());
    if matches!(subj, Value::Assoc(_)) { return ev.eval(Value::expr(Value::symbol("__AssocSet"), vec![subj, key, val])); }
    Value::expr(Value::symbol("Write"), vec![subj, key, val])
}

fn reset_dispatch(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 1 { return Value::expr(Value::symbol("Reset"), args); }
    let subj = ev.eval(args[0].clone());
    if is_vector_store_handle(&subj) || matches!(subj, Value::String(_) | Value::Symbol(_)) {
        return ev.eval(Value::expr(Value::symbol("VectorReset"), vec![subj]));
    }
    Value::expr(Value::symbol("Reset"), vec![subj])
}

fn filter_dispatch(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 2 { return Value::expr(Value::symbol("Filter"), args); }
    let subj = ev.eval(args[0].clone());
    let pred = ev.eval(args[1].clone());
    if is_frame_handle(&subj) {
        return ev.eval(Value::expr(Value::symbol("FrameFilter"), vec![subj, pred]));
    }
    if is_dataset_handle(&subj) {
        #[cfg(feature = "dataset")]
        { return ev.eval(Value::expr(Value::symbol("FilterRows"), vec![subj, pred])); }
        #[cfg(not(feature = "dataset"))]
        { return Value::expr(Value::symbol("Filter"), vec![subj, pred]); }
    }
    // Generic list filtering: Filter[list, pred]
    if let Value::List(items) = subj {
        let mut out: Vec<Value> = Vec::new();
        for it in items.into_iter() {
            let keep = ev.eval(Value::Expr { head: Box::new(pred.clone()), args: vec![it.clone()] });
            if matches!(keep, Value::Boolean(true)) { out.push(it); }
        }
        return Value::List(out);
    }
    Value::expr(Value::symbol("Filter"), vec![subj, pred])
}

fn head_dispatch(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() { return Value::expr(Value::symbol("Head"), args); }
    let subj = ev.eval(args[0].clone());
    let n = if args.len()>=2 { ev.eval(args[1].clone()) } else { Value::Integer(10) };
    if is_frame_handle(&subj) {
        return ev.eval(Value::expr(Value::symbol("FrameHead"), vec![subj, n]));
    }
    #[cfg(feature = "dataset")]
    { return ev.eval(Value::expr(Value::symbol("__DatasetHead"), vec![subj, n])); }
    #[cfg(not(feature = "dataset"))]
    {
        // simple list/assoc fallback
        return match (subj, n) {
            (Value::List(items), Value::Integer(k)) => Value::List(items.into_iter().take(k.max(0) as usize).collect()),
            (Value::Assoc(m), Value::Integer(k)) => { let mut keys: Vec<String> = m.keys().cloned().collect(); keys.sort(); let mut out=HashMap::new(); for key in keys.into_iter().take(k.max(0) as usize){ if let Some(v)=m.get(&key){ out.insert(key,v.clone()); } } Value::Assoc(out) },
            (s, k) => Value::expr(Value::symbol("Head"), vec![s,k]),
        };
    }
}

fn tail_dispatch(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() { return Value::expr(Value::symbol("Tail"), args); }
    let subj = ev.eval(args[0].clone());
    let n = if args.len()>=2 { ev.eval(args[1].clone()) } else { Value::Integer(10) };
    if is_frame_handle(&subj) {
        return ev.eval(Value::expr(Value::symbol("FrameTail"), vec![subj, n]));
    }
    #[cfg(feature = "dataset")]
    { return ev.eval(Value::expr(Value::symbol("__DatasetTail"), vec![subj, n])); }
    #[cfg(not(feature = "dataset"))]
    { return Value::expr(Value::symbol("Tail"), vec![subj, n]); }
}

fn offset_dispatch(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()!=2 { return Value::expr(Value::symbol("Offset"), args); }
    let subj = ev.eval(args[0].clone());
    let n = ev.eval(args[1].clone());
    if is_frame_handle(&subj) {
        return ev.eval(Value::expr(Value::symbol("FrameOffset"), vec![subj, n]));
    }
    #[cfg(feature = "dataset")]
    { return ev.eval(Value::expr(Value::symbol("__DatasetOffset"), vec![subj, n])); }
    #[cfg(not(feature = "dataset"))]
    { return Value::expr(Value::symbol("Offset"), vec![subj, n]); }
}

fn sort_dispatch(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() { return Value::expr(Value::symbol("Sort"), args); }
    let subj = ev.eval(args[0].clone());
    if is_frame_handle(&subj) {
        if args.len()==1 { return ev.eval(Value::expr(Value::symbol("FrameSort"), vec![subj, Value::List(vec![])])); }
        let spec = ev.eval(args[1].clone());
        return ev.eval(Value::expr(Value::symbol("FrameSort"), vec![subj, spec]));
    }
    #[cfg(feature = "dataset")]
    {
        let mut v = vec![subj];
        if args.len()>=2 { v.push(ev.eval(args[1].clone())); }
        return ev.eval(Value::expr(Value::symbol("__DatasetSort"), v));
    }
    #[cfg(not(feature = "dataset"))]
    { return Value::expr(Value::symbol("Sort"), args.into_iter().map(|a| ev.eval(a)).collect()); }
}

fn sortby_dispatch(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // Expect SortBy[f, subject]
    if args.len() != 2 { return Value::expr(Value::symbol("SortBy"), args); }
    let f = args[0].clone();
    let subj = ev.eval(args[1].clone());
    match subj {
        Value::Assoc(_) => Value::expr(Value::symbol("__AssocSortBy"), vec![f, subj]),
        Value::List(_) => Value::expr(Value::symbol("__ListSortBy"), vec![f, subj]),
        other => Value::expr(Value::symbol("SortBy"), vec![f, other]),
    }
}

fn distinct_dispatch(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() { return Value::expr(Value::symbol("Distinct"), args); }
    let subj = ev.eval(args[0].clone());
    if is_frame_handle(&subj) {
        let mut v = vec![subj];
        if args.len()>=2 { v.push(ev.eval(args[1].clone())); }
        return ev.eval(Value::expr(Value::symbol("FrameDistinct"), v));
    }
    #[cfg(feature = "dataset")]
    {
        let mut v = vec![subj];
        if args.len()>=2 { v.push(ev.eval(args[1].clone())); }
        return ev.eval(Value::expr(Value::symbol("__DatasetDistinct"), v));
    }
    #[cfg(not(feature = "dataset"))]
    { return Value::expr(Value::symbol("Distinct"), args.into_iter().map(|a| ev.eval(a)).collect()); }
}

fn str_of(v: &Value) -> Option<String> {
    match v { Value::String(s) | Value::Symbol(s) => Some(s.clone()), _ => None }
}

fn ext_lower(path: &str) -> Option<String> {
    std::path::Path::new(path).extension().map(|e| e.to_string_lossy().to_string().to_ascii_lowercase())
}

fn opt_lookup<'a>(opts: &'a HashMap<String, Value>, k: &str) -> Option<&'a Value> {
    opts.get(k).or_else(|| opts.get(&k.to_string().to_ascii_lowercase().chars().next().map(|_| k.to_string()).unwrap_or_else(|| k.to_string())))
}

fn apply_tabular_post_ops(ev: &mut Evaluator, handle: Value, opts: &HashMap<String, Value>, is_frame: bool) -> Value {
    let mut cur = handle;
    // Columns -> Select
    if let Some(Value::List(cols)) = opts.get("Columns").or_else(|| opts.get("columns")) {
        cur = ev.eval(Value::expr(Value::symbol("Select"), vec![cur, Value::List(cols.clone())]));
    }
    // WithColumns -> for Dataset we have WithColumns; for Frame emulate via FrameSelect including identity columns
    if let Some(Value::Assoc(defs)) = opts.get("WithColumns").or_else(|| opts.get("withColumns")).or_else(|| opts.get("withcolumns")) {
        if is_frame {
            // Identity mapping for existing columns + defs
            let cols_v = ev.eval(Value::expr(Value::symbol("FrameColumns"), vec![cur.clone()]));
            let mut mapping: HashMap<String, Value> = HashMap::new();
            if let Value::List(cs) = cols_v {
                for c in cs {
                    if let Value::String(name) | Value::Symbol(name) = c {
                        mapping.insert(name.clone(), Value::pure_function(Some(vec!["row".into()]), Value::expr(Value::symbol("Part"), vec![Value::symbol("row"), Value::String(name)])));
                    }
                }
            }
            for (k, v) in defs.iter() { mapping.insert(k.clone(), v.clone()); }
            cur = ev.eval(Value::expr(Value::symbol("FrameSelect"), vec![cur, Value::Assoc(mapping)]));
        } else {
            cur = ev.eval(Value::expr(Value::symbol("WithColumns"), vec![cur, Value::Assoc(defs.clone())]));
        }
    }
    // Filter -> Filter
    if let Some(pred) = opts.get("Filter").or_else(|| opts.get("filter")) { cur = ev.eval(Value::expr(Value::symbol("Filter"), vec![cur, pred.clone()])); }
    // Limit -> Head
    if let Some(Value::Integer(n)) = opts.get("Limit").or_else(|| opts.get("limit")) { cur = ev.eval(Value::expr(Value::symbol("Head"), vec![cur, Value::Integer(*n)])); }
    cur
}

fn import_dispatch(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() { return Value::expr(Value::symbol("Import"), args); }
    let src = ev.eval(args[0].clone());
    let mut opts = if args.len()>=2 { match ev.eval(args[1].clone()) { Value::Assoc(m)=>m, _=> HashMap::new() } } else { HashMap::new() };
    // Auto-sniff defaults (Type, Delimiter, Header) when not provided
    if !opts.contains_key("Type") || !opts.contains_key("Delimiter") || !opts.contains_key("Header") {
        let sniff = ev.eval(Value::expr(Value::symbol("Sniff"), vec![src.clone()]));
        if let Value::Assoc(sugg) = sniff {
            for (k,v) in sugg.into_iter() {
                if !opts.contains_key(&k) && (k=="Type" || k=="Delimiter" || k=="Header") {
                    opts.insert(k, v);
                }
            }
        }
    }
    // Resolve Type and Target
    let mut ty = opt_lookup(&opts, "Type").and_then(|v| str_of(v)).unwrap_or_else(|| {
        if let Some(p) = str_of(&src) { ext_lower(&p).unwrap_or_default() } else { String::new() }
    });
    if ty == "" { if let Some(p)= str_of(&src) { if let Some(ext)= ext_lower(&p){ ty=ext; } } }
    ty = ty.to_ascii_uppercase();
    let target = opt_lookup(&opts, "Target").and_then(|v| str_of(v)).unwrap_or_else(|| String::from("Frame"));
    let mode = opt_lookup(&opts, "Mode").and_then(|v| str_of(v)).unwrap_or_else(|| String::from("Eager"));
    // Remote URLs: fetch and re-dispatch via ImportString
    if let Some(url) = str_of(&src) {
        let ul = url.to_ascii_lowercase();
        if ul.starts_with("http://") || ul.starts_with("https://") {
            // Fetch as text; best-effort even without net_https (http supported)
            let resp = ev.eval(Value::expr(Value::symbol("HttpGet"), vec![Value::String(url.clone()), Value::Assoc(HashMap::from([(String::from("As"), Value::String(String::from("Text")))]))]));
            if let Value::Assoc(m) = &resp {
                // Try content-type mapping
                if let Some(ctv) = m.get("headers").and_then(|h| if let Value::Assoc(hm) = h { hm.get("Content-Type") } else { None })
                    .or_else(|| m.get("headers").and_then(|h| if let Value::Assoc(hm) = h { hm.get("content-type") } else { None }))
                    .or_else(|| m.get("headersList").and_then(|hl| if let Value::List(vs) = hl { vs.iter().find_map(|v| if let Value::Assoc(am)=v { am.get("name").and_then(|n| match n { Value::String(s)|Value::Symbol(s) if s.eq_ignore_ascii_case("Content-Type") => am.get("value"), _=> None }) } else { None }) } else { None }))
                {
                    if let Some(cts) = str_of(ctv) {
                        let c = cts.to_ascii_lowercase();
                        let mapped = if c.contains("ndjson") || c.contains("jsonl") { "JSONL" } else if c.contains("json") { "JSON" } else if c.contains("tab-separated") { "TSV" } else if c.contains("csv") { "CSV" } else if c.contains("parquet") { "PARQUET" } else { "" };
                        if !mapped.is_empty() { ty = mapped.to_string(); }
                    }
                }
                if let Some(body) = m.get("body").cloned() {
                    // Ensure Type present for ImportString if we inferred it
                    let mut o2 = opts.clone();
                    if !o2.contains_key("Type") && !ty.is_empty() { o2.insert("Type".into(), Value::String(ty.clone())); }
                    return ev.eval(Value::expr(Value::symbol("ImportString"), vec![body, Value::Assoc(o2)]));
                }
            }
            return Value::expr(Value::symbol("Import"), vec![Value::String(url), Value::Assoc(opts)]);
        }
    }

    // Tabular: CSV/TSV/JSONL/NDJSON/JSON
    if let Some(path) = str_of(&src) {
        // Glob expansion
        let glob_flag = matches!(opt_lookup(&opts, "Glob"), Some(Value::Boolean(true)));
        let is_wild = path.contains('*') || path.contains('?');
        let paths: Vec<Value> = if glob_flag || is_wild {
            match ev.eval(Value::expr(Value::symbol("Glob"), vec![Value::String(path.clone())])) { Value::List(vs)=> vs, other=> vec![other] }
        } else { vec![Value::String(path.clone())] };
        match ty.as_str() {
            "CSV" | "TSV" => {
                // For TSV, inject delimiter if not provided
                let mut opts_delim = opts.clone();
                if ty == "TSV" && !opts_delim.contains_key("Delimiter") && !opts_delim.contains_key("delimiter") {
                    opts_delim.insert("Delimiter".into(), Value::String("\t".into()));
                }
                let is_dataset = target.eq_ignore_ascii_case("dataset") || mode.eq_ignore_ascii_case("lazy");
                if is_dataset {
                    let mut dsets: Vec<Value> = Vec::new();
                    for p in paths.iter() {
                        dsets.push(ev.eval(Value::expr(Value::symbol("ReadCSVDataset"), vec![p.clone(), Value::Assoc(opts_delim.clone())])));
                    }
                    let h = if dsets.len()==1 { dsets.remove(0) } else { ev.eval(Value::expr(Value::symbol("Union"), dsets)) };
                    return apply_tabular_post_ops(ev, h, &opts, false);
                } else {
                    let mut frames: Vec<Value> = Vec::new();
                    for p in paths.iter() {
                        let ds = ev.eval(Value::expr(Value::symbol("ReadCSVDataset"), vec![p.clone(), Value::Assoc(opts_delim.clone())]));
                        let rows = ev.eval(Value::expr(Value::symbol("Collect"), vec![ds]));
                        frames.push(ev.eval(Value::expr(Value::symbol("FrameFromRows"), vec![rows])));
                    }
                    let h = if frames.len()==1 { frames.remove(0) } else { ev.eval(Value::expr(Value::symbol("FrameUnion"), frames)) };
                    return apply_tabular_post_ops(ev, h, &opts, true);
                }
            }
            "JSONL" | "NDJSON" => {
                let is_dataset = target.eq_ignore_ascii_case("dataset") || mode.eq_ignore_ascii_case("lazy");
                if is_dataset {
                    let mut dsets: Vec<Value> = Vec::new();
                    for p in paths.iter() {
                        dsets.push(ev.eval(Value::expr(Value::symbol("ReadJsonLinesDataset"), vec![p.clone(), Value::Assoc(opts.clone())])));
                    }
                    let h = if dsets.len()==1 { dsets.remove(0) } else { ev.eval(Value::expr(Value::symbol("Union"), dsets)) };
                    return apply_tabular_post_ops(ev, h, &opts, false);
                } else {
                    let mut frames: Vec<Value> = Vec::new();
                    for p in paths.iter() {
                        let ds = ev.eval(Value::expr(Value::symbol("ReadJsonLinesDataset"), vec![p.clone(), Value::Assoc(opts.clone())]));
                        let rows = ev.eval(Value::expr(Value::symbol("Collect"), vec![ds]));
                        frames.push(ev.eval(Value::expr(Value::symbol("FrameFromRows"), vec![rows])));
                    }
                    let h = if frames.len()==1 { frames.remove(0) } else { ev.eval(Value::expr(Value::symbol("FrameUnion"), frames)) };
                    return apply_tabular_post_ops(ev, h, &opts, true);
                }
            }
            "JSON" => {
                // Combine multiple JSON files when they are arrays-of-objects
                let mut rows_lists: Vec<Value> = Vec::new();
                for p in paths.iter() {
                    let s = ev.eval(Value::expr(Value::symbol("ReadFile"), vec![p.clone()]));
                    let v = ev.eval(Value::expr(Value::symbol("FromJson"), vec![s]));
                    if let Value::List(rows) = &v { if rows.iter().all(|r| matches!(r, Value::Assoc(_))) { rows_lists.push(v); } else { return v; } }
                }
                if rows_lists.is_empty() { return Value::expr(Value::symbol("Import"), vec![Value::String(path), Value::Assoc(opts)]); }
                // Merge all rows
                let mut all_rows: Vec<Value> = Vec::new();
                for rv in rows_lists.into_iter() { if let Value::List(mut xs) = rv { all_rows.append(&mut xs); } }
                let rows_v = Value::List(all_rows);
                let is_dataset = target.eq_ignore_ascii_case("dataset") || mode.eq_ignore_ascii_case("lazy");
                let out = if is_dataset { ev.eval(Value::expr(Value::symbol("DatasetFromRows"), vec![rows_v])) } else { ev.eval(Value::expr(Value::symbol("FrameFromRows"), vec![rows_v])) };
                return apply_tabular_post_ops(ev, out, &opts, !is_dataset);
            }
            "PARQUET" => {
                // Read Parquet via DuckDB if available
                #[cfg(feature = "db_duckdb")]
                {
                    let is_dataset = target.eq_ignore_ascii_case("dataset") || mode.eq_ignore_ascii_case("lazy");
                    let mut dsets: Vec<Value> = Vec::new();
                    for p in paths.iter() {
                        let conn = ev.eval(Value::expr(Value::symbol("Connect"), vec![Value::String("duckdb::memory:".into())]));
                        let sql = Value::String("SELECT * FROM read_parquet($path)".into());
                        let params = Value::Assoc(HashMap::from([(String::from("path"), p.clone())]));
                        let ds = ev.eval(Value::expr(Value::symbol("SQL"), vec![conn, sql, params]));
                        dsets.push(ds);
                    }
                    let ds_union = if dsets.len()==1 { dsets.remove(0) } else { ev.eval(Value::expr(Value::symbol("Union"), dsets)) };
                    if is_dataset {
                        return apply_tabular_post_ops(ev, ds_union, &opts, false);
                    } else {
                        let rows = ev.eval(Value::expr(Value::symbol("Collect"), vec![ds_union]));
                        let f = ev.eval(Value::expr(Value::symbol("FrameFromRows"), vec![rows]));
                        return apply_tabular_post_ops(ev, f, &opts, true);
                    }
                }
                #[cfg(not(feature = "db_duckdb"))]
                {
                    return Value::Assoc(HashMap::from([
                        ("message".into(), Value::String("Parquet import requires db_duckdb feature".into())),
                        ("tag".into(), Value::String("Import::parquet_unsupported".into())),
                    ]));
                }
            }
            _ => {}
        }
    }
    // Fallback: leave unevaluated for now
    Value::expr(Value::symbol("Import"), vec![src, Value::Assoc(opts)])
}

fn import_string_dispatch(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() { return Value::expr(Value::symbol("ImportString"), args); }
    let content = ev.eval(args[0].clone());
    let mut opts = if args.len()>=2 { match ev.eval(args[1].clone()) { Value::Assoc(m)=>m, _=> HashMap::new() } } else { HashMap::new() };
    // Auto-sniff when Type missing
    if !opts.contains_key("Type") {
        let sniff = ev.eval(Value::expr(Value::symbol("Sniff"), vec![content.clone()]));
        if let Value::Assoc(sugg) = sniff { for (k,v) in sugg.into_iter() { if !opts.contains_key(&k) && (k=="Type" || k=="Delimiter" || k=="Header") { opts.insert(k, v); } } }
    }
    let ty = opt_lookup(&opts, "Type").and_then(|v| str_of(v)).unwrap_or_else(|| String::from("Text")).to_ascii_uppercase();
    let target = opt_lookup(&opts, "Target").and_then(|v| str_of(v)).unwrap_or_else(|| String::from("Frame"));
    match (ty.as_str(), content) {
        ("CSV", v) => {
            let rows = ev.eval(Value::expr(Value::symbol("ParseCSV"), vec![v, Value::Assoc(opts.clone())]));
            let is_dataset = target.eq_ignore_ascii_case("dataset");
            let handle = if is_dataset { ev.eval(Value::expr(Value::symbol("DatasetFromRows"), vec![rows])) } else { ev.eval(Value::expr(Value::symbol("FrameFromRows"), vec![rows])) };
            return apply_tabular_post_ops(ev, handle, &opts, !is_dataset);
        }
        ("JSONL" | "NDJSON", Value::String(s)) => {
            let lines: Vec<Value> = s.lines().map(|x| Value::String(x.to_string())).collect();
            let mut rows: Vec<Value> = Vec::new();
            for l in lines { let j = ev.eval(Value::expr(Value::symbol("FromJson"), vec![l])); if matches!(j, Value::Assoc(_)) { rows.push(j); } }
            let rows_v = Value::List(rows);
            let is_dataset = target.eq_ignore_ascii_case("dataset");
            let handle = if is_dataset { ev.eval(Value::expr(Value::symbol("DatasetFromRows"), vec![rows_v])) } else { ev.eval(Value::expr(Value::symbol("FrameFromRows"), vec![rows_v])) };
            return apply_tabular_post_ops(ev, handle, &opts, !is_dataset);
        }
        ("JSON", v) => {
            let j = ev.eval(Value::expr(Value::symbol("FromJson"), vec![v]));
            if let Value::List(rows) = &j { if rows.iter().all(|r| matches!(r, Value::Assoc(_))) {
                let is_dataset = target.eq_ignore_ascii_case("dataset");
                let handle = if is_dataset { ev.eval(Value::expr(Value::symbol("DatasetFromRows"), vec![j.clone()])) } else { ev.eval(Value::expr(Value::symbol("FrameFromRows"), vec![j.clone()])) };
                return apply_tabular_post_ops(ev, handle, &opts, !is_dataset);
            }}
            return j;
        }
        ("TEXT", v) => v,
        _ => Value::expr(Value::symbol("ImportString"), vec![Value::String("".into()), Value::Assoc(opts)]),
    }
}

fn import_bytes_dispatch(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() { return Value::expr(Value::symbol("ImportBytes"), args); }
    let bytes_v = ev.eval(args[0].clone());
    let opts = if args.len()>=2 { match ev.eval(args[1].clone()) { Value::Assoc(m)=>m, _=> HashMap::new() } } else { HashMap::new() };
    let ty = opt_lookup(&opts, "Type").and_then(|v| str_of(v)).unwrap_or_else(|| String::from("Binary")).to_ascii_uppercase();
    match (ty.as_str(), bytes_v) {
        ("TEXT", v) => ev.eval(Value::expr(Value::symbol("TextDecode"), vec![v])),
        ("JSON", v) => { let s = ev.eval(Value::expr(Value::symbol("TextDecode"), vec![v])); ev.eval(Value::expr(Value::symbol("FromJson"), vec![s])) }
        _ => Value::expr(Value::symbol("ImportBytes"), vec![Value::Symbol("<bytes>".into()), Value::Assoc(opts)]),
    }
}

fn export_dispatch(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() < 2 { return Value::expr(Value::symbol("Export"), args); }
    let value = ev.eval(args[0].clone());
    let dest = ev.eval(args[1].clone());
    let opts = if args.len()>=3 { match ev.eval(args[2].clone()) { Value::Assoc(m)=>m, _=> HashMap::new() } } else { HashMap::new() };
    let path = match dest { Value::String(s)|Value::Symbol(s)=> s, _=> return Value::expr(Value::symbol("Export"), vec![value, dest, Value::Assoc(opts)]) };
    // Decide by extension if not overridden
    let ty = opt_lookup(&opts, "Type").and_then(|v| str_of(v)).unwrap_or_else(|| ext_lower(&path).unwrap_or_else(|| String::from("json"))).to_ascii_uppercase();
    match ty.as_str() {
        "CSV" | "TSV" => {
            // Normalize to rows
            let rows = if is_frame_handle(&value) {
                ev.eval(Value::expr(Value::symbol("FrameCollect"), vec![value.clone()]))
            } else if is_dataset_handle(&value) {
                ev.eval(Value::expr(Value::symbol("Collect"), vec![value.clone()]))
            } else {
                value.clone()
            };
            // Inject delimiter for TSV when not provided
            let mut o2 = opts.clone();
            if ty == "TSV" && !o2.contains_key("Delimiter") && !o2.contains_key("delimiter") { o2.insert("Delimiter".into(), Value::String("\t".into())); }
            return ev.eval(Value::expr(Value::symbol("WriteCSV"), vec![Value::String(path), rows, Value::Assoc(o2)]));
        }
        "JSON" => {
            let s = ev.eval(Value::expr(Value::symbol("ToJson"), vec![value.clone(), Value::Assoc(HashMap::new())]));
            return ev.eval(Value::expr(Value::symbol("WriteFile"), vec![Value::String(path), s]));
        }
        "JSONL" | "NDJSON" => {
            // Expect list-of-assoc rows; if Frame/Dataset, collect rows
            let rows = if is_frame_handle(&value) { ev.eval(Value::expr(Value::symbol("FrameCollect"), vec![value.clone()])) } else if is_dataset_handle(&value) { ev.eval(Value::expr(Value::symbol("Collect"), vec![value.clone()])) } else { value.clone() };
            // Render lines via ToJson and write
            let mut lines: Vec<Value> = Vec::new();
            if let Value::List(rs) = rows { for r in rs { lines.push(ev.eval(Value::expr(Value::symbol("ToJson"), vec![r, Value::Assoc(HashMap::new())]))); } }
            let text = Value::String(lines.into_iter().map(|v| match v { Value::String(s)=>s, other=> lyra_core::pretty::format_value(&other)}).collect::<Vec<_>>().join("\n"));
            return ev.eval(Value::expr(Value::symbol("WriteFile"), vec![Value::String(path), text]));
        }
        "PARQUET" => {
            #[cfg(feature = "db_duckdb")]
            {
                // Normalize to Dataset
                let ds = if is_dataset_handle(&value) {
                    value.clone()
                } else if is_frame_handle(&value) {
                    let rows = ev.eval(Value::expr(Value::symbol("FrameCollect"), vec![value.clone()]));
                    ev.eval(Value::expr(Value::symbol("DatasetFromRows"), vec![rows]))
                } else if matches!(value, Value::List(_)) {
                    ev.eval(Value::expr(Value::symbol("DatasetFromRows"), vec![value.clone()]))
                } else {
                    return Value::expr(Value::symbol("Export"), vec![value, Value::String(path), Value::Assoc(opts)]);
                };
                // Infer columns from first row
                let rows = ev.eval(Value::expr(Value::symbol("Collect"), vec![ds.clone()]));
                let cols: Vec<String> = if let Value::List(vs) = &rows { if let Some(Value::Assoc(m)) = vs.get(0) { m.keys().cloned().collect() } else { vec![] } } else { vec![] };
                if cols.is_empty() { return Value::Boolean(false); }
                let conn = ev.eval(Value::expr(Value::symbol("Connect"), vec![Value::String("duckdb::memory:".into())]));
                // Create table with VARCHAR columns
                let col_defs = cols.iter().map(|c| format!("{} VARCHAR", c)).collect::<Vec<_>>().join(", ");
                let _ = ev.eval(Value::expr(Value::symbol("Exec"), vec![conn.clone(), Value::String(format!("CREATE TABLE t ({});", col_defs))]));
                let _ = ev.eval(Value::expr(Value::symbol("WriteDataset"), vec![ds, conn.clone(), Value::String("t".into())]));
                let _ = ev.eval(Value::expr(Value::symbol("Exec"), vec![conn, Value::String(format!("COPY t TO '{}' (FORMAT PARQUET);", path.replace("'", "''")))]));
                return Value::Boolean(true);
            }
            #[cfg(not(feature = "db_duckdb"))]
            {
                return Value::Assoc(HashMap::from([
                    ("message".into(), Value::String("Parquet export requires db_duckdb feature".into())),
                    ("tag".into(), Value::String("Export::parquet_unsupported".into())),
                ]));
            }
        }
        _ => {}
    }
    Value::expr(Value::symbol("Export"), vec![value, Value::String(path), Value::Assoc(opts)])
}

fn sniff_dispatch(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()!=1 { return Value::expr(Value::symbol("Sniff"), args); }
    let src = ev.eval(args[0].clone());
    // Helper: sniff content string
    fn sniff_string_content(s: &str) -> HashMap<String, Value> {
        let mut out = HashMap::new();
        let trimmed = s.trim_start();
        if trimmed.starts_with('[') || trimmed.starts_with('{') {
            // Detect JSONL vs JSON array
            let is_jsonl = s.lines().take(5).filter(|l| !l.trim().is_empty()).all(|l| {
                let lt = l.trim_start(); lt.starts_with('{') || lt.starts_with('[')
            }) && !trimmed.starts_with('[');
            if is_jsonl { out.insert("Type".into(), Value::String("JSONL".into())); }
            else { out.insert("Type".into(), Value::String("JSON".into())); }
            return out;
        }
        // Delimiter heuristics
        let mut lines = s.lines();
        let first = lines.find(|l| !l.trim().is_empty()).unwrap_or("");
        if first.contains('\t') { out.insert("Type".into(), Value::String("TSV".into())); out.insert("Delimiter".into(), Value::String("\t".into())); }
        else if first.contains(',') { out.insert("Type".into(), Value::String("CSV".into())); out.insert("Delimiter".into(), Value::String(",".into())); }
        else if first.contains(';') { out.insert("Type".into(), Value::String("CSV".into())); out.insert("Delimiter".into(), Value::String(";".into())); }
        if let Some(Value::String(t)) = out.get("Type") {
            if t == "CSV" || t == "TSV" {
                let delim = if let Some(Value::String(d)) = out.get("Delimiter") { d.clone() } else { String::from(",") };
                let header = first.split(delim.as_str()).all(|p| p.chars().all(|c| c.is_alphanumeric() || c=='_' ));
                out.insert("Header".into(), Value::Boolean(header));
            }
        }
        out
    }
    // Helper: sniff bytes for Parquet magic or UTF-8 text
    fn sniff_bytes_content(bytes: &[u8]) -> HashMap<String, Value> {
        let mut out = HashMap::new();
        if bytes.len() >= 4 && &bytes[0..4] == b"PAR1" { out.insert("Type".into(), Value::String("PARQUET".into())); return out; }
        // Try interpret as UTF-8 and delegate to string
        if let Ok(s) = std::str::from_utf8(bytes) { return sniff_string_content(s); }
        out
    }
    // String or Symbol: path or URL or inline content
    if let Some(s) = str_of(&src) {
        let mut out: HashMap<String, Value> = HashMap::new();
        // URL path
        let sl = s.to_ascii_lowercase();
        if sl.starts_with("http://") || sl.starts_with("https://") {
            // Prefer HEAD
            let head = ev.eval(Value::expr(Value::symbol("HttpHead"), vec![Value::String(s.clone()), Value::Assoc(HashMap::new())]));
            if let Value::Assoc(m) = &head {
                if let Some(ctv) = m.get("headers").and_then(|h| if let Value::Assoc(hm) = h { hm.get("Content-Type") } else { None })
                    .or_else(|| m.get("headers").and_then(|h| if let Value::Assoc(hm) = h { hm.get("content-type") } else { None }))
                {
                    if let Some(cts) = str_of(ctv) {
                        let c = cts.to_ascii_lowercase();
                        if c.contains("ndjson") || c.contains("jsonl") { out.insert("Type".into(), Value::String("JSONL".into())); }
                        else if c.contains("json") { out.insert("Type".into(), Value::String("JSON".into())); }
                        else if c.contains("tab-separated") { out.insert("Type".into(), Value::String("TSV".into())); out.insert("Delimiter".into(), Value::String("\t".into())); }
                        else if c.contains("csv") { out.insert("Type".into(), Value::String("CSV".into())); out.insert("Delimiter".into(), Value::String(",".into())); }
                        else if c.contains("parquet") { out.insert("Type".into(), Value::String("PARQUET".into())); }
                    }
                }
            }
            if out.is_empty() {
                // Fetch small sample as bytes
                let resp = ev.eval(Value::expr(Value::symbol("HttpGet"), vec![Value::String(s), Value::Assoc(HashMap::from([(String::from("As"), Value::String(String::from("Bytes")))]))]));
                if let Value::Assoc(m) = resp {
                    if let Some(Value::List(bs)) = m.get("bytes") {
                        let bytes: Vec<u8> = bs.iter().filter_map(|v| if let Value::Integer(n)=v { Some(*n as u8) } else { None }).collect();
                        out = sniff_bytes_content(&bytes);
                    }
                }
            }
            return Value::Assoc(out);
        }
        // File path vs inline content: prefer file if exists
        let exists = ev.eval(Value::expr(Value::symbol("FileExistsQ"), vec![Value::String(s.clone())]));
        if matches!(exists, Value::Boolean(true)) {
            if let Some(ext) = ext_lower(&s) {
                match ext.as_str() {
                    "csv" => { out.insert("Type".into(), Value::String("CSV".into())); },
                    "tsv" => { out.insert("Type".into(), Value::String("TSV".into())); out.insert("Delimiter".into(), Value::String("\t".into())); },
                    "jsonl" | "ndjson" => { out.insert("Type".into(), Value::String("JSONL".into())); },
                    "json" => { out.insert("Type".into(), Value::String("JSON".into())); },
                    "parquet" | "pq" => { out.insert("Type".into(), Value::String("PARQUET".into())); },
                    _ => {}
                }
            }
            if !out.contains_key("Type") || matches!(out.get("Type"), Some(Value::String(t)) if t=="CSV" || t=="TSV") {
                // Heuristics: first lines
                let sample_v = ev.eval(Value::expr(Value::symbol("ReadLines"), vec![Value::String(s)]));
                if let Value::List(lines) = sample_v {
                    let mut content = String::new();
                    for l in lines.iter().take(5) { if let Value::String(ss) = l { content.push_str(ss); content.push('\n'); } }
                    let m = sniff_string_content(&content);
                    for (k,v) in m { out.insert(k, v); }
                }
            }
            return Value::Assoc(out);
        } else {
            // Inline content
            return Value::Assoc(sniff_string_content(&s));
        }
    }
    // Bytes-like: List of integers
    if let Value::List(vs) = &src {
        if vs.iter().all(|v| matches!(v, Value::Integer(_))) {
            let bytes: Vec<u8> = vs.iter().filter_map(|v| if let Value::Integer(n)=v { Some(*n as u8) } else { None }).collect();
            return Value::Assoc(sniff_bytes_content(&bytes));
        }
    }
    Value::expr(Value::symbol("Sniff"), vec![src])
}

fn insert_dispatch(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() < 2 { return Value::expr(Value::symbol("Insert"), args); }
    let target = ev.eval(args[0].clone());
    let value = ev.eval(args[1].clone());
    if is_graph_handle(&target) {
        // Insert into Graph: infer edges vs nodes by assoc keys
        let is_edge_assoc = |v: &Value| -> bool {
            if let Value::Assoc(m) = v {
                m.contains_key("src") || m.contains_key("Src") || m.contains_key("dst") || m.contains_key("Dst")
            } else { false }
        };
        match &value {
            Value::Assoc(_) if is_edge_assoc(&value) => {
                return ev.eval(Value::expr(Value::symbol("AddEdges"), vec![target, value]));
            }
            Value::List(xs) if xs.iter().any(|x| is_edge_assoc(x)) => {
                return ev.eval(Value::expr(Value::symbol("AddEdges"), vec![target, value]));
            }
            _ => {
                return ev.eval(Value::expr(Value::symbol("AddNodes"), vec![target, value]));
            }
        }
    }
    if is_set_handle(&target) {
        return ev.eval(Value::expr(Value::symbol("SetInsert"), vec![target, value]));
    }
    if is_pq_handle(&target) {
        return ev.eval(Value::expr(Value::symbol("PQInsert"), vec![target, value]));
    }
    if is_bag_handle(&target) {
        return ev.eval(Value::expr(Value::symbol("BagAdd"), vec![target, value]));
    }
    if is_queue_handle(&target) {
        return ev.eval(Value::expr(Value::symbol("Enqueue"), vec![target, value]));
    }
    if is_stack_handle(&target) {
        return ev.eval(Value::expr(Value::symbol("Push"), vec![target, value]));
    }
    // Fallback: leave unevaluated
    Value::expr(Value::symbol("Insert"), vec![target, value])
}

fn remove_dispatch(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() { return Value::expr(Value::symbol("Remove"), args); }
    let target = ev.eval(args[0].clone());
    if is_graph_handle(&target) {
        let value = if args.len() >= 2 { Some(ev.eval(args[1].clone())) } else { None };
        let is_edge_assoc = |v: &Value| -> bool {
            if let Value::Assoc(m) = v {
                m.contains_key("src") || m.contains_key("Src") || m.contains_key("dst") || m.contains_key("Dst")
            } else { false }
        };
        if let Some(val) = value {
            let name = match &val {
                Value::Assoc(_) if is_edge_assoc(&val) => "RemoveEdges",
                Value::List(xs) if xs.iter().any(|x| is_edge_assoc(x)) => "RemoveEdges",
                _ => "RemoveNodes",
            };
            return ev.eval(Value::expr(Value::symbol(name), vec![target, val]));
        }
        return Value::expr(Value::symbol("Remove"), vec![target]);
    }
    if is_set_handle(&target) {
        if args.len() < 2 { return Value::expr(Value::symbol("Remove"), vec![target]); }
        let value = ev.eval(args[1].clone());
        return ev.eval(Value::expr(Value::symbol("SetRemove"), vec![target, value]));
    }
    if is_bag_handle(&target) {
        if args.len() < 2 { return Value::expr(Value::symbol("Remove"), vec![target]); }
        let value = ev.eval(args[1].clone());
        return ev.eval(Value::expr(Value::symbol("BagRemove"), vec![target, value]));
    }
    if is_queue_handle(&target) {
        return ev.eval(Value::expr(Value::symbol("Dequeue"), vec![target]));
    }
    if is_stack_handle(&target) {
        return ev.eval(Value::expr(Value::symbol("Pop"), vec![target]));
    }
    if is_pq_handle(&target) {
        return ev.eval(Value::expr(Value::symbol("PQPop"), vec![target]));
    }
    // Fallback to filesystem removal via explicit alias to avoid shadowing recursion
    match target {
        Value::String(_) | Value::Symbol(_) => ev.eval(Value::expr(Value::symbol("PathRemove"), vec![target])),
        _ => Value::expr(Value::symbol("Remove"), vec![target]),
    }
}

fn add_dispatch(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() < 2 { return Value::expr(Value::symbol("Add"), args); }
    let target = ev.eval(args[0].clone());
    let value = ev.eval(args[1].clone());
    if is_bag_handle(&target) {
        return ev.eval(Value::expr(Value::symbol("BagAdd"), vec![target, value]));
    }
    if is_set_handle(&target) {
        return ev.eval(Value::expr(Value::symbol("SetInsert"), vec![target, value]));
    }
    if is_queue_handle(&target) {
        return ev.eval(Value::expr(Value::symbol("Enqueue"), vec![target, value]));
    }
    if is_stack_handle(&target) {
        return ev.eval(Value::expr(Value::symbol("Push"), vec![target, value]));
    }
    if is_pq_handle(&target) {
        return ev.eval(Value::expr(Value::symbol("PQInsert"), vec![target, value]));
    }
    if let Some(path) = looks_like_index_assoc(&target) {
        return ev.eval(Value::expr(Value::symbol("IndexAdd"), vec![Value::String(path), value]));
    }
    // Fallback: leave unevaluated
    Value::expr(Value::symbol("Add"), vec![target, value])
}

fn info_dispatch(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 1 { return Value::expr(Value::symbol("Info"), args); }
    let target = ev.eval(args[0].clone());
    if is_graph_handle(&target) {
        return ev.eval(Value::expr(Value::symbol("GraphInfo"), vec![target]));
    }
    if is_frame_handle(&target) {
        #[cfg(feature = "frame")]
        {
            return crate::frame::frame_info(ev, vec![target]);
        }
        #[cfg(not(feature = "frame"))]
        {
            return Value::expr(Value::symbol("Info"), vec![target]);
        }
    }
    if is_dataset_handle(&target) {
        #[cfg(feature = "dataset")]
        {
            let rows = crate::dataset::count_ds(ev, vec![target.clone()]);
            let cols = ev.eval(Value::expr(Value::symbol("Columns"), vec![target.clone()]));
            let mut m = HashMap::new();
            m.insert("Type".into(), Value::String("Dataset".into()));
            m.insert("Rows".into(), rows);
            m.insert("Columns".into(), cols);
            return Value::Assoc(m);
        }
        #[cfg(not(feature = "dataset"))]
        {
            return Value::expr(Value::symbol("Info"), vec![target]);
        }
    }
    if let Some(path) = looks_like_index_assoc(&target) {
        return ev.eval(Value::expr(Value::symbol("IndexInfo"), vec![Value::String(path)]));
    }
    if is_set_handle(&target) {
        let n = ev.eval(Value::expr(Value::symbol("Length"), vec![target.clone()]));
        let mut m = HashMap::new(); m.insert("Type".into(), Value::String("Set".into())); m.insert("Size".into(), n); return Value::Assoc(m);
    }
    if is_bag_handle(&target) {
        let n = ev.eval(Value::expr(Value::symbol("Count"), vec![target.clone()]));
        let mut m = HashMap::new(); m.insert("Type".into(), Value::String("Bag".into())); m.insert("Size".into(), n); return Value::Assoc(m);
    }
    if is_queue_handle(&target) {
        let n = ev.eval(Value::expr(Value::symbol("Length"), vec![target.clone()]));
        let mut m = HashMap::new(); m.insert("Type".into(), Value::String("Queue".into())); m.insert("Size".into(), n); return Value::Assoc(m);
    }
    if is_stack_handle(&target) {
        let n = ev.eval(Value::expr(Value::symbol("Length"), vec![target.clone()]));
        let mut m = HashMap::new(); m.insert("Type".into(), Value::String("Stack".into())); m.insert("Size".into(), n); return Value::Assoc(m);
    }
    if is_pq_handle(&target) {
        let n = ev.eval(Value::expr(Value::symbol("Length"), vec![target.clone()]));
        let mut m = HashMap::new(); m.insert("Type".into(), Value::String("PriorityQueue".into())); m.insert("Size".into(), n); return Value::Assoc(m);
    }
    if is_vector_store_handle(&target) {
        let count = ev.eval(Value::expr(Value::symbol("VectorCount"), vec![target.clone()]));
        let mut m: HashMap<String, Value> = HashMap::new();
        if let Value::Assoc(h) = &target {
            if let Some(Value::String(n)) = h.get("Name") { m.insert("Name".into(), Value::String(n.clone())); }
        }
        m.insert("Count".into(), count);
        m.insert("Type".into(), Value::String("VectorStore".into()));
        return Value::Assoc(m);
    }
    if is_conn_handle(&target) {
        return ev.eval(Value::expr(Value::symbol("ConnectionInfo"), vec![target]));
    }
    if is_process_handle(&target) {
        return ev.eval(Value::expr(Value::symbol("ProcessInfo"), vec![target]));
    }
    if is_cursor_handle(&target) {
        return ev.eval(Value::expr(Value::symbol("CursorInfo"), vec![target]));
    }
    if is_channel_handle(&target) {
        // No ChannelInfo; leave unevaluated for now
        return Value::expr(Value::symbol("Info"), vec![target]);
    }
    // Default: leave unevaluated
    Value::expr(Value::symbol("Info"), vec![target])
}

fn length_dispatch(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 1 { return Value::expr(Value::symbol("Length"), args); }
    let x = ev.eval(args[0].clone());
    if is_frame_handle(&x) {
        #[cfg(feature = "frame")]
        {
            return crate::frame::frame_count(ev, vec![x]);
        }
        #[cfg(not(feature = "frame"))]
        {
            return Value::expr(Value::symbol("Length"), vec![x]);
        }
    }
    if is_dataset_handle(&x) {
        #[cfg(feature = "dataset")]
        {
            return crate::dataset::count_ds(ev, vec![x]);
        }
        #[cfg(not(feature = "dataset"))]
        {
            return Value::expr(Value::symbol("Length"), vec![x]);
        }
    }
    if is_set_handle(&x) {
        let lst = ev.eval(Value::expr(Value::symbol("SetToList"), vec![x]));
        return match lst { Value::List(v) => Value::Integer(v.len() as i64), _ => Value::Integer(0) };
    }
    if is_vector_store_handle(&x) { return ev.eval(Value::expr(Value::symbol("VectorCount"), vec![x])); }
    if is_bag_handle(&x) { return ev.eval(Value::expr(Value::symbol("BagSize"), vec![x])); }
    if is_queue_handle(&x) || is_stack_handle(&x) || is_pq_handle(&x) {
        return ev.eval(Value::expr(Value::symbol("Length"), vec![x]));
    }
    match x {
        Value::List(v) => Value::Integer(v.len() as i64),
        Value::String(s) => Value::Integer(s.chars().count() as i64),
        Value::Assoc(m) => Value::Integer(m.len() as i64),
        _ => Value::Integer(0),
    }
}

fn emptyq_dispatch(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 1 { return Value::expr(Value::symbol("EmptyQ"), args); }
    let x = ev.eval(args[0].clone());
    if is_frame_handle(&x) {
        #[cfg(feature = "frame")]
        {
            let n = crate::frame::frame_count(ev, vec![x]);
            return match n { Value::Integer(k) => Value::Boolean(k == 0), _ => Value::Boolean(false) };
        }
        #[cfg(not(feature = "frame"))]
        {
            return Value::expr(Value::symbol("EmptyQ"), vec![x]);
        }
    }
    if is_dataset_handle(&x) {
        #[cfg(feature = "dataset")]
        {
            let n = crate::dataset::count_ds(ev, vec![x]);
            return match n { Value::Integer(k) => Value::Boolean(k == 0), _ => Value::Boolean(false) };
        }
        #[cfg(not(feature = "dataset"))]
        {
            return Value::expr(Value::symbol("EmptyQ"), vec![x]);
        }
    }
    if is_vector_store_handle(&x) {
        let n = ev.eval(Value::expr(Value::symbol("VectorCount"), vec![x]));
        return match n { Value::Integer(k) => Value::Boolean(k == 0), _ => Value::Boolean(false) };
    }
    if is_set_handle(&x) || is_queue_handle(&x) || is_stack_handle(&x) || is_pq_handle(&x) {
        let n = ev.eval(Value::expr(Value::symbol("Length"), vec![x]));
        return match n { Value::Integer(k) => Value::Boolean(k == 0), _ => Value::Boolean(false) };
    }
    if is_bag_handle(&x) {
        let n = ev.eval(Value::expr(Value::symbol("Count"), vec![x]));
        return match n { Value::Integer(k) => Value::Boolean(k == 0), _ => Value::Boolean(false) };
    }
    match x {
        Value::List(v) => Value::Boolean(v.is_empty()),
        Value::Assoc(m) => Value::Boolean(m.is_empty()),
        Value::String(s) => Value::Boolean(s.is_empty()),
        _ => Value::Boolean(false),
    }
}

fn count_dispatch(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 1 { return Value::expr(Value::symbol("Count"), args); }
    let x = ev.eval(args[0].clone());
    if is_frame_handle(&x) {
        #[cfg(feature = "frame")]
        {
            return crate::frame::frame_count(ev, vec![x]);
        }
        #[cfg(not(feature = "frame"))]
        {
            return Value::expr(Value::symbol("Count"), vec![x]);
        }
    }
    if is_dataset_handle(&x) {
        #[cfg(feature = "dataset")]
        {
            return crate::dataset::count_ds(ev, vec![x]);
        }
        #[cfg(not(feature = "dataset"))]
        {
            return Value::expr(Value::symbol("Count"), vec![x]);
        }
    }
    if is_bag_handle(&x) { return ev.eval(Value::expr(Value::symbol("BagCount"), vec![x])); }
    if is_vector_store_handle(&x) || matches!(x, Value::String(_) | Value::Symbol(_)) {
        // VectorStore accepts a Name string or assoc handle
        return ev.eval(Value::expr(Value::symbol("VectorCount"), vec![x]));
    }
    if is_set_handle(&x) {
        let lst = ev.eval(Value::expr(Value::symbol("SetToList"), vec![x]));
        return match lst { Value::List(v) => Value::Integer(v.len() as i64), _ => Value::Integer(0) };
    }
    if is_queue_handle(&x) { return ev.eval(Value::expr(Value::symbol("QueueSize"), vec![x])); }
    if is_stack_handle(&x) { return ev.eval(Value::expr(Value::symbol("StackSize"), vec![x])); }
    if is_pq_handle(&x) { return ev.eval(Value::expr(Value::symbol("PQSize"), vec![x])); }
    match x {
        Value::List(v) => Value::Integer(v.len() as i64),
        Value::Assoc(m) => Value::Integer(m.len() as i64),
        Value::String(s) => Value::Integer(s.chars().count() as i64),
        _ => Value::Integer(0),
    }
}

fn search_dispatch(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() < 2 { return Value::expr(Value::symbol("Search"), args); }
    let target = ev.eval(args[0].clone());
    let query = ev.eval(args[1].clone());
    let opts = if args.len() >= 3 { Some(ev.eval(args[2].clone())) } else { None };
    if is_vector_store_handle(&target) || matches!(target, Value::String(_) | Value::Symbol(_)) {
        let mut v = vec![target, query];
        if let Some(o) = opts { v.push(o); }
        return ev.eval(Value::expr(Value::symbol("VectorSearch"), v));
    }
    if let Some(path) = looks_like_index_assoc(&target) {
        // IndexSearch[indexPath, q]
        return ev.eval(Value::expr(Value::symbol("IndexSearch"), vec![Value::String(path), query]));
    }
    Value::expr(Value::symbol("Search"), vec![target, query])
}

fn subsetq_dispatch(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 2 { return Value::expr(Value::symbol("SubsetQ"), args); }
    let a = ev.eval(args[0].clone());
    let b = ev.eval(args[1].clone());
    if is_set_handle(&a) && is_set_handle(&b) {
        return ev.eval(Value::expr(Value::symbol("SetSubsetQ"), vec![a, b]));
    }
    if let (Value::List(la), Value::List(lb)) = (&a, &b) {
        use std::collections::HashSet;
        let sb: HashSet<String> = lb.iter().map(|x| lyra_runtime::eval::value_order_key(x)).collect();
        let sub = la.iter().all(|x| sb.contains(&lyra_runtime::eval::value_order_key(x)));
        return Value::Boolean(sub);
    }
    Value::expr(Value::symbol("SubsetQ"), vec![a, b])
}

fn equalq_dispatch(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 2 { return Value::expr(Value::symbol("EqualQ"), args); }
    let a = ev.eval(args[0].clone());
    let b = ev.eval(args[1].clone());
    if is_set_handle(&a) && is_set_handle(&b) {
        return ev.eval(Value::expr(Value::symbol("SetEqualQ"), vec![a, b]));
    }
    Value::expr(Value::symbol("EqualQ"), vec![a, b])
}

fn close_dispatch(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 1 { return Value::expr(Value::symbol("Close"), args); }
    let x = ev.eval(args[0].clone());
    if is_channel_handle(&x) {
        return ev.eval(Value::expr(Value::symbol("CloseChannel"), vec![x]));
    }
    if is_cursor_handle(&x) {
        return ev.eval(Value::expr(Value::symbol("__DBClose"), vec![x]));
    }
    Value::expr(Value::symbol("Close"), vec![x])
}
fn split_dispatch(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [s] | [s, _] => {
            let subj = ev.eval(s.clone());
            if matches!(subj, Value::String(_)) {
                let mut pass = vec![subj];
                if args.len() == 2 { pass.push(args[1].clone()); }
                return ev.eval(Value::Expr { head: Box::new(Value::Symbol("Split".into())), args: pass });
            }
        }
        _ => {}
    }
    Value::Expr { head: Box::new(Value::Symbol("Split".into())), args }
}
