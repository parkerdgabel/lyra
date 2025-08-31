use lyra_runtime::Evaluator;

// Seed concise, human-facing summaries and param hints for common builtins.
// These power REPL `?Symbol`, hover, and completion details via Documentation[].
pub fn register_docs(ev: &mut Evaluator) {
    // Core arithmetic
    ev.set_doc("Plus", "Add numbers; Listable, Flat, Orderless.", &["a", "b", "…"]);
    ev.set_doc("Times", "Multiply numbers; Listable, Flat, Orderless.", &["a", "b", "…"]);
    ev.set_doc("Minus", "Subtract or unary negate.", &["a", "b?"]);
    ev.set_doc("Divide", "Divide two numbers.", &["a", "b"]);
    ev.set_doc("Power", "Exponentiation (right-associative). Tensor-aware: elementwise when any tensor.", &["a", "b"]);
    ev.set_doc_examples(
        "Plus",
        &["Plus[1, 2]  ==> 3", "Plus[1, 2, 3]  ==> 6", "Plus[{1,2,3}]  ==> {1,2,3} (Listable)"],
    );
    ev.set_doc_examples(
        "Times",
        &["Times[2, 3]  ==> 6", "Times[2, 3, 4]  ==> 24", "Times[{2,3}, 10]  ==> {20,30}"],
    );
    ev.set_doc_examples("Minus", &["Minus[5, 2]  ==> 3", "Minus[5]  ==> -5"]);
    ev.set_doc_examples("Divide", &["Divide[6, 3]  ==> 2", "Divide[7, 2]  ==> 3.5"]);
    ev.set_doc_examples("Power", &["Power[2, 8]  ==> 256", "Power[Tensor[{2,3}], 2]  ==> Tensor[...]", "2^3^2 parses as 2^(3^2)"]);

    // Combinatorics
    ev.set_doc("Permutations", "All permutations (or k-permutations) of a list or range 1..n.", &["listOrN", "k?"]);
    ev.set_doc("Combinations", "All k-combinations (subsets) of a list or 1..n.", &["listOrN", "k"]);
    ev.set_doc("PermutationsCount", "Count of permutations; n! or nPk.", &["n", "k?"]);
    ev.set_doc("CombinationsCount", "Count of combinations; Binomial[n,k].", &["n", "k"]);
    ev.set_doc_examples(
        "Permutations",
        &[
            "Permutations[{1,2}]  ==> {{1,2},{2,1}}",
            "Length[Permutations[{1,2,3}]]  ==> 6",
            "Permutations[3, 2]  ==> {{1,2},{1,3},{2,1},{2,3},{3,1},{3,2}}"
        ],
    );
    ev.set_doc_examples(
        "Combinations",
        &[
            "Combinations[{1,2,3}, 2]  ==> {{1,2},{1,3},{2,3}}",
            "Length[Combinations[4, 2]]  ==> 6",
        ],
    );
    ev.set_doc_examples(
        "PermutationsCount",
        &["PermutationsCount[5]  ==> 120", "PermutationsCount[5, 2]  ==> 20"],
    );
    ev.set_doc_examples(
        "CombinationsCount",
        &["CombinationsCount[5, 2]  ==> 10"],
    );

    // Number theory
    ev.set_doc("ExtendedGCD", "Extended GCD: returns {g, x, y} with a x + b y = g.", &["a", "b"]);
    ev.set_doc("ModInverse", "Multiplicative inverse of a modulo m (if coprime).", &["a", "m"]);
    ev.set_doc("ChineseRemainder", "Solve x ≡ r_i (mod m_i) for coprime moduli.", &["residues", "moduli"]);
    ev.set_doc("DividesQ", "Predicate: does a divide b?", &["a", "b"]);
    ev.set_doc("CoprimeQ", "Predicate: are integers coprime?", &["a", "b"]);
    ev.set_doc("PrimeQ", "Predicate: is integer prime?", &["n"]);
    ev.set_doc("NextPrime", "Next prime greater than n (or 2 if n<2).", &["n"]);
    ev.set_doc("FactorInteger", "Prime factorization as {{p1,e1},{p2,e2},…}.", &["n"]);
    ev.set_doc("PrimeFactors", "List of prime factors with multiplicity.", &["n"]);
    ev.set_doc("EulerPhi", "Euler's totient function: count of 1<=k<=n with CoprimeQ[k,n].", &["n"]);
    ev.set_doc("MobiusMu", "Möbius mu function: 0 if non-square-free, else (-1)^k.", &["n"]);
    ev.set_doc("PowerMod", "Modular exponentiation: a^b mod m (supports negative b if invertible).", &["a","b","m"]);
    ev.set_doc_examples("ExtendedGCD", &["ExtendedGCD[240, 46]  ==> {2, -9, 47}"]);
    ev.set_doc_examples("ModInverse", &["ModInverse[3, 11]  ==> 4"]);
    ev.set_doc_examples(
        "ChineseRemainder",
        &[
            "ChineseRemainder[{2,3,2}, {3,5,7}]  ==> 23",
            "23 mod 3==2, mod 5==3, mod 7==2",
        ],
    );
    ev.set_doc_examples("DividesQ", &["DividesQ[3, 12]  ==> True", "DividesQ[5, 12]  ==> False"]);
    ev.set_doc_examples("CoprimeQ", &["CoprimeQ[12, 35]  ==> True", "CoprimeQ[12, 18]  ==> False"]);
    ev.set_doc_examples("PrimeQ", &["PrimeQ[17]  ==> True", "PrimeQ[1]  ==> False"]);
    ev.set_doc_examples("NextPrime", &["NextPrime[10]  ==> 11", "NextPrime[-5]  ==> 2"]);
    ev.set_doc_examples("FactorInteger", &["FactorInteger[84]  ==> {{2,2},{3,1},{7,1}}"]);
    ev.set_doc_examples("PrimeFactors", &["PrimeFactors[84]  ==> {2,2,3,7}"]);
    ev.set_doc_examples("EulerPhi", &["EulerPhi[36]  ==> 12"]);
    ev.set_doc_examples("MobiusMu", &["MobiusMu[36]  ==> 0", "MobiusMu[35]  ==> 1", "MobiusMu[30]  ==> -1"]);
    ev.set_doc_examples("PowerMod", &["PowerMod[2, 10, 1000]  ==> 24", "PowerMod[3, -1, 11]  ==> 4"]);

    // Logging (resource + generic verbs)
    ev.set_doc("Logger", "Create/configure a logger (global).", &["opts?"]);
    ev.set_doc(
        "Write",
        "Write to a Logger or set key in an association (dispatched). Overloads: Write[logger, msg, opts?]; Write[assoc, key, value]",
        &["logger|assoc", "msg|key", "opts?|value"],
    );
    ev.set_doc("SetLevel", "Set logger level (trace|debug|info|warn|error).", &["logger", "level"]);
    ev.set_doc_examples("Logger", &["logger := Logger[<|\"Level\"->\"debug\"|>]  ==> <|__type->\"Logger\",Name->\"default\"|>"]);
    ev.set_doc_examples("Write", &["Write[Logger[], \"service started\", <|\"Level\"->\"info\", \"Meta\"-><|\"port\"->8080|>|>]  ==> True"]);
    ev.set_doc_examples("WithLogger", &["WithLogger[<|\"requestId\"->\"abc\"|>, Write[Logger[], \"ok\"]]  ==> True"]);
    // Info is documented generically below; avoid duplicate set_doc here.

    // Lists and mapping
    ev.set_doc("Map", "Map a function over a list.", &["f", "list"]);
    ev.set_doc("Thread", "Thread Sequence and lists into arguments.", &["expr"]);
    ev.set_doc_examples("Map", &["Map[ToUpper, {\"a\", \"b\"}]  ==> {\"A\", \"B\"}", "Map[#^2 &, {1,2,3}]  ==> {1,4,9}"]);
    ev.set_doc_examples(
        "Thread",
        &["Thread[f[{1,2}, {3,4}]]  ==> {f[1,3], f[2,4]}", "Thread[Plus[Sequence[{1,2},{3,4}]]]  ==> {4,6}"],
    );

    // Replacement rules
    ev.set_doc("Replace", "Replace first match by rule(s).", &["expr", "rules"]);
    ev.set_doc("ReplaceAll", "Replace all matches by rule(s).", &["expr", "rules"]);
    ev.set_doc("ReplaceFirst", "Replace first element(s) matching pattern.", &["expr", "rule"]);
    ev.set_doc_examples(
        "ReplaceAll",
        &["ReplaceAll[{1,2,1,3}, 1->9]  ==> {9,2,9,3}", "ReplaceAll[\"a-b-a\", \"a\"->\"x\"]  ==> \"x-b-x\""],
    );
    ev.set_doc_examples(
        "Replace",
        &["Replace[{1,2,1,3}, 1->9]  ==> {9,2,1,3}", "Replace[\"2024-08-01\", DigitCharacter.. -> \"#\"]"],
    );

    // Assignment and scoping
    ev.set_doc("Set", "Assignment: Set[symbol, value].", &["symbol", "value"]);
    ev.set_doc("Unset", "Clear definition: Unset[symbol].", &["symbol"]);
    ev.set_doc("SetDelayed", "Delayed assignment evaluated on use.", &["symbol", "expr"]);
    ev.set_doc("With", "Lexically bind symbols within a body.", &["<|vars|>", "body"]);
    ev.set_doc_examples("Set", &["Set[x, 10]; x  ==> 10"]);
    ev.set_doc_examples("Unset", &["Set[x, 10]; Unset[x]; x  ==> x"]);
    ev.set_doc_examples("SetDelayed", &["SetDelayed[now, CurrentTime[]]; now  ==> 1690000000 (changes)"]);
    ev.set_doc_examples("With", &["With[<|\"x\"->2|>, x^3]  ==> 8"]);

    // Introspection and meta
    ev.set_doc("Schema", "Return a minimal schema for a value/association.", &["value"]);
    ev.set_doc("Explain", "Explain evaluation; returns trace steps when enabled.", &["expr"]);
    ev.set_doc("DescribeBuiltins", "List builtins with attributes (and docs when available).", &[]);
    ev.set_doc("Documentation", "Documentation card for a builtin.", &["name"]);
    ev.set_doc_examples("Schema", &["Schema[<|\"a\"->1, \"b\"->\"x\"|>]  ==> <|a->Integer, b->String|>"]);
    ev.set_doc_examples("Explain", &["Explain[Plus[1,2]]  ==> <|steps->...|>"]);

    // Selected stdlib helpers (non-exhaustive; expand over time)
    ev.set_doc("ToUpper", "Uppercase string.", &["s"]);
    ev.set_doc("ToLower", "Lowercase string.", &["s"]);
    ev.set_doc("StringJoin", "Concatenate list of parts.", &["parts"]);
    ev.set_doc("Length", "Length of a list or string.", &["x"]);
    ev.set_doc_examples("ToUpper", &["ToUpper[\"hi\"]  ==> \"HI\""]);
    ev.set_doc_examples("ToLower", &["ToLower[\"Hello\"]  ==> \"hello\""]);
    ev.set_doc_examples("StringJoin", &["StringJoin[{\"a\",\"b\",\"c\"}]  ==> \"abc\""]);
    ev.set_doc_examples("Length", &["Length[{1,2,3}]  ==> 3", "Length[\"ok\"]  ==> 2"]);

    // Generic verbs (dispatched by type)
    ev.set_doc("Insert", "Insert into collection or structure (dispatched)", &["target", "value"]);
    ev.set_doc("Upsert", "Insert or update within a collection or structure (dispatched)", &["target", "value"]);
    ev.set_doc(
        "Remove",
        "Remove from a collection/structure or remove a file/directory (dispatched). Overloads: Remove[target, value?]; Remove[path, opts?]",
        &["target|path", "value?|opts?"],
    );
    ev.set_doc("Add", "Add value to a collection (alias of Insert for some types)", &["target", "value"]);
    ev.set_doc("Info", "Information about a handle (Logger, Graph, etc.)", &["target"]);
    ev.set_doc("EmptyQ", "Is the subject empty? (lists, strings, assocs, handles)", &["x"]);
    ev.set_doc("Count", "Count items/elements (lists, assocs, Bag/VectorStore)", &["x"]);
    ev.set_doc("Search", "Search within a store or index (VectorStore, Index)", &["target", "query", "opts?"]);
    ev.set_doc("Reset", "Reset or clear a handle (e.g., VectorStore)", &["handle"]);
    ev.set_doc_examples(
        "Insert",
        &[
            "s := HashSet[{1,2}]; Insert[s, 3]  ==> s'",
            "q := Queue[]; Insert[q, 5]  ==> q'",
            "st := Stack[]; Insert[st, \"x\"]",
            "pq := PriorityQueue[]; Insert[pq, <|\"Key\"->1, \"Value\"->\"a\"|>]",
            "g := Graph[]; Insert[g, {\"a\",\"b\"}]",
            "Insert[g, <|Src->\"a\",Dst->\"b\"|>]",
        ],
    );
    ev.set_doc_examples(
        "Upsert",
        &[
            "vs := VectorStore[<|name->\"vs\"|>]; Upsert[vs, {<|id->\"a\", vec->{0.1,0.2,0.3}|>}]",
        ],
    );
    ev.set_doc_examples(
        "Remove",
        &[
            "Remove[{1,2,3}, 2]  ==> {1,3}",
            "Remove[Queue[]]  ==> Null (dequeue)",
            "Remove[Stack[]]  ==> Null (pop)",
            "Remove[PriorityQueue[]]  ==> Null (pop)",
            "Remove[g, {\"a\"}]",
            "Remove[g, {<|Src->\"a\",Dst->\"b\"|>}]",
        ],
    );
    ev.set_doc_examples(
        "Info",
        &[
            "Info[Graph[]]  ==> <|nodes->..., edges->...|>",
            "Info[DatasetFromRows[{<|a->1|>}]]  ==> <|Type->\"Dataset\", Rows->1, Columns->{\"a\"}|>",
            "Info[VectorStore[<|name->\"vs\"|>]]  ==> <|Type->\"VectorStore\", Name->\"vs\", Count->0|>",
            "Info[HashSet[{1,2,3}]]  ==> <|Type->\"Set\", Size->3|>",
            "Info[Queue[]]  ==> <|Type->\"Queue\", Size->0|>",
            "Info[Index[\"/tmp/idx.db\"]]  ==> <|indexPath->..., numDocs->...|>",
            "conn := Connect[\"mock://\"]; Info[conn]  ==> <|Type->\"Connection\", ...|>",
        ],
    );
    ev.set_doc_examples(
        "EmptyQ",
        &[
            "EmptyQ[{}]  ==> True",
            "EmptyQ[\"\"]  ==> True",
            "EmptyQ[Queue[]]  ==> True",
            "EmptyQ[DatasetFromRows[{}]]  ==> True",
        ],
    );
    ev.set_doc_examples(
        "Count",
        &[
            "Count[{1,2,3}]  ==> 3",
            "Count[<|a->1,b->2|>]  ==> 2",
            "Count[DatasetFromRows[{<|a->1|>,<|a->2|>}]]  ==> 2",
        ],
    );
    ev.set_doc_examples(
        "Search",
        &[
            "Search[VectorStore[<|name->\"vs\"|>], {0.1,0.2,0.3}]  ==> {...}",
            "idx := Index[\"/tmp/idx.db\"]; Search[idx, \"foo\"]",
        ],
    );

    // Import/Export + Sniff
    ev.set_doc(
        "Import",
        "Import data from path/URL into Frame (default), Dataset (Target->\"Dataset\"), or Value. Automatically sniffs Type/Delimiter/Header.",
        &["source", "opts?"],
    );
    ev.set_doc_examples(
        "Import",
        &[
            "Import[\"data.csv\"]  ==> Frame[...]",
            "Import[\"data.csv\", <|Target->\"Dataset\"|>]  ==> Dataset[...]",
            "Import[\"logs/*.jsonl\", <|Type->\"JSONL\"|>]  ==> Frame[...]",
        ],
    );
    ev.set_doc(
        "ImportString",
        "Parse in-memory strings into Frame/Dataset/Value. Automatically sniffs Type if missing.",
        &["content", "opts?"],
    );
    ev.set_doc_examples(
        "ImportString",
        &[
            "ImportString[\"a,b\\n1,2\"]  ==> Frame[...]",
            "ImportString[\"[{\\\"a\\\":1}]\", <|Target->\"Dataset\"|>]  ==> Dataset[...]",
        ],
    );
    ev.set_doc(
        "Export",
        "Export Frame/Dataset/Value to csv/json/jsonl/parquet (duckdb feature).",
        &["value", "dest", "opts?"],
    );
    ev.set_doc_examples(
        "Export",
        &[
            "Export[f, \"out.csv\"]  ==> True",
            "Export[ds, \"out.jsonl\"]  ==> True",
        ],
    );
    ev.set_doc(
        "Sniff",
        "Suggest Type and options for a source (file/url/string/bytes).",
        &["source"],
    );
    ev.set_doc_examples(
        "Sniff",
        &[
            "Sniff[\"data.csv\"]  ==> <|Type->\"CSV\", Delimiter->\",\", Header->True|>",
            "Sniff[\"https://ex.com/data.jsonl\"]  ==> <|Type->\"JSONL\"|>",
        ],
    );

    // Common predicate aliases (Q-suffixed)
    ev.set_doc("StartsWithQ", "Alias: StartsWith predicate", &["s", "prefix"]);
    ev.set_doc("EndsWithQ", "Alias: EndsWith predicate", &["s", "suffix"]);
    ev.set_doc("BlankQ", "Alias: IsBlank string predicate", &["s"]);
    ev.set_doc_examples("StartsWithQ", &["StartsWithQ[\"hello\", \"he\"]  ==> True"]);
    ev.set_doc_examples("EndsWithQ", &["EndsWithQ[\"hello\", \"lo\"]  ==> True"]);
    ev.set_doc_examples("BlankQ", &["BlankQ[\"   \"]  ==> True", "BlankQ[\"x\"]  ==> False"]);

    // Generic metadata and predicates
    ev.set_doc("Keys", "Keys/columns for assoc, rows, Dataset, or Frame", &["subject"]);
    ev.set_doc_examples(
        "Keys",
        &[
            "Keys[<|a->1,b->2|>]  ==> {a,b}",
            "Keys[{<|a->1|>,<|b->2|>}]  ==> {a,b}",
        ],
    );
    ev.set_doc("MemberQ", "Alias: membership predicate (Contains)", &["container", "item"]);
    ev.set_doc_examples(
        "MemberQ",
        &[
            "MemberQ[{1,2,3}, 2]  ==> True",
            "MemberQ[\"foobar\", \"bar\"]  ==> True",
        ],
    );
    ev.set_doc("ContainsKeyQ", "Key membership for assoc/rows/Dataset/Frame", &["subject", "key"]);
    ev.set_doc("HasKeyQ", "Alias: key membership predicate", &["subject", "key"]);
    ev.set_doc_examples(
        "ContainsKeyQ",
        &[
            "ContainsKeyQ[<|a->1|>, \"a\"]  ==> True",
            "ContainsKeyQ[{<|a->1|>,<|b->2|>}, \"b\"]  ==> True",
        ],
    );

    // Logic and control
    ev.set_doc("If", "Conditional: If[cond, then, else?] (held)", &["cond", "then", "else?"]);
    ev.set_doc("When", "Evaluate body when condition is True (held)", &["cond", "body"]);
    ev.set_doc("Unless", "Evaluate body when condition is False (held)", &["cond", "body"]);
    ev.set_doc("Switch", "Multi-way conditional by equals (held)", &["expr", "rules…"]);
    ev.set_doc("And", "Logical AND (short-circuit)", &["args…"]);
    ev.set_doc("Or", "Logical OR (short-circuit)", &["args…"]);
    ev.set_doc("Not", "Logical NOT", &["x"]);
    ev.set_doc("MatchQ", "Pattern match predicate (held)", &["expr", "pattern"]);
    ev.set_doc("PatternQ", "Is value a pattern? (held)", &["expr"]);
    ev.set_doc("Equal", "Test equality across arguments", &["args…"]);
    ev.set_doc("Less", "Strictly increasing sequence", &["args…"]);
    ev.set_doc("LessEqual", "Non-decreasing sequence", &["args…"]);
    ev.set_doc("Greater", "Strictly decreasing sequence", &["args…"]);
    ev.set_doc("GreaterEqual", "Non-increasing sequence", &["args…"]);
    ev.set_doc_examples("If", &["If[1<2, \"yes\", \"no\"]  ==> \"yes\""]);
    ev.set_doc_examples("When", &["When[True, Print[\"ok\"]]"]);
    ev.set_doc_examples("Unless", &["Unless[False, Print[\"ok\"]]"]);
    ev.set_doc_examples("And", &["And[True, False]  ==> False"]);
    ev.set_doc_examples("Or", &["Or[False, True]  ==> True"]);
    ev.set_doc_examples("Not", &["Not[True]  ==> False"]);
    ev.set_doc_examples("Equal", &["Equal[1,1,1]  ==> True"]);
    ev.set_doc_examples("Less", &["Less[1,2,3]  ==> True"]);
    ev.set_doc_examples("GreaterEqual", &["GreaterEqual[3,2,2]  ==> True"]);

    // Functional
    ev.set_doc("Apply", "Apply head to list elements: Apply[f, {…}]", &["f", "list", "level?"]);
    ev.set_doc("Compose", "Compose functions left-to-right", &["f", "g", "…"]);
    ev.set_doc("RightCompose", "Compose functions right-to-left", &["f", "g", "…"]);
    ev.set_doc("Nest", "Nest function n times: Nest[f, x, n]", &["f", "x", "n"]);
    ev.set_doc("NestList", "Nest and collect intermediate values", &["f", "x", "n"]);
    ev.set_doc("FoldList", "Cumulative fold producing intermediates", &["f", "init", "list"]);
    ev.set_doc("FixedPoint", "Iterate f until convergence", &["f", "x"]);
    ev.set_doc("FixedPointList", "List of iterates until convergence", &["f", "x"]);
    ev.set_doc("Through", "Through[{f,g}, x] applies each to x", &["fs", "x"]);
    ev.set_doc("Identity", "Identity function: returns its argument", &["x"]);
    ev.set_doc("ConstantFunction", "Constant function returning c", &["c"]);
    ev.set_doc("Try", "Try body; capture failures (held)", &["body", "handler?"]);
    ev.set_doc("OnFailure", "Handle Failure values (held)", &["body", "handler"]);
    ev.set_doc("Catch", "Catch a thrown value (held)", &["body", "handlers"]);
    ev.set_doc("Throw", "Throw a value for Catch", &["x"]);
    ev.set_doc("Finally", "Ensure cleanup runs (held)", &["body", "cleanup"]);
    ev.set_doc("TryOr", "Try body else default (held)", &["body", "default"]);
    ev.set_doc_examples("Apply", &["Apply[Plus, {1,2,3}]  ==> 6"]);
    ev.set_doc_examples("Compose", &["Compose[f,g][x]  ==> f[g[x]]"]);
    ev.set_doc_examples("Nest", &["Nest[#*2 &, 1, 3]  ==> 8"]);
    ev.set_doc_examples("FoldList", &["FoldList[Plus, 0, {1,2,3}]  ==> {0,1,3,6}"]);
    ev.set_doc_examples("FixedPoint", &["FixedPoint[Cos, 1.0]  ==> 0.739... "]);
    ev.set_doc_examples("Identity", &["Identity[42]  ==> 42"]);
    ev.set_doc_examples("TryOr", &["TryOr[1/0, \"fallback\"]  ==> \"fallback\""]);
    // Core extras
    ev.set_doc("ReplaceRepeated", "Repeatedly apply rules until fixed point (held)", &["expr", "rules"]);
    ev.set_doc("SetDownValues", "Attach DownValues to a symbol (held)", &["symbol", "defs"]);
    ev.set_doc("GetDownValues", "Return DownValues for a symbol", &["symbol"]);
    ev.set_doc("SetUpValues", "Attach UpValues to a symbol (held)", &["symbol", "defs"]);
    ev.set_doc("GetUpValues", "Return UpValues for a symbol", &["symbol"]);
    ev.set_doc("SetOwnValues", "Attach OwnValues to a symbol (held)", &["symbol", "defs"]);
    ev.set_doc("GetOwnValues", "Return OwnValues for a symbol", &["symbol"]);
    ev.set_doc("SetSubValues", "Attach SubValues to a symbol (held)", &["symbol", "defs"]);
    ev.set_doc("GetSubValues", "Return SubValues for a symbol", &["symbol"]);

    // Associations (canonical names)
    // Lookup removed (use Get)
    // Write is documented in a consolidated form above.
    ev.set_doc(
        "Select",
        "Select keys/columns or compute columns (dispatched). Overloads: Select[assoc, pred|keys]; Select[ds, cols]",
        &["assoc|ds", "pred|keys|cols"],
    );
    ev.set_doc("Delete", "Delete keys from association", &["assoc", "keys"]);
    ev.set_doc("Drop", "Drop keys and return remaining assoc", &["assoc", "keys"]);
    // ContainsKeyQ documented generically earlier; avoid duplicate.
    ev.set_doc("Invert", "Invert mapping values -> list of keys", &["assoc"]);
    ev.set_doc("RenameKeys", "Rename keys by mapping or function", &["assoc", "map|f"]);
    ev.set_doc("MapValues", "Map values with f[v]", &["f", "assoc"]);
    ev.set_doc("MapKeys", "Map keys with f[k]", &["f", "assoc"]);
    ev.set_doc("MapPairs", "Map over (k,v) pairs", &["f", "assoc"]);
    // Legacy Lookup removed; use Get
    ev.set_doc_examples("Write", &["Write[<|\"a\"->1|>, \"b\", 2]  ==> <|\"a\"->1, \"b\"->2|>"]);
    ev.set_doc_examples("MapValues", &["MapValues[ToUpper, <|\"a\"->\"x\"|>]  ==> <|\"a\"->\"X\"|>"]);

    // Lists (common)
    ev.set_doc("Range", "Create numeric range", &["a", "b", "step?"]);
    ev.set_doc(
        "Join",
        "Join lists or datasets (dispatched). Overloads: Join[list1, list2]; Join[left, right, on, how?]",
        &["a|left", "b|right", "on?", "how?"],
    );
    ev.set_doc("Reverse", "Reverse a list", &["list"]);
    ev.set_doc("Flatten", "Flatten by levels (default 1)", &["list", "levels?"]);
    ev.set_doc("Partition", "Partition into fixed-size chunks", &["list", "n", "step?"]);
    ev.set_doc("Transpose", "Transpose list of lists", &["rows"]);
    ev.set_doc("Filter", "Keep elements where pred[x] is True", &["pred", "list"]);
    ev.set_doc("Reject", "Drop elements where pred[x] is True", &["pred", "list"]);
    ev.set_doc("Any", "True if any matches (optionally with pred)", &["list", "pred?"]);
    ev.set_doc("All", "True if all match (optionally with pred)", &["list", "pred?"]);
    ev.set_doc("Find", "First element where pred[x]", &["pred", "list"]);
    ev.set_doc("Position", "1-based index of first match", &["pred", "list"]);
    ev.set_doc("Take", "Take first n (last if negative)", &["list", "n"]);
    ev.set_doc("Drop", "Drop first n (last if negative)", &["list", "n"]);
    ev.set_doc("TakeWhile", "Take while pred[x] holds", &["pred", "list"]);
    ev.set_doc("DropWhile", "Drop while pred[x] holds", &["pred", "list"]);
    ev.set_doc(
        "Zip",
        "Zip lists into pairs or create a .zip archive (dispatched). Overloads: Zip[a, b]; Zip[dest, inputs]",
        &["a|dest", "b|inputs"],
    );
    ev.set_doc(
        "Unzip",
        "Unzip list of pairs or extract a .zip (dispatched). Overloads: Unzip[pairs]; Unzip[src, dest]",
        &["pairs|src", "dest?"],
    );
    ev.set_doc(
        "Sort",
        "Sort a list or dataset (dispatched). Overloads: Sort[list]; Sort[ds, by, opts?]",
        &["list|ds", "by?", "opts?"],
    );
    ev.set_doc("Unique", "Stable deduplicate list", &["list"]);
    ev.set_doc("Tally", "Counts by value (assoc)", &["list"]);
    ev.set_doc("CountBy", "Counts by key function (assoc)", &["f", "list"]);
    ev.set_doc("Reduce", "Fold list with function", &["f", "init?", "list"]);
    ev.set_doc("Scan", "Prefix scan with function", &["f", "init?", "list"]);
    ev.set_doc("MapIndexed", "Map with index (1-based)", &["f", "list"]);
    ev.set_doc("Slice", "Slice list by start and length", &["list", "start", "len?"]);
    ev.set_doc_examples("Range", &["Range[1, 5]  ==> {1,2,3,4,5}", "Range[0, 10, 2]  ==> {0,2,4,6,8,10}"]);
    ev.set_doc_examples("Join", &["Join[{1,2},{3}]  ==> {1,2,3}"]);
    ev.set_doc_examples("Flatten", &["Flatten[{{1},{2,3}}]  ==> {1,2,3}"]);
    ev.set_doc_examples("Partition", &["Partition[{1,2,3,4}, 2]  ==> {{1,2},{3,4}}"]);
    ev.set_doc_examples("Filter", &["Filter[OddQ, {1,2,3,4}]  ==> {1,3}"]);
    ev.set_doc_examples("Reduce", &["Reduce[Plus, 0, {1,2,3}]  ==> 6"]);
    ev.set_doc_examples("Scan", &["Scan[Plus, 0, {1,2,3}]  ==> {0,1,3,6}"]);
    ev.set_doc_examples("MapIndexed", &["MapIndexed[({#1, #2} &), {10,20}]  ==> {{10,1},{20,2}}"]);
    ev.set_doc_examples("Slice", &["Slice[{10,20,30,40}, 2, 2]  ==> {20,30}"]);

    // Concurrency
    ev.set_doc("Future", "Create a Future from an expression (held)", &["expr"]);
    ev.set_doc("Await", "Wait for Future and return value", &["future"]);
    ev.set_doc("ParallelMap", "Map in parallel over list", &["f", "list"]);
    ev.set_doc("ParallelTable", "Evaluate list of expressions in parallel (held)", &["exprs"]);
    ev.set_doc("MapAsync", "Map to Futures over list", &["f", "list"]);
    ev.set_doc("Gather", "Await Futures in same structure", &["futures"]);
    ev.set_doc("Cancel", "Request cooperative cancellation", &["future"]);
    ev.set_doc("BusyWait", "Block for n milliseconds (testing only)", &["ms"]);
    ev.set_doc("Fail", "Construct a failure value (optionally with message)", &["message?"]);
    ev.set_doc("Scope", "Run body with resource limits (held)", &["opts", "body"]);
    ev.set_doc("StartScope", "Start a managed scope (held)", &["opts", "body"]);
    ev.set_doc("InScope", "Run body inside a scope (held)", &["scope", "body"]);
    ev.set_doc("CancelScope", "Cancel running scope", &["scope"]);
    ev.set_doc("EndScope", "End scope and release resources", &["scope"]);
    ev.set_doc("ParallelEvaluate", "Evaluate expressions concurrently (held)", &["exprs", "opts?"]);
    ev.set_doc("BoundedChannel", "Create bounded channel", &["n"]);
    ev.set_doc("Send", "Send value to channel (held)", &["ch", "value"]);
    ev.set_doc("Receive", "Receive value from channel", &["ch", "opts?"]);
    ev.set_doc("CloseChannel", "Close channel", &["ch"]);
    ev.set_doc("TrySend", "Non-blocking send (held)", &["ch", "value"]);
    ev.set_doc("TryReceive", "Non-blocking receive", &["ch"]);
    ev.set_doc("Actor", "Create actor with handler (held)", &["handler"]);
    ev.set_doc("Tell", "Send message to actor (held)", &["actor", "msg"]);
    ev.set_doc("Ask", "Request/response with actor (held)", &["actor", "msg"]);
    ev.set_doc("StopActor", "Stop actor", &["actor"]);
    ev.set_doc_examples("Future", &["f := Future[Range[1,1_000]]; Await[f]  ==> {1,2,...}"]);
    ev.set_doc_examples("ParallelMap", &["ParallelMap[#^2 &, Range[1,4]]  ==> {1,4,9,16}"]);
    ev.set_doc_examples("BoundedChannel", &["ch := BoundedChannel[2]; Send[ch, 1]; Receive[ch]  ==> 1"]);

    // File system (core)
    ev.set_doc("MakeDirectory", "Create a directory (Parents option)", &["path", "opts?"]);
    // Remove is documented in a consolidated form above.
    ev.set_doc("Copy", "Copy file or directory (Recursive option)", &["src", "dst", "opts?"]);
    ev.set_doc("Move", "Move or rename a file/directory", &["src", "dst"]);
    ev.set_doc("Touch", "Create file if missing (update mtime)", &["path"]);
    ev.set_doc("Symlink", "Create a symbolic link", &["src", "dst"]);
    ev.set_doc("Glob", "Expand glob pattern to matching paths", &["pattern"]);
    ev.set_doc("ReadBytes", "Read entire file as bytes", &["path"]);
    ev.set_doc("WriteBytes", "Write bytes to file", &["path", "bytes"]);
    ev.set_doc_examples("Glob", &["Glob[\"**/*.lyra\"]  ==> {\"examples/app.lyra\", ...}"]);
    ev.set_doc_examples("ReadBytes", &["ReadBytes[\"/etc/hosts\"]  ==> <byte list>"]);
    ev.set_doc_examples("WriteBytes", &["WriteBytes[\"/tmp/x.bin\", {0,255}]  ==> \"/tmp/x.bin\""]);
    ev.set_doc("TempFile", "Create a unique temporary file", &[]);
    ev.set_doc("TempDir", "Create a unique temporary directory", &[]);
    ev.set_doc(
        "WatchDirectory",
        "Watch directory and stream events (held)",
        &["path", "handler", "opts?"],
    );
    ev.set_doc(
        "Watch",
        "Watch directory and stream events (held)",
        &["path", "handler", "opts?"],
    );
    ev.set_doc("CancelWatch", "Cancel a directory watch", &["token"]);

    // I/O and paths (high-level; many also auto-seeded)
    ev.set_doc("ReadFile", "Read entire file as string", &["path"]);
    ev.set_doc("WriteFile", "Write stringified content to file", &["path", "content"]);
    ev.set_doc("ReadLines", "Read file into list of lines", &["path"]);
    ev.set_doc("FileExistsQ", "Does path exist?", &["path"]);
    ev.set_doc("ListDirectory", "List names in directory", &["path"]);
    ev.set_doc("Stat", "Basic file metadata as assoc", &["path"]);
    ev.set_doc("PathJoin", "Join path segments", &["parts"]);
    ev.set_doc("PathSplit", "Split path into parts", &["path"]);
    ev.set_doc("CanonicalPath", "Resolve symlinks and normalize", &["path"]);
    ev.set_doc("ExpandPath", "Expand ~ and env vars", &["path"]);
    ev.set_doc("PathNormalize", "Normalize path separators", &["path"]);
    ev.set_doc("PathRelative", "Relative path from base", &["base", "path"]);
    ev.set_doc("PathResolve", "Resolve against base directory", &["base", "path"]);
    ev.set_doc("CurrentDirectory", "Get current working directory", &[]);
    ev.set_doc("SetDirectory", "Change current working directory", &["path"]);
    ev.set_doc("Basename", "Filename without directories", &["path"]);
    ev.set_doc("Dirname", "Parent directory path", &["path"]);
    ev.set_doc("FileStem", "Filename without extension", &["path"]);
    ev.set_doc("FileExtension", "File extension (no dot)", &["path"]);

    // Process execution
    ev.set_doc("Run", "Run a process and capture output", &["cmd", "args?", "opts?"]);
    ev.set_doc("Which", "Resolve command path from PATH", &["cmd"]);
    ev.set_doc("CommandExistsQ", "Does a command exist in PATH?", &["cmd"]);
    ev.set_doc("Popen", "Spawn process and return handle", &["cmd", "args?", "opts?"]);
    ev.set_doc("WriteProcess", "Write to process stdin", &["proc", "data"]);
    ev.set_doc("ReadProcess", "Read from process stdout/stderr", &["proc", "opts?"]);
    ev.set_doc("WaitProcess", "Wait for process to exit", &["proc"]);
    ev.set_doc("KillProcess", "Send signal to process", &["proc", "signal?"]);
    ev.set_doc("Pipe", "Compose processes via pipes", &["cmds"]);

    // Time & scheduling
    ev.set_doc("NowMs", "Current UNIX time in milliseconds", &[]);
    ev.set_doc("MonotonicNow", "Monotonic clock milliseconds since start", &[]);
    ev.set_doc("Sleep", "Sleep for N milliseconds", &["ms"]);
    ev.set_doc("DateTime", "Build/parse DateTime assoc (UTC)", &["spec"]);
    ev.set_doc("DateParse", "Parse date/time string to epochMs", &["s"]);
    ev.set_doc("DateFormat", "Format DateTime or epochMs to string", &["dt", "fmt?"]);
    ev.set_doc("Duration", "Build Duration assoc from ms or fields", &["spec"]);
    ev.set_doc("DurationParse", "Parse human duration (e.g., 1h30m)", &["s"]);
    ev.set_doc("AddDuration", "Add duration to DateTime/epochMs", &["dt", "dur"]);
    ev.set_doc("DiffDuration", "Difference between DateTimes", &["a", "b"]);
    ev.set_doc("StartOf", "Start of unit (day/week/month)", &["dt", "unit"]);
    ev.set_doc("EndOf", "End of unit (day/week/month)", &["dt", "unit"]);
    ev.set_doc("TimeZoneConvert", "Convert DateTime to another timezone", &["dt", "tz"]);
    ev.set_doc("ScheduleEvery", "Schedule recurring task (held)", &["ms", "body"]);
    ev.set_doc("Cron", "Schedule with cron expression (held)", &["expr", "body"]);
    ev.set_doc("CancelSchedule", "Cancel scheduled task", &["token"]);

    // Logging
    ev.set_doc("ConfigureLogging", "Configure log level/format/output", &["opts"]);
    ev.set_doc("LogMessage", "Emit a log message with level and meta", &["level", "msg", "meta?"]);
    ev.set_doc(
        "WithLogger",
        "Add contextual metadata while evaluating body (held)",
        &["meta", "body"],
    );
    ev.set_doc("SetLogLevel", "Set global log level", &["level"]);
    ev.set_doc("GetLogger", "Get current logger configuration", &[]);
}

// Best-effort auto population from ToolsDescribe specs exposed by modules.
// This sweeps all builtins and, when a spec exists, seeds the Evaluator docs registry.
pub fn autoseed_from_tools(ev: &mut Evaluator) {
    // Get names via DescribeBuiltins
    let names = match ev.eval(lyra_core::value::Value::Expr {
        head: Box::new(lyra_core::value::Value::Symbol("DescribeBuiltins".into())),
        args: vec![],
    }) {
        lyra_core::value::Value::List(vs) => vs,
        _ => vec![],
    };
    for v in names {
        if let lyra_core::value::Value::Assoc(m) = v {
            if let Some(lyra_core::value::Value::String(name)) = m.get("name") {
                // Skip if already has a non-empty summary
                if let Some((sum, _p)) = ev.get_doc(name) {
                    if !sum.is_empty() {
                        continue;
                    }
                }
                // Ask ToolsDescribe; many modules publish specs via tool_spec! macros
                let spec = ev.eval(lyra_core::value::Value::Expr {
                    head: Box::new(lyra_core::value::Value::Symbol("ToolsDescribe".into())),
                    args: vec![lyra_core::value::Value::String(name.clone())],
                });
                if let lyra_core::value::Value::Assoc(sm) = spec {
                    let summary = match sm.get("summary") {
                        Some(lyra_core::value::Value::String(s)) => s.clone(),
                        _ => String::new(),
                    };
                    let params: Vec<String> = match sm.get("params") {
                        Some(lyra_core::value::Value::List(vs)) => vs
                            .iter()
                            .filter_map(|x| match x {
                                lyra_core::value::Value::String(s) => Some(s.clone()),
                                _ => None,
                            })
                            .collect(),
                        _ => vec![],
                    };
                    if !summary.is_empty() || !params.is_empty() {
                        ev.set_doc(
                            name,
                            summary,
                            &params.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
                        );
                    }
                    if let Some(lyra_core::value::Value::List(vs)) = sm.get("examples") {
                        let exs: Vec<String> = vs
                            .iter()
                            .filter_map(|x| match x {
                                lyra_core::value::Value::String(s) => Some(s.clone()),
                                _ => None,
                            })
                            .collect();
                        if !exs.is_empty() {
                            ev.set_doc_examples(
                                name,
                                &exs.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
                            );
                        }
                    }
                }
            }
        }
    }
}

// Additional broad coverage seeds for modules without inline tool specs.
pub fn register_docs_extra(ev: &mut Evaluator) {
    // Database (connections, SQL, cursors)
    ev.set_doc("Connect", "Open database connection (mock/sqlite/duckdb)", &["dsn|opts"]);
    ev.set_doc("Disconnect", "Close a database connection", &["conn"]);
    ev.set_doc("Ping", "Check connectivity for a database connection", &["conn"]);
    ev.set_doc("ListTables", "List tables on a connection", &["conn"]);
    ev.set_doc(
        "RegisterTable",
        "Register in-memory rows as a table (mock)",
        &["conn", "name", "rows"],
    );
    ev.set_doc("Table", "Reference a table as a Dataset", &["conn", "name"]);
    ev.set_doc("Query", "Run a SELECT query and return rows", &["conn", "sql", "opts?"]);
    ev.set_doc("Execute", "Execute DDL/DML (non-SELECT)", &["conn", "sql", "opts?"]);
    ev.set_doc("SQLCursor", "Run a query and return a cursor handle", &["conn", "sql", "params?"]);
    ev.set_doc("Fetch", "Fetch next batch of rows from a cursor", &["cursor", "limit?"]);
    ev.set_doc("Close", "Close an open handle (cursor, channel)", &["handle"]);
    ev.set_doc(
        "InsertRows",
        "Insert multiple rows (assoc list) into a table",
        &["conn", "table", "rows"],
    );
    ev.set_doc(
        "UpsertRows",
        "Upsert rows (assoc list) into a table",
        &["conn", "table", "rows", "keys?"],
    );
    ev.set_doc(
        "WriteDataset",
        "Write a Dataset into a table",
        &["conn", "table", "dataset", "opts?"],
    );
    ev.set_doc("Begin", "Begin a transaction", &["conn"]);
    ev.set_doc("Commit", "Commit the current transaction", &["conn"]);
    ev.set_doc("Rollback", "Rollback the current transaction", &["conn"]);
    // DB examples
    ev.set_doc_examples("Connect", &["conn := Connect[\"mock://\"]", "Ping[conn]  ==> True"]);
    ev.set_doc_examples(
        "RegisterTable",
        &[
            "rows := {<|\"id\"->1, \"name\"->\"a\"|>, <|\"id\"->2, \"name\"->\"b\"|>}",
            "conn := Connect[\"mock://\"]; RegisterTable[conn, \"t\", rows]  ==> True",
        ],
    );
    ev.set_doc_examples("ListTables", &["ListTables[conn]  ==> {\"t\"}"]);
    ev.set_doc_examples("Table", &["ds := Table[conn, \"t\"]; Head[ds,1]  ==> {<|...|>}"]);
    ev.set_doc_examples("Query", &["Query[conn, \"SELECT * FROM t\"]  ==> Dataset[...] "]);
    ev.set_doc_examples("Execute", &["Execute[conn, \"CREATE TABLE x(id INT)\"]  ==> <|Status->0|>"]);
    ev.set_doc_examples("Begin", &["Begin[conn]; Execute[conn, \"INSERT ...\"]; Commit[conn]"]);

    // Associations (immutable helpers)
    // Assoc* legacy names removed (pre-release). Use Get/ContainsKeyQ/Set/Delete/Select/Drop/Invert/RenameKeys.
    ev.set_doc("MapKeyValues", "Map over (k,v) pairs to new values or pairs", &["f", "assoc"]);
    // Legacy Assoc* examples removed.
    ev.set_doc_examples("MapKeyValues", &[
        "MapKeyValues[(k,v)=>k<>ToString[v], <|\"a\"->1, \"b\"->2|>]  ==> {\"a1\", \"b2\"}",
    ]);

    // Collections (set/list operations): use Union/Intersection/Difference via dispatchers.

    // Strings
    // Legacy StringLength/StringSplit removed (pre-release). Use Length/Split.
    // Generic helpers (dispatchers)
    ev.set_doc("Get", "Get value by key/index from a structure (dispatch)", &["subject", "key", "default?"]);
    ev.set_doc_examples("Get", &[
        "Get[<|\"a\"->1|>, \"a\"]  ==> 1",
        "Get[<|\"a\"->1|>, \"b\", 9]  ==> 9",
    ]);

    // Net/HTTP
    // HttpServer legacy name removed (pre-release). Use HttpServe or HttpServeRoutes.
    ev.set_doc("Respond", "Construct HTTP response from body/status/headers", &["body", "status?", "headers?"]);
    // Example removed with legacy name.
    ev.set_doc_examples("Respond", &[
        "Respond[\"ok\", 200]  ==> <|\"Status\"->200, ...|>",
        "Respond[\"Json\", <|\"a\"->1|>]  ==> <|\"Status\"->200, \"Body\"->\"{\\\"a\\\":1}\", ...|>",
    ]);

    // Files/Resources
    ev.set_doc("UsingFile", "Open a file and pass handle to a function (ensures close)", &["path", "fn"]);
    ev.set_doc_examples("UsingFile", &[
        "UsingFile[\"/tmp/x.txt\", (f)=>Puts[f, \"hi\"]]",
        "ReadFile[\"/tmp/x.txt\"]  ==> \"hi\"",
    ]);

    // Notebook helpers
    ev.set_doc("Cells", "Get list of cells in a notebook", &["notebook"]);
    ev.set_doc("CellCreate", "Create a new cell association", &["type", "input", "opts?"]);
    ev.set_doc("CellInsert", "Insert a cell into a notebook at position", &["notebook", "cell", "pos"]);
    ev.set_doc("CellDelete", "Delete a cell by UUID", &["notebook", "id"]);
    ev.set_doc("CellMove", "Move a cell to index", &["notebook", "id", "toIndex"]);
    ev.set_doc("CellUpdate", "Update fields on a cell by UUID", &["notebook", "id", "updates"]);
    ev.set_doc("ClearOutputs", "Clear output of code cells", &["notebook"]);
    ev.set_doc("NotebookMetadata", "Get notebook-level metadata", &["notebook"]);
    ev.set_doc("NotebookSetMetadata", "Set notebook-level metadata", &["notebook", "updates"]);

    // Linear algebra
    ev.set_doc("LU", "LU factorization: returns <|L,U,P|>", &["A"]);
    ev.set_doc("Cholesky", "Cholesky factorization for SPD matrices", &["A"]);
    ev.set_doc_examples("LU", &[
        "LU[{{1,2},{3,4}}]  ==> <|L->..., U->..., P->...|>",
        "{L,U,P} := Values[LU[A]]; L.U ≈ P.A",
    ]);
    ev.set_doc_examples("Cholesky", &[
        "Cholesky[{{4,1},{1,3}}]  ==> <|L->...|>",
        "L := Cholesky[A][\"L\"]; L.Transpose[L] == A",
    ]);

    // Signal processing (stubs)
    ev.set_doc("FilterFIR", "Finite impulse response filter (stub)", &["x", "coeffs", "opts?"]);
    ev.set_doc("FilterIIR", "Infinite impulse response filter (stub)", &["x", "coeffs", "opts?"]);
    ev.set_doc_examples("FilterFIR", &["FilterFIR[{1,2,3}, {0.2,0.2,0.2}]  ==> {...}"]);
    ev.set_doc_examples("FilterIIR", &["FilterIIR[{1,2,3}, {1.0, -0.5}]  ==> {...}"]);

    // Math (missing singletons)
    ev.set_doc("Tanh", "Hyperbolic tangent (Listable)", &["x"]);
    ev.set_doc_examples("Tanh", &[
        "Tanh[0]  ==> 0",
        "Tanh[1]  ==> 0.76159...",
        "Tanh[{0,1}]  ==> {0, 0.76159...}",
    ]);

    // Neural network layers and ops (constructor docs)
    ev.set_doc("LinearLayer", "Fully-connected linear layer", &["out|opts"]);
    ev.set_doc("ActivationLayer", "Activation layer (e.g., ReLU, Tanh)", &["name|opts"]);
    ev.set_doc("BatchNormLayer", "Batch normalization layer", &["opts?"]);
    ev.set_doc("LayerNormLayer", "Layer normalization layer", &["opts?"]);
    ev.set_doc("DropoutLayer", "Dropout layer", &["p|opts"]);
    ev.set_doc("EmbeddingLayer", "Embedding lookup layer", &["vocab", "dim", "opts?"]);
    ev.set_doc("ConcatLayer", "Concatenate along channel/feature axis", &["opts?"]);
    ev.set_doc("MulLayer", "Elementwise multiplication layer", &["opts?"]);
    ev.set_doc("SoftmaxLayer", "Softmax over features", &["opts?"]);
    ev.set_doc("ConvolutionLayer", "1D convolution layer", &["outChannels", "kernelSize", "opts?"]);
    ev.set_doc("PoolingLayer", "1D pooling layer (Max/Avg)", &["kind", "size", "opts?"]);
    ev.set_doc("FlattenLayer", "Flatten leading dims to vector", &["opts?"]);
    ev.set_doc("ReshapeLayer", "Reshape tensor to new shape", &["shape|opts"]);
    ev.set_doc("TransposeLayer", "Transpose/permute axes", &["perm|opts"]);
    ev.set_doc("AddLayer", "Elementwise addition layer", &["opts?"]);
    // Networks
    ev.set_doc("NetChain", "Compose layers sequentially", &["layers", "opts?"]);
    ev.set_doc("NetInitialize", "Initialize a network (weights/state)", &["net", "opts?"]);
    ev.set_doc("NetTrain", "Train a network on data", &["net", "data", "opts?"]);
    ev.set_doc("NetApply", "Apply network to input(s)", &["net", "x", "opts?"]);
    ev.set_doc("NetProperty", "Get network property", &["net", "key"]);
    ev.set_doc("NetSummary", "Human-readable network summary", &["net"]);
    ev.set_doc("NetEncoder", "Construct an input encoder", &["spec|auto"]);
    ev.set_doc("NetDecoder", "Construct an output decoder", &["spec|auto"]);
    ev.set_doc_examples("NetChain", &[
        "net := NetChain[{LinearLayer[8], ActivationLayer[\"relu\"], LinearLayer[1]}]",
    ]);
    ev.set_doc_examples("NetInitialize", &[
        "neti := NetInitialize[net]  ==> net' (initialized)",
    ]);
    ev.set_doc_examples("NetApply", &[
        "NetApply[neti, {1.0, 2.0}]  ==> {...}",
    ]);
    ev.set_doc_examples("NetSummary", &[
        "NetSummary[neti]  ==> \"Layer (out) ...\"",
    ]);

    // Datasets (declarative transformations)
    ev.set_doc("DatasetFromRows", "Create dataset from list of row assocs", &["rows"]);
    ev.set_doc("Collect", "Materialize dataset rows as a list", &["ds", "limit?", "opts?"]);
    ev.set_doc("Columns", "List column names for a dataset", &["ds"]);
    ev.set_doc("DatasetSchema", "Describe schema for a dataset", &["ds"]);
    ev.set_doc("ShowDataset", "Pretty-print a dataset table to string", &["ds", "opts?"]);
    ev.set_doc("ReadCSVDataset", "Read a CSV file into a dataset", &["path", "opts?"]);
    ev.set_doc("ReadJsonLinesDataset", "Read a JSONL file into a dataset", &["path", "opts?"]);
    // Select documented in a consolidated form above.
    ev.set_doc("SelectCols", "Select subset of columns by name", &["ds", "cols"]);
    ev.set_doc("RenameCols", "Rename columns via mapping", &["ds", "mapping"]);
    ev.set_doc("FilterRows", "Filter rows by predicate (held)", &["ds", "pred"]);
    ev.set_doc("LimitRows", "Limit number of rows", &["ds", "n"]);
    ev.set_doc("Offset", "Skip first n rows", &["ds", "n"]);
    ev.set_doc("Head", "Take first n rows", &["ds", "n"]);
    ev.set_doc("Tail", "Take last n rows", &["ds", "n"]);
    ev.set_doc("WithColumns", "Add/compute new columns (held)", &["ds", "defs"]);
    ev.set_doc("GroupBy", "Group rows by key(s)", &["ds", "keys"]);
    ev.set_doc("Agg", "Aggregate groups to single rows", &["ds", "aggs"]);
    // Sort documented in a consolidated form above.
    ev.set_doc("Distinct", "Drop duplicate rows (optionally by columns)", &["ds", "cols?"]);
    ev.set_doc(
        "DistinctOn",
        "Keep one row per key with order policy",
        &["ds", "keys", "orderBy?", "keepLast?"],
    );
    // Join documented in a consolidated form above.
    ev.set_doc("Union", "Union multiple datasets (by columns)", &["inputs", "byColumns?"]);
    ev.set_doc("Concat", "Concatenate datasets by rows (schema-union)", &["inputs"]);
    ev.set_doc("ExplainDataset", "Inspect logical plan for a dataset", &["ds"]);
    ev.set_doc("ExplainSQL", "Render SQL for pushdown-capable parts", &["ds"]);
    ev.set_doc_examples("ReadCSVDataset", &["ds := ReadCSVDataset[\"people.csv\"]", "Head[ds, 3]  ==> {{...},{...},{...}}"]);
    ev.set_doc_examples("Select", &["Select[ds, <|\"name\"->name, \"age2\"->age*2|>]  ==> ds'"]);
    ev.set_doc_examples("FilterRows", &["FilterRows[ds, age > 30]  ==> ds'"]);
    ev.set_doc_examples("GroupBy", &["GroupBy[ds, dept]  ==> grouped"]);
    ev.set_doc_examples("Agg", &["Agg[grouped, <|\"n\"->Count[], \"avg\"->Mean[salary]|>]  ==> ds'"]);
    ev.set_doc_examples("ExplainDataset", &["ExplainDataset[ds]  ==> <|plan->...|>"]);

    // Containers
    ev.set_doc("ConnectContainers", "Connect to container runtime", &["opts?"]);
    ev.set_doc("DisconnectContainers", "Disconnect from container runtime", &[]);
    ev.set_doc("RuntimeInfo", "Runtime version and info", &[]);
    ev.set_doc("RuntimeCapabilities", "Supported features and APIs", &[]);
    ev.set_doc("PullImage", "Pull an image", &["ref", "opts?"]);
    ev.set_doc("BuildImage", "Build image from context", &["context", "opts?"]);
    ev.set_doc("TagImage", "Tag an image", &["src", "dest"]);
    ev.set_doc("PushImage", "Push an image to registry", &["ref", "opts?"]);
    ev.set_doc("RemoveImage", "Remove local image", &["ref"]);
    ev.set_doc("PruneImages", "Remove unused images", &["opts?"]);
    ev.set_doc("SaveImage", "Save image to tar", &["ref", "path"]);
    ev.set_doc("LoadImage", "Load image from tar", &["path"]);
    ev.set_doc("ListImages", "List local images", &["opts?"]);
    ev.set_doc("InspectImage", "Inspect image details", &["ref"]);
    ev.set_doc("SearchImages", "Search registry images", &["query", "opts?"]);
    ev.set_doc("InspectRegistryImage", "Inspect remote registry image", &["ref", "opts?"]);
    ev.set_doc("ExportImages", "Export images to an archive", &["refs", "path"]);
    // SSH (feature ssh)
    ev.set_doc("SshConnect", "Open SSH connection (password/key/agent).", &["opts?"]);
    ev.set_doc("SshDisconnect", "Close SSH session.", &["session"]);
    ev.set_doc("SshSessionInfo", "Describe SSH session.", &["session"]);
    ev.set_doc("SshExec", "Execute a remote command; returns code/stdout/stderr.", &["session", "command", "opts?"]);
    ev.set_doc("SshUpload", "Upload a local file to remote path (SCP).", &["session","sourcePath","destPath","opts?"]);
    ev.set_doc("SshDownload", "Download remote file (SCP).", &["session","path","opts?"]);
    ev.set_doc("SshHostKey", "Return server host key and SHA256 fingerprint.", &["session"]);
    ev.set_doc("SshCopyId", "Append a public key to ~/.ssh/authorized_keys.", &["session","publicKeyOpenSsh","opts?"]);
    ev.set_doc("SshKeyGen", "Generate SSH keypair: ed25519 (default) and rsa (with ssh_openssh).", &["opts?"]);

    ev.set_doc("Container", "Create a container", &["image", "opts?"]);
    ev.set_doc("StartContainer", "Start a container", &["id"]);
    ev.set_doc("StopContainer", "Stop a container", &["id", "opts?"]);
    ev.set_doc("RestartContainer", "Restart a container", &["id"]);
    ev.set_doc("RemoveContainer", "Remove a container", &["id", "opts?"]);
    ev.set_doc("PauseContainer", "Pause a container", &["id"]);
    ev.set_doc("UnpauseContainer", "Unpause a container", &["id"]);
    ev.set_doc("RenameContainer", "Rename a container", &["id", "name"]);
    ev.set_doc("ExecInContainer", "Run a command inside a container", &["id", "cmd", "opts?"]);
    ev.set_doc("CopyToContainer", "Copy file/dir into container", &["id", "src", "dst"]);
    ev.set_doc("CopyFromContainer", "Copy file/dir from container", &["id", "src", "dst"]);
    ev.set_doc("InspectContainer", "Inspect container", &["id"]);
    ev.set_doc("WaitContainer", "Wait for container to stop", &["id", "opts?"]);
    ev.set_doc("ListContainers", "List containers", &["opts?"]);
    ev.set_doc("Logs", "Stream container logs", &["id", "opts?"]);
    ev.set_doc("Stats", "Stream container stats", &["id", "opts?"]);
    ev.set_doc("Events", "Subscribe to runtime events", &["opts?"]);
    ev.set_doc("ContainersFetch", "Fetch external resources for build", &["paths", "opts?"]);
    ev.set_doc("ContainersClose", "Close open fetch handles", &[]);

    ev.set_doc("ListVolumes", "List volumes", &[]);
    ev.set_doc("Volume", "Create volume", &["opts?"]);
    ev.set_doc("RemoveVolume", "Remove volume", &["name"]);
    ev.set_doc("InspectVolume", "Inspect volume", &["name"]);
    ev.set_doc("ListNetworks", "List networks", &[]);
    ev.set_doc("Network", "Create network", &["opts?"]);
    ev.set_doc("RemoveNetwork", "Remove network", &["name"]);
    ev.set_doc("InspectNetwork", "Inspect network", &["name"]);
    ev.set_doc("AddRegistryAuth", "Add registry credentials", &["server", "user", "password"]);
    ev.set_doc("ListRegistryAuth", "List stored registry credentials", &[]);
    ev.set_doc("ExplainContainers", "Explain container runtime configuration", &[]);
    ev.set_doc("DescribeContainers", "Describe available container APIs", &[]);
    ev.set_doc_examples("PullImage", &["PullImage[\"alpine:latest\"]  ==> <|id->..., size->...|>"]);
    ev.set_doc_examples("Container", &["cid := Container[\"alpine\", <|\"cmd\"->\"echo hi\"|>]", "StartContainer[cid]"]);
    ev.set_doc_examples("ExecInContainer", &["ExecInContainer[cid, {\"ls\", \"/\"}]  ==> <|code->0, out->..., err->...|>"]);

    // Graphs
    ev.set_doc("Graph", "Create a graph handle", &["opts?"]);
    ev.set_doc("DropGraph", "Drop a graph handle", &["graph"]);
    ev.set_doc("GraphInfo", "Summary and counts for graph", &["graph"]);
    ev.set_doc("AddNodes", "Add nodes by id and attributes", &["graph", "nodes"]);
    ev.set_doc("UpsertNodes", "Insert or update nodes", &["graph", "nodes"]);
    ev.set_doc("AddEdges", "Add edges with optional keys and weights", &["graph", "edges"]);
    ev.set_doc("UpsertEdges", "Insert or update edges", &["graph", "edges"]);
    ev.set_doc("RemoveNodes", "Remove nodes by id", &["graph", "ids"]);
    ev.set_doc("RemoveEdges", "Remove edges by id or (src,dst,key)", &["graph", "edges"]);
    ev.set_doc("ListNodes", "List nodes", &["graph", "opts?"]);
    ev.set_doc("ListEdges", "List edges", &["graph", "opts?"]);
    ev.set_doc("HasNode", "Does graph contain node?", &["graph", "id"]);
    ev.set_doc("HasEdge", "Does graph contain edge?", &["graph", "spec"]);
    ev.set_doc("Neighbors", "Neighbor node ids for a node", &["graph", "id", "opts?"]);
    ev.set_doc("IncidentEdges", "Edges incident to a node", &["graph", "id", "opts?"]);
    ev.set_doc("Subgraph", "Induced subgraph from node set", &["graph", "ids"]);
    ev.set_doc("SampleNodes", "Sample k nodes uniformly", &["graph", "k"]);
    ev.set_doc("SampleEdges", "Sample k edges uniformly", &["graph", "k"]);
    ev.set_doc("BFS", "Breadth-first search order", &["graph", "start", "opts?"]);
    ev.set_doc("DFS", "Depth-first search order", &["graph", "start", "opts?"]);
    ev.set_doc("ShortestPaths", "Shortest path distances from start", &["graph", "start", "opts?"]);
    ev.set_doc("ConnectedComponents", "Weakly connected components", &["graph"]);
    ev.set_doc("StronglyConnectedComponents", "Strongly connected components", &["graph"]);
    ev.set_doc("TopologicalSort", "Topologically sort DAG nodes", &["graph"]);
    ev.set_doc("PageRank", "PageRank centrality scores", &["graph", "opts?"]);
    ev.set_doc("DegreeCentrality", "Per-node degree centrality", &["graph"]);
    ev.set_doc("ClosenessCentrality", "Per-node closeness centrality", &["graph"]);
    ev.set_doc("LocalClustering", "Per-node clustering coefficient", &["graph"]);
    ev.set_doc("GlobalClustering", "Global clustering coefficient", &["graph"]);
    ev.set_doc("KCoreDecomposition", "K-core index per node", &["graph"]);
    ev.set_doc("KCore", "Induced subgraph with k-core", &["graph", "k"]);
    ev.set_doc("MinimumSpanningTree", "Edges in a minimum spanning tree", &["graph"]);
    ev.set_doc("MaxFlow", "Maximum flow value and cut", &["graph", "src", "dst"]);
    ev.set_doc_examples("Graph", &["g := Graph[]; AddNodes[g, {\"a\",\"b\"}]"]);
    ev.set_doc_examples("AddEdges", &["AddEdges[g, {{\"a\",\"b\"}}]"]);
    ev.set_doc_examples("BFS", &["BFS[g, \"a\"]  ==> {\"a\",\"b\"}"]);

    // NDArray (numerical arrays)
    ev.set_doc("NDArray", "Create NDArray from list/shape/data (held)", &["spec"]);
    ev.set_doc("NDShape", "Shape of an NDArray", &["a"]);
    ev.set_doc("NDReshape", "Reshape array to new shape", &["a", "shape"]);
    ev.set_doc("NDTranspose", "Transpose array axes", &["a", "perm?"]);
    ev.set_doc("NDPermuteDims", "Permute dimensions by order", &["a", "perm"]);
    ev.set_doc("NDConcat", "Concatenate arrays along axis", &["arrays", "axis?"]);
    ev.set_doc("NDSlice", "Slice array by ranges per axis", &["a", "slices"]);
    ev.set_doc("NDMap", "Map function elementwise (held)", &["f", "a"]);
    ev.set_doc("NDReduce", "Reduce over axes with function (held)", &["f", "a", "axes?"]);
    ev.set_doc("NDMatMul", "Matrix multiply A x B", &["a", "b"]);
    ev.set_doc("NDSum", "Sum over all elements or axes", &["a", "axes?"]);
    ev.set_doc("NDMean", "Mean over all elements or axes", &["a", "axes?"]);
    ev.set_doc("NDArgMax", "Argmax index per axis or flattened", &["a", "axis?"]);
    ev.set_doc("NDType", "Element type (f64/i64/...) of array", &["a"]);
    ev.set_doc("NDAsType", "Cast array to a new element type", &["a", "type"]);
    ev.set_doc("NDAdd", "Elementwise addition with broadcast", &["a", "b"]);
    ev.set_doc("NDSub", "Elementwise subtraction with broadcast", &["a", "b"]);
    ev.set_doc("NDMul", "Elementwise multiplication with broadcast", &["a", "b"]);
    ev.set_doc("NDDiv", "Elementwise division with broadcast", &["a", "b"]);
    ev.set_doc("NDEltwise", "Apply custom elementwise op (held)", &["f", "a", "b?"]);
    ev.set_doc("NDPow", "Elementwise exponentiation with broadcast", &["a", "b"]);
    ev.set_doc("NDClip", "Clip array values to [min,max]", &["a", "min", "max"]);
    ev.set_doc("NDRelu", "ReLU activation (max(0, x))", &["a"]);
    ev.set_doc("NDExp", "Elementwise exp", &["a"]);
    ev.set_doc("NDSqrt", "Elementwise sqrt", &["a"]);
    ev.set_doc("NDLog", "Elementwise natural log", &["a"]);
    ev.set_doc("NDSin", "Elementwise sin", &["a"]);
    ev.set_doc("NDCos", "Elementwise cos", &["a"]);
    ev.set_doc("NDTanh", "Elementwise tanh", &["a"]);
    ev.set_doc_examples("NDArray", &["a := NDArray[<|\"shape\"->{2,2}, \"data\"->{1,2,3,4}|>]"]);
    ev.set_doc_examples("NDShape", &["NDShape[a]  ==> {2,2}"]);
    ev.set_doc_examples("NDMatMul", &["NDMatMul[NDArray[{{1,2},{3,4}}], NDArray[{{5,6},{7,8}}]]  ==> NDArray[...] "]);

    // Canonical tensor entries (dispatch to ND implementations)
    ev.set_doc("Tensor", "Create a tensor (alias of NDArray)", &["spec"]);
    ev.set_doc("Shape", "Shape of a tensor", &["tensor"]);
    ev.set_doc("Reshape", "Reshape a tensor to new dims (supports -1)", &["tensor", "dims"]);

    // Elementwise activation (generic)
    ev.set_doc("Relu", "Rectified Linear Unit: max(0, x). Tensor-aware: elementwise on tensors.", &["x"]);
    ev.set_doc_examples("Relu", &["Relu[-2]  ==> 0", "Relu[Tensor[{-1,0,2}]]  ==> Tensor[...]"]);

    // Generic NN/ML verbs (dispatch)
    ev.set_doc("Train", "Train a network (dispatch to NetTrain)", &["net", "data", "opts?"]);
    ev.set_doc("Initialize", "Initialize a network (dispatch to NetInitialize)", &["net", "opts?"]);
    ev.set_doc("Property", "Property of a network or ML model (dispatch)", &["obj", "key"]);
    ev.set_doc("Summary", "Summary of a network (dispatch to NetSummary)", &["obj"]);

    // IO examples
    ev.set_doc_examples("ReadFile", &["WriteFile[\"/tmp/x.txt\", \"hi\"]; ReadFile[\"/tmp/x.txt\"]  ==> \"hi\""]);
    ev.set_doc_examples("WriteFile", &["WriteFile[\"/tmp/y.txt\", <|\"a\"->1|>]  ==> True"]);
    ev.set_doc_examples("ReadLines", &["ReadLines[\"/etc/hosts\"]  ==> {\"127.0.0.1 localhost\", ...}"]);
    ev.set_doc_examples("FileExistsQ", &["FileExistsQ[\"/etc/passwd\"]  ==> True"]);
    ev.set_doc_examples("PathJoin", &["PathJoin[{\"/tmp\", \"dir\", \"file.txt\"}]  ==> \"/tmp/dir/file.txt\""]);
    ev.set_doc_examples("Basename", &["Basename[\"/a/b/c.txt\"]  ==> \"c.txt\""]);
    ev.set_doc_examples("ToJson", &["ToJson[<|\"a\"->1|>]  ==> \"{\\\"a\\\":1}\""]);
    ev.set_doc_examples("FromJson", &["FromJson[\"{\\\"a\\\":1}\"]  ==> <|\"a\"->1|>"]);
    ev.set_doc_examples("ParseCSV", &["ParseCSV[\"a,b\\n1,2\\n\"]  ==> {{\"a\",\"b\"},{\"1\",\"2\"}}"]);
    ev.set_doc_examples("RenderCSV", &["RenderCSV[{{\"a\",\"b\"},{1,2}}]  ==> \"a,b\\n1,2\\n\""]);
    // File IO convenience docs
    ev.set_doc("ReadJson", "Read JSON file and parse.", &["path"]);
    ev.set_doc("WriteJson", "Write value to JSON file (opts: pretty, sortKeys).", &["path","value","opts?"]);
    ev.set_doc("ReadCsv", "Read CSV file into rows/assocs.", &["path","opts?"]);
    ev.set_doc("WriteCsv", "Write rows/assocs to CSV file.", &["path","rows","opts?"]);
    ev.set_doc("ReadParquet", "Read Parquet file(s) into a Dataset (requires DuckDB).", &["path","opts?"]);
    ev.set_doc("WriteParquet", "Write Dataset to Parquet (requires DuckDB).", &["dataset","path","opts?"]);
    ev.set_doc("ReadArrow", "Read Arrow/Feather file(s) into a Dataset (requires DuckDB).", &["path","opts?"]);
    ev.set_doc_examples("ReadJson", &["WriteJson[\"/tmp/z.json\", <|a->1|>]; ReadJson[\"/tmp/z.json\"]  ==> <|a->1|>"]);
    ev.set_doc_examples("ReadCsv", &["ReadCsv[\"/tmp/people.csv\"]  ==> {<|name->\"a\"|>, ...}"]);

    // IO docs (terminal formatting and primitives)
    ev.set_doc("Puts", "Write string to file (overwrite)", &["content", "path"]);
    ev.set_doc("PutsAppend", "Append string to file", &["content", "path"]);
    ev.set_doc("Gets", "Read entire stdin or file as string", &["path?"]);
    ev.set_doc("TermSize", "Current terminal width/height", &[]);
    ev.set_doc("AnsiStyle", "Style text with ANSI codes", &["text", "opts?"]);
    ev.set_doc("AnsiEnabled", "Are ANSI colors enabled?", &[]);
    ev.set_doc("StripAnsi", "Remove ANSI escape codes from string", &["text"]);
    ev.set_doc("AlignLeft", "Pad right to width", &["text", "width", "pad?"]);
    ev.set_doc("AlignRight", "Pad left to width", &["text", "width", "pad?"]);
    ev.set_doc("AlignCenter", "Pad on both sides to center", &["text", "width", "pad?"]);
    ev.set_doc("Truncate", "Truncate to width with ellipsis", &["text", "width", "ellipsis?"]);
    ev.set_doc("Wrap", "Wrap text to width", &["text", "width"]);
    ev.set_doc("TableSimple", "Render a simple table from rows", &["rows", "opts?"]);
    ev.set_doc("Rule", "Format a rule (k->v) as a string", &["k", "v"]);
    ev.set_doc("BoxText", "Draw a box around text", &["text", "opts?"]);
    ev.set_doc("Columnize", "Align lines in columns", &["lines", "opts?"]);
    ev.set_doc("PathExtname", "File extension without dot", &["path"]);
    ev.set_doc_examples("Puts", &["Puts[\"hello\", \"/tmp/hi.txt\"]  ==> True"]);
    ev.set_doc_examples("PutsAppend", &["PutsAppend[\"!\", \"/tmp/hi.txt\"]  ==> True"]);
    ev.set_doc_examples("Gets", &["Gets[\"/tmp/hi.txt\"]  ==> \"hello!\""]);
    ev.set_doc_examples("AnsiStyle", &["AnsiStyle[\"hi\", <|\"Color\"->\"green\", \"Bold\"->True|>]"]);
    ev.set_doc_examples("AlignCenter", &["AlignCenter[\"ok\", 6, \"-\"]  ==> \"--ok--\""]);
    ev.set_doc_examples("Truncate", &["Truncate[\"abcdef\", 4]  ==> \"abc…\""]);
    ev.set_doc_examples("Wrap", &["Wrap[\"aaaa bbbb cccc\", 5]  ==> \"aaaa\\nbbbb\\ncccc\""]);

    // Net examples
    ev.set_doc_examples("HttpGet", &["HttpGet[\"https://example.com\"]  ==> <|\"status\"->200, ...|>"]);
    ev.set_doc_examples(
        "Download",
        &["Download[\"https://example.com\", \"/tmp/index.html\"]  ==> \"/tmp/index.html\""]);
    ev.set_doc_examples(
        "HttpServe",
        &[
            "srv := HttpServe[(req) => RespondText[\"ok\"], <|\"Port\"->0|>]",
            "HttpServerAddr[srv]  ==> \"127.0.0.1:PORT\"",
            "HttpServerStop[srv]",
        ],
    );

    // Compression + Net docs sweep
    // Compression
    // Zip documented in a consolidated form above.
    ev.set_doc_examples(
        "Zip",
        &[
            "Zip[\"/tmp/bundle.zip\", {\"/tmp/a.txt\", \"/tmp/dir\"}]  ==> <|\"path\"->\"/tmp/bundle.zip\", ...|>",
        ],
    );
    ev.set_doc("ZipExtract", "Extract a .zip archive into a directory.", &["src", "dest"]);
    // Unzip documented in a consolidated form above.
    ev.set_doc_examples(
        "ZipExtract",
        &[
            "ZipExtract[\"/tmp/bundle.zip\", \"/tmp/unzipped\"]  ==> <|\"path\"->\"/tmp/unzipped\", \"files\"->...|>",
        ],
    );
    ev.set_doc("Tar", "Create a .tar (optionally .tar.gz) archive from inputs.", &["dest", "inputs", "opts?"]);
    ev.set_doc_examples(
        "Tar",
        &[
            "Tar[\"/tmp/bundle.tar\", {\"/tmp/data\"}]  ==> <|\"path\"->\"/tmp/bundle.tar\"|>",
            "Tar[\"/tmp/bundle.tar.gz\", {\"/tmp/data\"}, <|\"Gzip\"->True|>]  ==> <|\"path\"->...|>",
        ],
    );
    ev.set_doc("TarExtract", "Extract a .tar or .tar.gz archive into a directory.", &["src", "dest"]);
    ev.set_doc("Untar", "Extract a .tar or .tar.gz archive into a directory.", &["src", "dest"]);
    ev.set_doc_examples(
        "TarExtract",
        &[
            "TarExtract[\"/tmp/bundle.tar\", \"/tmp/untar\"]  ==> <|\"path\"->\"/tmp/untar\"|>",
        ],
    );
    ev.set_doc("Gzip", "Gzip-compress a string or a file; optionally write to path.", &["dataOrPath", "opts?"]);
    ev.set_doc_examples(
        "Gzip",
        &[
            "Gzip[\"hello\"]  ==> <compressed bytes as string>",
            "Gzip[\"/tmp/a.txt\", <|\"Out\"->\"/tmp/a.txt.gz\"|>]  ==> <|\"path\"->\"/tmp/a.txt.gz\", \"bytes_written\"->...|>",
        ],
    );
    ev.set_doc("Gunzip", "Gunzip-decompress a string or a .gz file; optionally write to path.", &["dataOrPath", "opts?"]);
    ev.set_doc_examples(
        "Gunzip",
        &[
            "Gunzip[Gzip[\"hello\"]]  ==> \"hello\"",
            "Gunzip[\"/tmp/a.txt.gz\", <|\"Out\"->\"/tmp/a.txt\"|>]  ==> <|\"path\"->\"/tmp/a.txt\", \"bytes_written\"->...|>",
        ],
    );

    // Net (streaming + caching + retry)
    ev.set_doc(
        "HttpStreamRequest",
        "Start a streaming HTTP request; returns headers, status, and a stream handle.",
        &["method", "url", "opts?"],
    );
    ev.set_doc_examples(
        "HttpStreamRequest",
        &[
            "r := HttpStreamRequest[\"GET\", \"https://example.com/large.bin\"]",
            "While[! Part[r2, \"done\"], r2 := HttpStreamRead[Part[r, \"stream\"], 8192]]",
            "HttpStreamClose[Part[r, \"stream\"]]",
        ],
    );
    ev.set_doc(
        "HttpStreamRead",
        "Read a chunk from a streaming HTTP handle; returns <|\"chunk\"->bytes, \"done\"->bool|>.",
        &["streamId", "maxBytes?"],
    );
    ev.set_doc_examples(
        "HttpStreamRead",
        &[
            "HttpStreamRead[handle, 16384]  ==> <|\"chunk\"->{...}, \"done\"->False|>",
        ],
    );
    ev.set_doc("HttpStreamClose", "Close a streaming HTTP handle.", &["streamId"]);
    ev.set_doc_examples("HttpStreamClose", &["HttpStreamClose[handle]  ==> True"]);

    ev.set_doc(
        "HttpDownloadCached",
        "Download a URL to a file with ETag/TTL caching; returns path and bytes written.",
        &["url", "path", "opts?"],
    );
    ev.set_doc_examples(
        "HttpDownloadCached",
        &[
            "HttpDownloadCached[\"https://httpbin.org/get\", \"/tmp/get.json\", <|ttlMs->60000|>]  ==> <|\"path\"->..., \"from_cache\"->True|>",
        ],
    );
    ev.set_doc(
        "HttpRetry",
        "Retry a generic HTTP request map until success or attempts exhausted.",
        &["request", "opts?"],
    );
    ev.set_doc_examples(
        "HttpRetry",
        &[
            "HttpRetry[<|\"Method\"->\"GET\", \"Url\"->\"https://example.com\"|>, <|\"Attempts\"->3, \"BackoffMs\"->200|>]  ==> <|\"status\"->200,...|>",
        ],
    );

    // Process examples
    ev.set_doc_examples("Run", &["Run[\"echo\", {\"hi\"}]  ==> <|\"Status\"->0, \"Stdout\"->\"hi\\n\",...|>"]);
    ev.set_doc_examples("Which", &["Which[\"sh\"]  ==> \"/bin/sh\""]);
    ev.set_doc_examples(
        "Pipe",
        &["Pipe[{{\"echo\",\"hello\"},{\"wc\",\"-c\"}}]  ==> <|\"Stdout\"->\"6\\n\"|>"]);

    // Time examples
    ev.set_doc_examples("NowMs", &["NowMs[]  ==> 1710000000000"]);
    ev.set_doc_examples("Sleep", &["Sleep[100]  ==> Null"]);
    ev.set_doc_examples(
        "DateTime",
        &["DateTime[\"2024-08-01T00:00:00Z\"]  ==> <|\"epochMs\"->...|>"]);
    ev.set_doc_examples(
        "DateFormat",
        &["DateFormat[DateTime[<|\"Year\"->2024,\"Month\"->8,\"Day\"->1|>], \"%Y-%m-%d\"]  ==> \"2024-08-01\""]);
    ev.set_doc_examples("DurationParse", &["DurationParse[\"2h30m\"]  ==> <|...|>"]);

    // Logging examples
    ev.set_doc_examples("ConfigureLogging", &["ConfigureLogging[<|\"Level\"->\"debug\"|>]  ==> True"]);
    ev.set_doc_examples("LogMessage", &["LogMessage[\"info\", \"service started\", <|\"port\"->8080|>]  ==> True"]);
    ev.set_doc_examples(
        "WithLogger",
        &["WithLogger[<|\"requestId\"->\"abc\"|>, LogMessage[\"info\", \"ok\"]]  ==> True"],
    );

    // Image examples
    ev.set_doc_examples(
        "ImageCanvas",
        &["png := ImageCanvas[<|\"Width\"->64, \"Height\"->64, \"Background\"->\"#ff0000\"|>]"]);
    ev.set_doc_examples(
        "ImageResize",
        &["out := ImageResize[png, <|\"Width\"->32, \"Height\"->32|>]  ==> base64url"]);
    ev.set_doc_examples("ImageInfo", &["ImageInfo[png]  ==> <|\"width\"->64, \"height\"->64|>"]);

    // Audio examples
    ev.set_doc_examples(
        "AudioConvert",
        &["wav := AudioConvert[<|\"Path\"->\"ding.mp3\"|>, \"wav\"]  ==> base64url"]);
    ev.set_doc_examples("AudioInfo", &["AudioInfo[wav]  ==> <|\"sampleRate\"->..., \"channels\"->...|>"]);

    // String and text utilities
    ev.set_doc("StringQ", "Is value a string?", &["x"]);
    ev.set_doc("StringChars", "Split string into list of characters", &["s"]);
    ev.set_doc("StringFromChars", "Join list of characters into string", &["chars"]);
    ev.set_doc("StringContains", "Does string contain substring?", &["s", "substr"]);
    ev.set_doc("StringJoinWith", "Join strings with a separator", &["parts", "sep"]);
    ev.set_doc("StringPadLeft", "Pad left to width with char", &["s", "width", "pad?"]);
    ev.set_doc("StringPadRight", "Pad right to width with char", &["s", "width", "pad?"]);
    ev.set_doc("StringRepeat", "Repeat string n times", &["s", "n"]);
    ev.set_doc("StringReplace", "Replace all substring matches", &["s", "from", "to"]);
    ev.set_doc("StringReplaceFirst", "Replace first substring match", &["s", "from", "to"]);
    ev.set_doc("StringReverse", "Reverse characters in a string", &["s"]);
    ev.set_doc("StringSlice", "Slice by start and optional length", &["s", "start", "len?"]);
    ev.set_doc("Split", "Split string by separator", &["s", "sep"]);
    ev.set_doc("StringTrim", "Trim whitespace from both ends", &["s"]);
    ev.set_doc("StringTrimLeft", "Trim from left", &["s"]);
    ev.set_doc("StringTrimRight", "Trim from right", &["s"]);
    ev.set_doc("StringTrimPrefix", "Remove prefix if present", &["s", "prefix"]);
    ev.set_doc("StringTrimSuffix", "Remove suffix if present", &["s", "suffix"]);
    ev.set_doc("StringTrimChars", "Trim characters from ends", &["s", "chars"]);
    ev.set_doc("StringTruncate", "Truncate string to length", &["s", "len", "ellipsis?"]);
    ev.set_doc("StringFormat", "Format using placeholders: {0}, {name}", &["fmt", "args"]);
    ev.set_doc("StringFormatMap", "Format using map placeholders", &["fmt", "map"]);
    ev.set_doc("StringInterpolate", "Interpolate ${var} from env or map", &["fmt", "map?"]);
    ev.set_doc("StringInterpolateWith", "Interpolate with custom resolver", &["fmt", "resolver"]);
    ev.set_doc("JoinLines", "Join list into lines with \n", &["lines"]);
    ev.set_doc("SplitLines", "Split string on \n into lines", &["s"]);
    ev.set_doc("CamelCase", "Convert to camelCase", &["s"]);
    ev.set_doc("SnakeCase", "Convert to snake_case", &["s"]);
    ev.set_doc("KebabCase", "Convert to kebab-case", &["s"]);
    ev.set_doc("Slugify", "Slugify for URLs", &["s"]);
    ev.set_doc("TitleCase", "Convert to Title Case", &["s"]);
    ev.set_doc("Capitalize", "Capitalize first letter", &["s"]);
    ev.set_doc("EqualsIgnoreCase", "Case-insensitive string equality", &["a", "b"]);
    ev.set_doc_examples("StringChars", &["StringChars[\"abc\"]  ==> {\"a\",\"b\",\"c\"}"]);
    ev.set_doc_examples("StringFromChars", &["StringFromChars[{\"a\",\"b\"}]  ==> \"ab\""]);
    ev.set_doc_examples("StringContains", &["StringContains[\"hello\", \"ell\"]  ==> True"]);
    ev.set_doc_examples("StringJoinWith", &["StringJoinWith[{\"a\",\"b\"}, \"-\"]  ==> \"a-b\""]);
    ev.set_doc_examples("StringPadLeft", &["StringPadLeft[\"7\", 3, \"0\"]  ==> \"007\""]);
    ev.set_doc_examples("Replace", &["Replace[\"foo bar\", \"o\", \"0\"]  ==> \"f00 bar\""]);
    ev.set_doc_examples("StringReverse", &["StringReverse[\"abc\"]  ==> \"cba\""]);
    ev.set_doc_examples("Split", &["Split[\"a,b,c\", \",\"]  ==> {\"a\",\"b\",\"c\"}"]);
    ev.set_doc_examples("Intersection", &["Intersection[{1,2,3},{2,4}]  ==> {2}"]);
    ev.set_doc_examples("Difference", &["Difference[{1,2,3},{2}]  ==> {1,3}"]);
    ev.set_doc_examples("StringTrim", &["StringTrim[\"  hi  \"]  ==> \"hi\""]);
    ev.set_doc_examples("CamelCase", &["CamelCase[\"hello world\"]  ==> \"helloWorld\""]);
    ev.set_doc_examples("SnakeCase", &["SnakeCase[\"HelloWorld\"]  ==> \"hello_world\""]);
    ev.set_doc_examples("KebabCase", &["KebabCase[\"HelloWorld\"]  ==> \"hello-world\""]);
    ev.set_doc_examples("Slugify", &["Slugify[\"Hello, World!\"]  ==> \"hello-world\""]);
    ev.set_doc_examples("TitleCase", &["TitleCase[\"hello world\"]  ==> \"Hello World\""]);
    // Additional string/search helpers
    ev.set_doc("StartsWith", "True if string starts with prefix", &["s", "prefix"]);
    ev.set_doc("EndsWith", "True if string ends with suffix", &["s", "suffix"]);
    ev.set_doc("IndexOf", "Index of substring (0-based; -1 if not found)", &["s", "substr", "from?"]);
    ev.set_doc(
        "LastIndexOf",
        "Last index of substring (0-based; -1 if not found)",
        &["s", "substr", "from?"],
    );
    ev.set_doc("IsBlank", "True if string is empty or whitespace", &["s"]);
    ev.set_doc("TemplateRender", "Render Mustache-like template with assoc data.", &["template", "data", "opts?"]);
    ev.set_doc("HtmlTemplate", "Render HTML/XML templates with Mustache semantics (sections, inverted, partials, comments, indented partials, standalone trimming; unescaped via {{{...}}} or {{& name}}). Options: Mode(html|xml), Strict, Whitespace(preserve|trim-tags|smart), Partials, Components, Layout, Loader.", &["templateOrPath", "data", "opts?"]);
    ev.set_doc("HtmlTemplateCompile", "Precompile HTML template (returns handle)", &["templateOrPath", "opts?"]);
    ev.set_doc("HtmlTemplateRender", "Render compiled HTML template with data", &["handle", "data", "opts?"]);
    ev.set_doc("HtmlAttr", "Escape string for HTML attribute context", &["s"]);
    ev.set_doc("SafeHtml", "Mark string as safe HTML (no escaping)", &["s"]);
    ev.set_doc_examples("StartsWith", &["StartsWith[\"foobar\", \"foo\"]  ==> True"]);
    ev.set_doc_examples("EndsWith", &["EndsWith[\"foobar\", \"bar\"]  ==> True"]);
    ev.set_doc_examples("IndexOf", &["IndexOf[\"banana\", \"na\"]  ==> 2", "IndexOf[\"banana\", \"x\"]  ==> -1"]);
    ev.set_doc_examples("LastIndexOf", &["LastIndexOf[\"banana\", \"na\"]  ==> 4"]);
    ev.set_doc_examples(
        "IsBlank",
        &["IsBlank[\"   \"]  ==> True", "IsBlank[\"a\"]  ==> False"],
    );
    ev.set_doc_examples(
        "TemplateRender",
        &[
            "TemplateRender[\"Hello {{name}}!\", <|\"name\"->\"Lyra\"|>]  ==> \"Hello Lyra!\"",
        ],
    );
    ev.set_doc_examples(
        "HtmlTemplate",
        &[
            // basic interpolation
            "HtmlTemplate[\"<b>{{name}}</b>\", <|name->\"X\"|>]  ==> \"<b>X</b>\"",
            // partial with props
            "HtmlTemplate[\"{{> Header <|title->\\\"Docs\\\"|>}}\", <||>, <|Partials-><|\"Header\"->\"<header>{{title}}</header>\"|>|>]  ==> \"<header>Docs</header>\"",
            // component with props + UrlEncode filter
            "HtmlTemplate[\"{{< Button <|label->\\\"Go\\\", href->\\\"/a?b=1&c=2\\\"|>}}\", <||>, <|Components-><|\"Button\"->\"<a href=\\\"{{href|UrlEncode}}\\\" class=\\\"btn\\\">{{label}}</a>\"|>|>]  ==> \"<a href=\\\"/a%3Fb%3D1%26c%3D2\\\" class=\\\"btn\\\">Go</a>\"",
            // layout + blocks + yield
            "HtmlTemplate[\"{{#block \\\"content\\\"}}<p>Hello</p>{{/block}}\", <|title->\"Home\"|>, <|Layout->\"<html><head><title>{{title}}</title></head><body>{{yield \\\"content\\\"}}</body></html>\"|>]  ==> \"<html><head><title>Home</title></head><body><p>Hello</p></body></html>\"",
            // SafeHtml bypasses escaping
            "HtmlTemplate[\"<div>{{{bio}}}</div>\", <|bio->SafeHtml[\"<em>writer</em>\"]|>]  ==> \"<div><em>writer</em></div>\"",
            // XML mode uses &apos; by default
            "HtmlTemplate[\"<note>{{text}}</note>\", <|text->\"O'Reilly\"|>, <|Mode->\"xml\"|>]  ==> \"<note>O&apos;Reilly</note>\"",
        ],
    );
    ev.set_doc_examples(
        "HtmlTemplateRender",
        &[
            "t := HtmlTemplateCompile[\"<i>{{msg}}</i>\"]; HtmlTemplateRender[t, <|msg->\"hi\"|>]  ==> \"<i>hi</i>\"",
        ],
    );
    ev.set_doc_examples(
        "HtmlAttr",
        &[
            "HtmlAttr[\"a&b\"]  ==> \"a&amp;b\"",
        ],
    );
    ev.set_doc_examples(
        "SafeHtml",
        &[
            "SafeHtml[\"<strong>x</strong>\"]  ==> <|__type->\"SafeHtml\"|>",
        ],
    );

    // Regex utilities
    ev.set_doc("RegexIsMatch", "Test if regex matches string", &["s", "pattern"]);
    ev.set_doc("RegexMatch", "Return first regex match", &["s", "pattern"]);
    ev.set_doc("RegexFind", "Find first regex capture groups", &["s", "pattern"]);
    ev.set_doc("RegexFindAll", "Find all regex capture groups", &["s", "pattern"]);
    ev.set_doc("RegexReplace", "Replace matches using regex", &["s", "pattern", "repl"]);
    ev.set_doc_examples("RegexIsMatch", &[r#"RegexIsMatch[\"abc123\", \"\\d+\"]  ==> True"#]);
    ev.set_doc_examples("RegexFindAll", &[r#"RegexFindAll[\"a1 b22\", \"\\d+\"]  ==> {\"1\",\"22\"}"#]);

    // URL utils
    ev.set_doc("UrlEncode", "Percent-encode string for URLs", &["s"]);
    ev.set_doc("UrlDecode", "Decode percent-encoded string", &["s"]);
    ev.set_doc("UrlFormEncode", "application/x-www-form-urlencoded from assoc", &["params"]);
    ev.set_doc("UrlFormDecode", "Parse form-encoded string to assoc", &["s"]);
    ev.set_doc_examples("UrlEncode", &["UrlEncode[\"a b\"]  ==> \"a%20b\""]);
    ev.set_doc_examples("UrlFormEncode", &["UrlFormEncode[<|\"a\"->\"b\"|>]  ==> \"a=b\""]);

    // Math and stats
    ev.set_doc("Sin", "Sine (radians). Tensor-aware: elementwise on tensors.", &["x"]);
    ev.set_doc("Cos", "Cosine (radians). Tensor-aware: elementwise on tensors.", &["x"]);
    ev.set_doc_examples("Sin", &["Sin[0]  ==> 0", "Sin[Tensor[{0, Pi/2}]]  ==> Tensor[...]"]);
    ev.set_doc_examples("Cos", &["Cos[0]  ==> 1", "Cos[Tensor[{0, Pi}]]  ==> Tensor[...]"]);
    ev.set_doc("Tan", "Tangent (radians)", &["x"]);
    ev.set_doc("Exp", "Natural exponential e^x. Tensor-aware: elementwise on tensors.", &["x"]);
    ev.set_doc("Sqrt", "Square root. Tensor-aware: elementwise on tensors.", &["x"]);
    ev.set_doc_examples("Exp", &["Exp[1]  ==> 2.71828...", "Exp[Tensor[{0,1}]]  ==> Tensor[...]"]);
    ev.set_doc_examples("Sqrt", &["Sqrt[9]  ==> 3", "Sqrt[Tensor[{1,4,9}]]  ==> Tensor[...]"]);
    ev.set_doc("ToDegrees", "Convert radians to degrees (Listable)", &["x"]);
    ev.set_doc("ToRadians", "Convert degrees to radians (Listable)", &["x"]);
    ev.set_doc_examples("ToDegrees", &["ToDegrees[Pi]  ==> 180"]);
    ev.set_doc_examples("ToRadians", &["ToRadians[180]  ==> 3.14159..."]);
    ev.set_doc("Log", "Natural logarithm. Tensor-aware: elementwise on tensors.", &["x"]);
    ev.set_doc_examples("Log", &["Log[E]  ==> 1", "Log[Tensor[{1,E}]]  ==> Tensor[...]"]);
    ev.set_doc("Signum", "Sign of number (-1,0,1)", &["x"]);
    ev.set_doc("Mod", "Modulo remainder ((a mod n) >= 0)", &["a", "n"]);
    ev.set_doc("DivMod", "Quotient and remainder", &["a", "n"]);
    ev.set_doc("Quotient", "Integer division quotient", &["a", "n"]);
    ev.set_doc("Remainder", "Integer division remainder", &["a", "n"]);
    ev.set_doc("Round", "Round to nearest integer", &["x"]);
    ev.set_doc("Floor", "Largest integer <= x", &["x"]);
    ev.set_doc("Ceiling", "Smallest integer >= x", &["x"]);
    ev.set_doc("GCD", "Greatest common divisor", &["a", "b", "…"]);
    ev.set_doc("LCM", "Least common multiple", &["a", "b", "…"]);
    ev.set_doc("Factorial", "n! (product 1..n)", &["n"]);
    ev.set_doc("Binomial", "Binomial coefficient nCk", &["n", "k"]);
    ev.set_doc("Mean", "Arithmetic mean of list", &["list"]);
    ev.set_doc("Median", "Median of list", &["list"]);
    ev.set_doc("StandardDeviation", "Standard deviation of list", &["list"]);
    ev.set_doc("Variance", "Variance of list", &["list"]);
    ev.set_doc("Quantile", "Quantile(s) of numeric data using R-7 interpolation.", &["data", "q|list"]);
    ev.set_doc("Percentile", "Percentile(s) of numeric data using R-7 interpolation.", &["data", "p|list"]);
    ev.set_doc("Mode", "Most frequent element (ties broken by first appearance).", &["data"]);
    ev.set_doc("Correlation", "Pearson correlation of two numeric lists (population moments).", &["a", "b"]);
    ev.set_doc("DescriptiveStats", "Summary stats: count, sum, mean, stddev, min/max, quartiles.", &["list"]);
    ev.set_doc("Quantiles", "Convenience wrapper for Quantile: return percentiles or by list.", &["list","qs?"]);
    ev.set_doc("RollingStats", "Rolling window stats over list (sum/mean/stddev/min/max).", &["list","window"]);
    ev.set_doc("RandomSample", "Sample k distinct elements (opts: seed).", &["list","k","opts?"]);
    ev.set_doc("Covariance", "Covariance of two numeric lists (population).", &["a", "b"]);
    ev.set_doc("Skewness", "Skewness (third standardized moment).", &["data"]);
    ev.set_doc("Kurtosis", "Kurtosis (fourth standardized moment).", &["data"]);
    ev.set_doc_examples("Quantile", &["Quantile[{1,2,3,4}, 0.25]  ==> 1.75", "Quantile[{1,2,3,4}, {0.25,0.5}]  ==> {1.75, 2.5}"]);
    ev.set_doc_examples("Percentile", &["Percentile[{1,2,3,4}, 25]  ==> 1.75"]);
    ev.set_doc_examples("Mode", &["Mode[{1,2,2,3}]  ==> 2"]);
    ev.set_doc_examples("Correlation", &["Correlation[{1,2,3},{2,4,6}]  ==> 1.0"]);
    ev.set_doc_examples("Covariance", &["Covariance[{1,2,3},{2,4,6}]  ==> 2.0"]);
    ev.set_doc_examples("Mod", &["Mod[7, 3]  ==> 1"]);
    ev.set_doc_examples("DivMod", &["DivMod[7, 3]  ==> {2, 1}"]);
    ev.set_doc_examples("Round", &["Round[2.6]  ==> 3"]);
    ev.set_doc_examples("GCD", &["GCD[18, 24]  ==> 6"]);
    ev.set_doc_examples("LCM", &["LCM[6, 8]  ==> 24"]);
    ev.set_doc_examples("Factorial", &["Factorial[5]  ==> 120"]);
    ev.set_doc_examples("Binomial", &["Binomial[5, 2]  ==> 10"]);

    // More math
    ev.set_doc("ASin", "Arc-sine (inverse sine)", &["x"]);
    ev.set_doc("ACos", "Arc-cosine (inverse cosine)", &["x"]);
    ev.set_doc("ATan", "Arc-tangent (inverse tangent)", &["x"]);
    ev.set_doc("ATan2", "Arc-tangent of y/x (quadrant aware)", &["y", "x"]);
    ev.set_doc("Clip", "Clamp value to [min,max]. Tensor-aware: elementwise on tensors.", &["x", "min", "max"]);
    ev.set_doc("Coalesce", "First non-null value", &["values…"]);
    ev.set_doc_examples("Clip", &["Clip[10, 0, 5]  ==> 5", "Clip[Tensor[{-1,2,7}], 0, 5]  ==> Tensor[...]"]);
    ev.set_doc_examples("Coalesce", &["Coalesce[Null, 0, 42]  ==> 0"]);

    // Collections: sets, stacks, queues, bags, priority queues
    ev.set_doc("HashSet", "Create a set from values", &["values"]);
    ev.set_doc("SetInsert", "Insert value into set", &["set", "value"]);
    ev.set_doc("SetRemove", "Remove value from set", &["set", "value"]);
    ev.set_doc("SetMemberQ", "Is value a member of set?", &["set", "value"]);
    ev.set_doc("SetToList", "Convert set to list", &["set"]);
    ev.set_doc("SetFromList", "Create set from list", &["list"]);
    // Generic set/list ops are documented under Union/Intersection/Difference
    ev.set_doc("SetEqualQ", "Are two sets equal?", &["a", "b"]);
    ev.set_doc("SetSubsetQ", "Is a subset of b?", &["a", "b"]);
    ev.set_doc("Stack", "Create a stack", &[]);
    // Size/emptiness use generic Length/EmptyQ
    ev.set_doc("Peek", "Peek top of stack/queue", &["handle"]);
    ev.set_doc("Push", "Push onto stack", &["stack", "value"]);
    ev.set_doc("Pop", "Pop from stack", &["stack"]);
    ev.set_doc("Queue", "Create a FIFO queue", &[]);
    // Size/emptiness use generic Length/EmptyQ
    ev.set_doc("Enqueue", "Enqueue value", &["queue", "value"]);
    ev.set_doc("Dequeue", "Dequeue value", &["queue"]);
    ev.set_doc("Bag", "Create a multiset bag", &[]);
    ev.set_doc("BagAdd", "Add item to bag", &["bag", "value"]);
    ev.set_doc("BagRemove", "Remove one item from bag", &["bag", "value"]);
    ev.set_doc("BagCount", "Count occurrences of value", &["bag", "value"]);
    ev.set_doc("BagSize", "Total items in bag", &["bag"]);
    ev.set_doc("BagUnion", "Union of two bags", &["a", "b"]);
    ev.set_doc("BagIntersection", "Intersection of two bags", &["a", "b"]);
    ev.set_doc("BagDifference", "Difference of two bags", &["a", "b"]);
    ev.set_doc("PriorityQueue", "Create a priority queue", &[]);
    ev.set_doc("PQInsert", "Insert with priority", &["pq", "priority", "value"]);
    ev.set_doc("PQPeek", "Peek min (or max) priority", &["pq"]);
    ev.set_doc("PQPop", "Pop min (or max) priority", &["pq"]);
    // Size/emptiness use generic Length/EmptyQ

    // Generic list/set utilities
    ev.set_doc("Union", "Union for lists (stable) or sets (dispatched)", &["args"]);
    ev.set_doc("Intersection", "Intersection for lists or sets (dispatched)", &["args"]);
    ev.set_doc("Difference", "Difference for lists or sets (dispatched)", &["a", "b"]);

    // Predicates (Q)
    ev.set_doc("BooleanQ", "Is value Boolean?", &["x"]);
    ev.set_doc("IntegerQ", "Is value an integer?", &["x"]);
    ev.set_doc("RealQ", "Is value a real number?", &["x"]);
    ev.set_doc("NumberQ", "Is value numeric (int/real)?", &["x"]);
    ev.set_doc("ListQ", "Is value a list?", &["x"]);
    ev.set_doc("AssocQ", "Is value an association (map)?", &["x"]);
    ev.set_doc("SymbolQ", "Is value a symbol?", &["x"]);
    ev.set_doc("PositiveQ", "Is number > 0?", &["x"]);
    ev.set_doc("NegativeQ", "Is number < 0?", &["x"]);
    ev.set_doc("NonNegativeQ", "Is number >= 0?", &["x"]);
    ev.set_doc("NonPositiveQ", "Is number <= 0?", &["x"]);
    ev.set_doc("NonEmptyQ", "Is list/string/assoc non-empty?", &["x"]);
    // EmptyQ documented earlier; avoid duplicate.

    // Counting
    // Count documented earlier; avoid duplicate.
    ev.set_doc_examples("Count", &["Count[{1,2,1,1}, 1]  ==> 3"]);

    // Date/time helpers
    ev.set_doc("DateDiff", "Difference between two DateTime in ms", &["a", "b"]);
    ev.set_doc("ParseDate", "Parse date string into DateTime", &["s"]);
    ev.set_doc("FormatDate", "Format DateTime with strftime pattern", &["dt", "fmt"]);

    // CLI helpers
    ev.set_doc("ArgsParse", "Parse CLI-like args to assoc", &["list"]);
    ev.set_doc("Prompt", "Prompt user for input (TTY)", &["text", "opts?"]);
    ev.set_doc("Confirm", "Ask yes/no question (TTY)", &["text", "opts?"]);
    ev.set_doc("PasswordPrompt", "Prompt for password without echo", &["text", "opts?"]);
    ev.set_doc("PromptSelect", "Prompt user to select one item from a list", &["text", "items", "opts?"]);
    ev.set_doc("PromptSelectMany", "Prompt user to select many items from a list", &["text", "items", "opts?"]);
    ev.set_doc("SetUiBackend", "Set UI backend: terminal | null | auto | gui (requires ui_egui)", &["mode"]);
    ev.set_doc("GetUiBackend", "Get current UI backend mode", &[]);
    ev.set_doc("SetUiTheme", "Set UI theme: system | light | dark. Optional opts: <|AccentColor->color, Rounding->px, FontSize->pt, Compact->True|False, SpacingScale->num, Palette-><|Primary->color, Success->color, Warning->color, Error->color, Info->color, Background->color, Surface->color, Text->color|>|>", &["mode", "opts?"]);
    ev.set_doc("GetUiTheme", "Get current UI theme", &[]);
    ev.set_doc("Notify", "Show a notification/message to the user", &["text", "opts?"]);
    ev.set_doc_examples("Notify", &[
        "Notify[\"Saved successfully\", <|Level->\"Success\"|>]",
        "Notify[\"Low disk space\", <|Level->\"Warning\", timeoutMs->5000|>]",
        "Notify[\"Build failed\", <|Level->\"Error\", Title->\"CI\"|>]",
        "Notify[\"Heads up\", <|Level->\"Info\", AccentColor->\"cyan\"|>]",
        "Notify[\"Tap anywhere to dismiss\", <|CloseOnClick->True, ShowDismiss->False|>]",
    ]);
    ev.set_doc("SpinnerStart", "Start an indeterminate spinner; returns id", &["text?", "opts?"]);
    ev.set_doc("SpinnerStop", "Stop a spinner by id", &["id"]);
    ev.set_doc("ProgressBar", "Create a progress bar; returns id.", &["total"]);
    ev.set_doc("ProgressAdvance", "Advance progress bar by n (default 1).", &["id", "n?"]);
    ev.set_doc("ProgressFinish", "Finish and remove a progress bar.", &["id"]);
    ev.set_doc_examples("Confirm", &["Confirm[\"Proceed?\"]  ==> True|False"]);
    ev.set_doc("PromptSelect", "Prompt user to select one item from a list.", &["text","items","opts?"]);
    ev.set_doc_examples("PromptSelect", &[
        "PromptSelect[\"Pick one\", {\"A\",\"B\",\"C\"}]  ==> \"B\"",
        "PromptSelect[\"Pick one\", {<|\"name\"->\"Human\", \"value\"->\"human\"|>, <|\"name\"->\"AI\", \"value\"->\"ai\"|>}]  ==> \"ai\"",
    ]);
    ev.set_doc_examples("PromptSelectMany", &[
        "PromptSelectMany[\"Pick some\", {\"A\",\"B\",\"C\"}]  ==> {\"A\",\"C\"}",
    ]);
    ev.set_doc_examples(
        "ProgressBar",
        &[
            "pb := ProgressBar[100]",
            "ProgressAdvance[pb, 10]  ==> True",
            "ProgressFinish[pb]  ==> True",
        ],
    );

    // JSON/HTML utils
    ev.set_doc("JsonEscape", "Escape string for JSON", &["s"]);
    ev.set_doc("JsonUnescape", "Unescape JSON-escaped string", &["s"]);
    ev.set_doc("HtmlEscape", "Escape string for HTML", &["s"]);
    ev.set_doc("HtmlUnescape", "Unescape HTML-escaped string", &["s"]);
    ev.set_doc("DotenvLoad", "Load .env variables into process env.", &["path?", "opts?"]);
    ev.set_doc("LoadDotenv", "Load .env with options (path, override).", &["opts?"]);
    ev.set_doc("ConfigFind", "Search upwards for config files (e.g., .env, lyra.toml).", &["names?", "startDir?"]);
    ev.set_doc("EnvExpand", "Expand $VAR or %VAR% style environment variables in text.", &["text", "opts?"]);
    ev.set_doc_examples("DotenvLoad", &["DotenvLoad[]  ==> <|\"path\"->\".../.env\", \"loaded\"->n|>"]);
    ev.set_doc_examples("LoadDotenv", &["LoadDotenv[<|path->\".env\", override->True|>]  ==> <|path->..., loaded->n|>"]);
    ev.set_doc_examples(
        "ConfigFind",
        &[
            "ConfigFind[\"lyra.toml\"]  ==> <|\"path\"->\".../lyra.toml\"|>",
        ],
    );
    ev.set_doc_examples(
        "EnvExpand",
        &[
            "EnvExpand[\"Hello $USER\"]  ==> \"Hello alice\"",
            "EnvExpand[\"%HOME%\\tmp\", <|\"Style\"->\"windows\"|>]  ==> \"/home/alice/tmp\"",
        ],
    );

    // HTTP helpers (server)
    ev.set_doc("Cors", "Build CORS middleware (wraps handler).", &["opts", "handler"]);
    ev.set_doc(
        "CorsApply",
        "Apply CORS preflight/headers using options and handler.",
        &["opts", "handler", "req"],
    );
    ev.set_doc_examples(
        "Cors",
        &[
            "srv := HttpServe[Cors[<|allowOrigin->\"*\"|>, (req)=>RespondText[\"ok\"]], <|port->0|>]",
            "HttpServerStop[srv]",
        ],
    );
    ev.set_doc_examples(
        "CorsApply",
        &[
            "CorsApply[<|allowOrigin->\"*\", allowMethods->\"GET\"|>, (r)=>RespondText[\"ok\"], <|\"method\"->\"OPTIONS\", \"headers\"-><||>|>]  ==> <|\"status\"->204, ...|>",
        ],
    );
    ev.set_doc("AuthJwt", "JWT auth middleware; verifies Bearer token and injects claims.", &["opts", "handler"]);
    ev.set_doc(
        "AuthJwtApply",
        "Verify JWT on request and call handler or return 401.",
        &["opts", "handler", "req"],
    );
    ev.set_doc_examples(
        "AuthJwtApply",
        &[
            "AuthJwtApply[<|secret->\"s\"|>, (r)=>RespondText[\"ok\"], <||>]  ==> <|\"status\"->401, ...|>",
            "tok := JwtSign[<|\"sub\"->\"u1\"|>, \"s\", <|alg->\"HS256\"|>]",
            "req := <|\"headers\"-><|\"Authorization\"->StringJoin[{\"Bearer \", tok}]|>|>",
            "AuthJwtApply[<|secret->\"s\"|>, (r)=>RespondText[\"ok\"], req]  ==> <|\"status\"->200, ...|>",
        ],
    );
    ev.set_doc("OpenApiGenerate", "Generate OpenAPI from routes", &["routes", "opts?"]);

    // Tracing
    ev.set_doc("Span", "Start a trace span and return its id.", &["name", "opts?"]);
    ev.set_doc("SpanEnd", "End the last span or the given span id.", &["id?"]);
    ev.set_doc("TraceGet", "Return collected spans as a list of assoc.", &[]);
    ev.set_doc("TraceExport", "Export spans to a file (json).", &["format", "opts?"]);
    ev.set_doc_examples(
        "Span",
        &[
            "id := Span[\"work\", <|\"Attrs\"-><|\"module\"->\"demo\"|>|>]",
            "SpanEnd[id]  ==> True",
            "TraceGet[]  ==> {<|\"Name\"->\"work\", ...|>, ...}",
        ],
    );
    ev.set_doc_examples(
        "TraceExport",
        &[
            "Span[\"build\"]; SpanEnd[]; TraceExport[\"json\", <|\"Path\"->\"/tmp/spans.json\"|>]  ==> True",
        ],
    );

    // Text searching utilities
    ev.set_doc("TextFind", "Find regex matches across files or text.", &["input", "pattern", "opts?"]);
    ev.set_doc("TextCount", "Count regex matches per file and total.", &["input", "pattern", "opts?"]);
    ev.set_doc(
        "TextFilesWithMatch",
        "List files that contain the pattern.",
        &["input", "pattern", "opts?"],
    );
    ev.set_doc(
        "TextLines",
        "Return matching lines with positions for a pattern.",
        &["input", "pattern", "opts?"],
    );
    ev.set_doc(
        "TextReplace",
        "Replace pattern across files; supports dry-run and backups.",
        &["input", "pattern", "replacement", "opts?"],
    );
    ev.set_doc(
        "TextDetectEncoding",
        "Detect likely text encoding for files.",
        &["input"],
    );
    ev.set_doc(
        "TextSearch",
        "Search text via regex, fuzzy, or index engine.",
        &["input", "query", "opts?"],
    );
    ev.set_doc_examples(
        "TextFind",
        &["TextFind[\"hello world\", \"\\w+\"]  ==> <|\"matches\"->...|>"]);
    ev.set_doc_examples(
        "TextCount",
        &["TextCount[\"a b a\", \"a\"]  ==> <|\"total\"->2, ...|>"]);
    ev.set_doc_examples(
        "TextFilesWithMatch",
        &["TextFilesWithMatch[\"src\", \"TODO\"]  ==> <|\"files\"->{...}|>"]);
    ev.set_doc_examples(
        "TextLines",
        &["TextLines[\"a\nTODO b\", \"TODO\"]  ==> <|\"lines\"->{<|\"lineNumber\"->2,...|>}|>"]);
    ev.set_doc_examples(
        "TextReplace",
        &["TextReplace[\"src\", \"foo\", \"bar\", <|\"dryRun\"->True|>]  ==> <|...|>"]);
    ev.set_doc_examples(
        "TextDetectEncoding",
        &["TextDetectEncoding[{\"file1.txt\"}]  ==> <|\"files\"->{<|\"file\"->..., \"encoding\"->...|>}|>"]);
    ev.set_doc_examples(
        "TextSearch",
        &["TextSearch[\"hello\", \"hell\"]  ==> <|\"engine\"->\"fuzzy\", ...|>"]);

    // NLP primitives
    ev.set_doc("Tokenize", "Split text into tokens (word/char/regex)", &["input","opts?"]);
    ev.set_doc_examples("Tokenize", &[
        "Tokenize[\"The quick brown fox.\"]  ==> {\"the\",\"quick\",\"brown\",\"fox\"}",
    ]);
    ev.set_doc("SentenceSplit", "Split text into sentences (heuristic)", &["text","opts?"]);
    ev.set_doc_examples("SentenceSplit", &[
        "SentenceSplit[\"One. Two!\"]  ==> {\"One.\",\"Two!\"}",
    ]);
    ev.set_doc("NormalizeText", "Normalize casing, punctuation, whitespace", &["text","opts?"]);
    ev.set_doc_examples("NormalizeText", &[
        "NormalizeText[\"A  B\", <|\"whitespace\"->\"collapse\"|>]  ==> \"A B\"",
    ]);
    ev.set_doc("Stopwords", "Return stopword list for a language", &["language"]);
    ev.set_doc("RemoveStopwords", "Remove stopwords from tokens or text", &["input","opts?"]);
    ev.set_doc_examples("RemoveStopwords", &[
        "RemoveStopwords[{\"the\",\"fox\"}, <|\"language\"->\"en\"|>]  ==> {\"fox\"}",
    ]);
    ev.set_doc("Ngrams", "Generate n-grams from tokens or text", &["input","opts?"]);
    ev.set_doc_examples("Ngrams", &[
        "Ngrams[{\"a\",\"b\",\"c\"}, <|\"n\"->2|>]  ==> {\"a b\",\"b c\"}",
    ]);
    ev.set_doc("TokenStats", "Token counts and top terms", &["input","opts?"]);
    ev.set_doc("BuildVocab", "Build vocabulary mapping term->index", &["docs","opts?"]);
    ev.set_doc("BagOfWords", "Bag-of-words vector or assoc of term counts", &["input","opts?"]);
    ev.set_doc("TfIdf", "TF-IDF features for documents", &["docs","opts?"]);
    ev.set_doc_examples("TfIdf", &[
        "TfIdf[{\"one two two\", \"two three\"}]  ==> <|\"vocab\"->..., \"idf\"->..., \"matrix\"->...|>",
    ]);
    ev.set_doc("ChunkText", "Split large text into overlapping chunks", &["text","opts?"]);


    ev.set_doc("Stem", "Stem tokens or text (Porter for English)", &["input","opts?"]);

    ev.set_doc("Lemmatize", "Lemmatize tokens or text (rule-based English)", &["input","opts?"]);
    ev.set_doc_examples("Lemmatize", &[
        "Lemmatize[\"powerfully\"]  ==> {\"powerful\"}",
        "Lemmatize[{\"running\",\"studies\",\"cars\"}]  ==> {\"run\",\"study\",\"car\"}",
    ]);

    ev.set_doc_examples("Stem", &[
        "Stem[\"running\"]  ==> {\"run\"}",
    ]);

    // UUIDs
    ev.set_doc("UuidV4", "Generate a random UUID v4 string.", &[]);
    ev.set_doc("UuidV7", "Generate a time-ordered UUID v7 string.", &[]);
    ev.set_doc_examples("UuidV4", &["UuidV4[]  ==> \"xxxxxxxx-xxxx-4xxx-...\""]);
    ev.set_doc_examples("UuidV7", &["UuidV7[]  ==> \"xxxxxxxx-xxxx-7xxx-...\""]);
    ev.set_doc("TlsSelfSigned", "Generate self-signed TLS cert + private key (PEM).", &["opts?"]);
    ev.set_doc_examples(
        "TlsSelfSigned",
        &["TlsSelfSigned[<|hosts->{\"localhost\"}, subject-><|CN->\"localhost\"|>|>]  ==> <|certPem->\"-----BEGIN CERTIFICATE-----...\"|>"]
    );

    // List accessors and ops
    ev.set_doc("First", "First element of a list (or Null).", &["list"]);
    ev.set_doc("Last", "Last element of a list (or Null).", &["list"]);
    ev.set_doc("Rest", "All but the first element.", &["list"]);
    ev.set_doc("Init", "All but the last element.", &["list"]);
    ev.set_doc("MapThread", "Map function over zipped lists (zip-with).", &["f", "lists"]);
    ev.set_doc("ReplacePart", "Replace element at 1-based index or key.", &["subject", "indexOrKey", "value"]);
    ev.set_doc("MapAt", "Apply function at 1-based index or key.", &["f", "subject", "indexOrKey"]);
    ev.set_doc("StableKey", "Canonical stable key string for ordering/dedup.", &["x"]);
    ev.set_doc("MaxBy", "Element with maximal derived key.", &["f", "list"]);
    ev.set_doc("MinBy", "Element with minimal derived key.", &["f", "list"]);
    ev.set_doc("ArgMax", "1-based index of maximal key.", &["f", "list"]);
    ev.set_doc("ArgMin", "1-based index of minimal key.", &["f", "list"]);
    ev.set_doc("UniqueBy", "Stable dedupe by derived key.", &["f", "list"]);
    ev.set_doc_examples("First", &["First[{1,2,3}]  ==> 1"]);
    ev.set_doc_examples("Last", &["Last[{1,2,3}]  ==> 3"]);
    ev.set_doc_examples("Rest", &["Rest[{1,2,3}]  ==> {2,3}"]);
    ev.set_doc_examples("Init", &["Init[{1,2,3}]  ==> {1,2}"]);
    ev.set_doc_examples("MapThread", &["MapThread[Plus, {{1,2},{10,20}}]  ==> {11,22}"]);
    ev.set_doc_examples("ReplacePart", &["ReplacePart[{1,2,3}, 2, 9]  ==> {1,9,3}"]);
    ev.set_doc_examples("MapAt", &["MapAt[ToUpper, <|\"a\"->\"x\"|>, \"a\"]  ==> <|\"a\"->\"X\"|>"]);
    ev.set_doc_examples("StableKey", &["StableKey[<|a->1|>]  ==> \"6:<|a=>0:00000000000000000001|>\""]);
    ev.set_doc_examples("MaxBy", &["MaxBy[Length, {\"a\",\"bbb\",\"cc\"}]  ==> \"bbb\""]);
    ev.set_doc_examples("MinBy", &["MinBy[Length, {\"a\",\"bbb\",\"cc\"}]  ==> \"a\""]);
    ev.set_doc_examples("ArgMax", &["ArgMax[Identity, {2,10,5}]  ==> 2"]);
    ev.set_doc_examples("ArgMin", &["ArgMin[Identity, {2,10,5}]  ==> 1"]);
    ev.set_doc_examples("UniqueBy", &["UniqueBy[Length, {\"a\",\"bb\",\"c\",\"dd\"}]  ==> {\"a\",\"bb\"}"]);

    // Generic SortBy dispatcher
    ev.set_doc("SortBy", "Sort list by key or association by derived key.", &["f", "subject"]);
    ev.set_doc_examples("SortBy", &["SortBy[Length, {\"a\",\"bbb\",\"cc\"}]  ==> {\"a\",\"cc\",\"bbb\"}"]);

    // Random (math + list helpers)
    ev.set_doc("SeedRandom", "Seed deterministic RNG scoped to this evaluator.", &["seed?"]);
    ev.set_doc("RandomInteger", "Random integer; supports {min,max}.", &["spec?"]);
    ev.set_doc("RandomReal", "Random real; supports {min,max}.", &["spec?"]);
    ev.set_doc("RandomChoice", "Random element from a list.", &["list"]);
    ev.set_doc("Shuffle", "Shuffle list uniformly.", &["list"]);
    ev.set_doc("Sample", "Sample k distinct elements from a list.", &["list", "k"]);
    ev.set_doc_examples("SeedRandom", &["SeedRandom[1]  ==> True"]);
    ev.set_doc_examples("RandomInteger", &["SeedRandom[1]; RandomInteger[{1,3}]  ==> 2"]);
    ev.set_doc_examples("RandomReal", &["SeedRandom[1]; RandomReal[{0.0,1.0}]  ==> 0.3..."]);
    ev.set_doc_examples("RandomChoice", &["SeedRandom[1]; RandomChoice[{\"a\",\"b\",\"c\"}]"]);
    ev.set_doc_examples("Shuffle", &["SeedRandom[1]; Shuffle[{1,2,3}]  ==> {3,1,2}"]);
    ev.set_doc_examples("Sample", &["SeedRandom[1]; Sample[{1,2,3,4}, 2]  ==> {3,1}"]);

    // Regex helpers
    ev.set_doc("RegexSplit", "Split string by regex pattern.", &["pattern", "s"]);
    ev.set_doc("RegexGroups", "Capture groups of first match.", &["pattern", "s"]);
    ev.set_doc("RegexCaptureNames", "Ordered list of named capture groups.", &["pattern"]);
    ev.set_doc_examples("RegexSplit", &["RegexSplit[\",\", \"a,b,c\"]  ==> {\"a\",\"b\",\"c\"}", "Split[\"a|b|c\", \"|\"]  ==> {\"a\",\"b\",\"c\"}"]);
    ev.set_doc_examples("RegexGroups", &["RegexGroups[\"(a)(b)\", \"ab\"]  ==> {\"a\",\"b\"}"]);
    ev.set_doc_examples("RegexCaptureNames", &["RegexCaptureNames[\"(?P<x>a)(?P<y>b)\"]  ==> {\"x\",\"y\"}"]);

    // Control flow
    ev.set_doc("While", "Repeat body while test evaluates to True.", &["test", "body"]);
    ev.set_doc("Do", "Execute body n times.", &["body", "n"]);
    ev.set_doc("For", "C-style loop with init/test/step.", &["init", "test", "step", "body"]);
    ev.set_doc_examples("While", &["i:=0; While[i<3, i:=i+1]; i  ==> 3"]);
    ev.set_doc_examples("Do", &["i:=0; Do[i:=i+1, 3]; i  ==> 3"]);
    ev.set_doc_examples("For", &["i:=0; For[i:=0, i<3, i:=i+1, Null]; i  ==> 3"]);

    // Part/Span helpers and packed arrays (Span docs defined earlier)
    ev.set_doc("PackedArray", "Create a packed numeric array.", &["list", "opts?"]);
    ev.set_doc("PackedToList", "Convert a packed array back to nested lists.", &["packed"]);
    ev.set_doc("PackedShape", "Return the shape of a packed array.", &["packed"]);
    // Packed* helpers are internal; prefer Tensor + Shape

    // Tools registry (agent tools)
    ev.set_doc("ToolsRegister", "Register one or more tool specs.", &["spec|list"]);
    ev.set_doc("ToolsUnregister", "Unregister a tool by id or name.", &["id|name"]);
    ev.set_doc("ToolsList", "List available tools as cards.", &[]);
    ev.set_doc("ToolsCards", "Paginate tool cards for external UIs.", &["cursor?", "limit?"]);
    ev.set_doc("ToolsDescribe", "Describe a tool by id or name.", &["id|name"]);
    ev.set_doc("ToolsSearch", "Search tools by name/summary.", &["query", "topK?"]);
    ev.set_doc("ToolsResolve", "Resolve tools matching a pattern.", &["pattern", "topK?"]);
    ev.set_doc("ToolsInvoke", "Invoke a tool with an args assoc.", &["id|name", "args?"]);
    ev.set_doc("ToolsExportOpenAI", "Export tools as OpenAI functions format.", &[]);
    ev.set_doc("ToolsExportBundle", "Export all registered tool specs.", &[]);
    ev.set_doc("ToolsSetCapabilities", "Set allowed capabilities (e.g., net, fs).", &["caps"]);
    ev.set_doc("ToolsGetCapabilities", "Get current capabilities list.", &[]);
    ev.set_doc("ToolsCacheClear", "Clear tool registry caches.", &[]);
    ev.set_doc("ToolsDryRun", "Validate a tool call and return normalized args and estimates.", &["id|name", "args"]);
    ev.set_doc("IdempotencyKey", "Generate a unique idempotency key.", &[]);
    ev.set_doc_examples(
        "ToolsRegister",
        &[
            "ToolsRegister[<|\"id\"->\"Hello\", \"summary\"->\"Say hi\", \"params\"->{\"name\"}|>]  ==> <|...|>",
            "ToolsList[]  ==> {...}",
        ],
    );
    ev.set_doc_examples(
        "ToolsInvoke",
        &[
            "ToolsInvoke[\"Hello\", <|\"name\"->\"Lyra\"|>]  ==> \"Hello, Lyra\"",
        ],
    );

    // Policy
    ev.set_doc("WithPolicy", "Evaluate body with temporary tool capabilities.", &["opts", "body"]);
    ev.set_doc_examples(
        "WithPolicy",
        &[
            "WithPolicy[<|\"Capabilities\"->{\"net\"}|>, HttpGet[\"https://example.com\"]]",
        ],
    );

    // Paths / env helpers
    ev.set_doc("XdgDirs", "Return XDG base directories (data, cache, config).", &[]);
    ev.set_doc("ResolveRelative", "Resolve a path relative to current file/module.", &["path"]);

    // Models / ML / NN
    ev.set_doc("Model", "Construct a model handle by id or spec.", &["id|spec"]);
    ev.set_doc("ModelsList", "List available model providers/ids.", &[]);
    ev.set_doc("NetGraph", "Construct a simple network graph from layers and edges.", &["nodes", "edges", "opts?"]);
    ev.set_doc("Chat", "Chat completion with messages; supports tools and streaming.", &["model?", "opts"]);
    ev.set_doc("Complete", "Text completion from prompt or options.", &["model?", "opts|prompt"]);
    ev.set_doc("Embed", "Compute embeddings for text using a provider.", &["opts"]);
    ev.set_doc("HybridSearch", "Combine keyword and vector search for retrieval.", &["store", "query", "opts?"]);
    // Metrics
    ev.set_doc("Metrics", "Return counters for tools/models/tokens/cost.", &[]);
    ev.set_doc("MetricsReset", "Reset metrics counters to zero.", &[]);
    ev.set_doc("CostAdd", "Add delta to accumulated USD cost; returns total.", &["amount"]);
    ev.set_doc("CostSoFar", "Return accumulated USD cost.", &[]);
    // RAG
    ev.set_doc("RAGChunk", "Split text into overlapping chunks for indexing.", &["text", "opts?"]);
    ev.set_doc("RAGIndex", "Embed and upsert documents into a vector store.", &["store", "docs", "opts?"]);
    ev.set_doc("RAGRetrieve", "Retrieve similar chunks for a query.", &["store", "query", "opts?"]);
    ev.set_doc("RAGAssembleContext", "Assemble a context string from matches.", &["matches", "opts?"]);
    ev.set_doc("RAGAnswer", "Answer a question using retrieved context and a model.", &["store", "query", "opts?"]);
    ev.set_doc("Cite", "Format citations from retrieval matches or answers.", &["matchesOrAnswer", "opts?"]);
    ev.set_doc("Citations", "Normalize citations from matches or answer.", &["matchesOrAnswer"]);

    // Collections and datasets
    ev.set_doc("Top", "Take top-k items (optionally by key).", &["list", "k", "opts?"]);
    ev.set_doc("UnionByPosition", "Union datasets by column position.", &["ds1", "ds2", "…"]);
    ev.set_doc("col", "Column accessor helper for Dataset expressions.", &["name"]);
    ev.set_doc("__DatasetFromDbTable", "Internal: create Dataset from DB table.", &["conn", "table"]);
    ev.set_doc("__SQLToRows", "Internal: run SQL and return rows.", &["conn", "sql", "params?"]);
    ev.set_doc("Cast", "Cast a value to a target type (string, integer, real, boolean).", &["value", "type"]);

    // Math extras
    ev.set_doc("Trunc", "Truncate toward zero (Listable).", &["x"]);
    ev.set_doc("NthRoot", "Principal nth root of a number.", &["x", "n"]);
    // Algebra
    ev.set_doc("Simplify", "Simplify algebraic expression.", &["expr"]);
    ev.set_doc("Expand", "Distribute products over sums once.", &["expr"]);
    ev.set_doc("ExpandAll", "Fully expand products over sums.", &["expr"]);
    ev.set_doc("CollectTerms", "Collect like terms in a sum.", &["expr"]);
    ev.set_doc("CollectTermsBy", "Collect terms by function or key.", &["expr", "by"]);
    ev.set_doc("Factor", "Factor a polynomial expression.", &["expr"]);
    ev.set_doc("D", "Differentiate expression w.r.t. variable.", &["expr", "var"]);
    ev.set_doc("Apart", "Partial fraction decomposition.", &["expr", "var?"]);
    ev.set_doc("Solve", "Solve equations for variables.", &["eqns", "vars?"]);
    ev.set_doc("Roots", "Polynomial roots for univariate polynomial.", &["poly", "var?"]);
    ev.set_doc("CancelRational", "Cancel common factors in a rational expression.", &["expr"]);

    // Memory/session
    ev.set_doc("Remember", "Append item to named session buffer.", &["session", "item"]);
    ev.set_doc("Recall", "Return recent items from session (with optional query).", &["session", "query?", "opts?"]);
    ev.set_doc("SessionClear", "Clear a named session buffer.", &["session"]);

    // Containers
    ev.set_doc("PingContainers", "Check if container engine is reachable.", &[]);
    ev.set_doc("RunContainer", "Run a container image; returns id or result.", &["spec", "opts?"]);
    ev.set_doc("ImageHistory", "Show history/metadata for an image.", &["ref"]);

    // Module and project helpers
    ev.set_doc("ModuleInfo", "Information about the current module (path, package).", &[]);
    ev.set_doc("Exported", "Mark symbols as exported from current module.", &["symbols"]);

    // ---------------- Git ----------------
    ev.set_doc("GitVersion", "Get git client version string", &[]);
    ev.set_doc("GitRoot", "Path to repository root (Null if absent)", &[]);
    ev.set_doc("GitInit", "Initialize a new git repository", &["opts?"]);
    ev.set_doc("GitStatus", "Status (porcelain) with branch/ahead/behind/changes", &["opts?"]);
    ev.set_doc("GitAdd", "Stage files for commit", &["paths", "opts?"]);
    ev.set_doc("GitCommit", "Create a commit with message", &["message", "opts?"]);
    ev.set_doc("GitCurrentBranch", "Current branch name", &[]);
    ev.set_doc("GitBranchList", "List local branches", &[]);
    ev.set_doc("GitBranch", "Create a new branch", &["name", "opts?"]);
    ev.set_doc("GitSwitch", "Switch to branch (optionally create)", &["name", "opts?"]);
    ev.set_doc("GitDiff", "Diff against base and optional paths", &["opts?"]);
    ev.set_doc("GitApply", "Apply a patch (or check only)", &["patch", "opts?"]);
    ev.set_doc("GitLog", "List commits with formatting options", &["opts?"]);
    ev.set_doc("GitRemoteList", "List remotes", &[]);
    ev.set_doc("GitFetch", "Fetch from remote", &["remote?"]);
    ev.set_doc("GitPull", "Pull from remote", &["remote?", "opts?"]);
    ev.set_doc("GitPush", "Push to remote", &["opts?"]);
    ev.set_doc("GitEnsureRepo", "Ensure Cwd is a git repo (init if needed)", &["opts?"]);
    ev.set_doc("GitStatusSummary", "Summarize status counts and branch", &["opts?"]);
    // Dev / formatting / linting / release helpers
    ev.set_doc("FormatLyraText", "Format Lyra source text (pretty printer).", &["text"]);
    ev.set_doc("FormatLyraFile", "Format a Lyra source file in place.", &["path"]);
    ev.set_doc("FormatLyra", "Format Lyra from text or file path.", &["x"]);
    ev.set_doc("LintLyraText", "Lint Lyra source text; returns diagnostics.", &["text"]);
    ev.set_doc("LintLyraFile", "Lint a Lyra source file; returns diagnostics.", &["path"]);
    ev.set_doc("LintLyra", "Lint Lyra from text or file path.", &["x"]);
    ev.set_doc("ConfigLoad", "Load project config and environment.", &["opts?"]);
    ev.set_doc("LoadConfig", "Load and merge config (files, envPrefix, overrides).", &["opts?"]);
    ev.set_doc("Env", "Read environment variables (keys, required, defaults).", &["opts?"]);
    ev.set_doc("SecretsGet", "Get secret by key from provider (Env or File).", &["key", "provider"]);
    ev.set_doc("GetSecret", "Get secret by key from provider (Env or File).", &["key", "provider"]);
    ev.set_doc("VersionBump", "Bump semver in files: major/minor/patch.", &["level", "paths"]);
    ev.set_doc("ChangelogGenerate", "Generate CHANGELOG entries from git log.", &["range?"]);
    ev.set_doc("ReleaseTag", "Create annotated git tag (and optionally push).", &["version", "opts?"]);

    // BDD helpers
    ev.set_doc("Describe", "Define a test suite (held).", &["name", "items", "opts?"]);
    ev.set_doc("It", "Define a test case (held).", &["name", "body", "opts?"]);
    ev.set_doc_examples(
        "Describe",
        &[
            "Describe[\"Math\", {It[\"adds\", 1+1==2]}]  ==> <|\"type\"->\"suite\"|>",
        ],
    );
    ev.set_doc("GitSmartCommit", "Stage + conventional commit (auto msg option)", &["opts?"]);
    ev.set_doc("GitFeatureBranch", "Create and switch to a feature branch", &["opts?"]);
    ev.set_doc("GitSyncUpstream", "Fetch, rebase (or merge), and push upstream", &["opts?"]);
    ev.set_doc_examples("GitVersion", &["GitVersion[]  ==> \"git version ...\""]);
    ev.set_doc_examples("GitRoot", &["GitRoot[]  ==> \"/path/to/repo\" | Null"]);
    ev.set_doc_examples("GitStatus", &["GitStatus[]  ==> <|Branch->..., Ahead->0, Behind->0, Changes->{...}|>"]);
    ev.set_doc_examples("GitAdd", &["GitAdd[\"src/main.rs\"]  ==> True"]);
    ev.set_doc_examples("GitCommit", &["GitCommit[\"feat: add api\"]  ==> <|Sha->..., Message->...|>"]);
    ev.set_doc_examples("GitBranch", &["GitBranch[\"feature/x\"]  ==> True"]);
    ev.set_doc_examples("GitSwitch", &["GitSwitch[\"feature/x\"]  ==> True"]);
    ev.set_doc_examples("GitDiff", &["GitDiff[<|\"Base\"->\"HEAD~1\"|>]  ==> \"diff...\""]);
    ev.set_doc_examples("GitLog", &["GitLog[<|\"Limit\"->5|>]  ==> {\"<sha>|<author>|...\", ...}"]);

    // ---------------- Package ----------------
    ev.set_doc("Using", "Load a package by name with import options", &["name", "opts?"]);
    ev.set_doc("Unuse", "Unload a package; hide imported symbols", &["name"]);
    ev.set_doc("ReloadPackage", "Reload a package", &["name"]);
    ev.set_doc("WithPackage", "Temporarily add a path to $PackagePath", &["name", "expr"]);
    ev.set_doc("BeginModule", "Start a module scope (record exports)", &["name"]);
    ev.set_doc("EndModule", "End current module scope", &[]);
    ev.set_doc("Export", "Mark symbol(s) as public", &["symbols"]);
    ev.set_doc("Private", "Mark symbol(s) as private", &["symbols"]);
    ev.set_doc("CurrentModule", "Current module path/name", &[]);
    ev.set_doc("ModulePath", "Get module search path", &[]);
    ev.set_doc("SetModulePath", "Set module search path", &["path"]);
    ev.set_doc("PackageInfo", "Read package metadata (name, version, path)", &["name"]);
    ev.set_doc("PackageVersion", "Read version from manifest", &["pkgPath"]);
    ev.set_doc("PackagePath", "Get current $PackagePath", &[]);
    ev.set_doc("ListInstalledPackages", "List packages available on $PackagePath", &[]);
    ev.set_doc("NewPackage", "Scaffold a new package directory", &["name", "opts?"]);
    ev.set_doc("NewModule", "Scaffold a new module file in a package", &["pkgPath", "name"]);
    ev.set_doc("ImportedSymbols", "Assoc of package -> imported symbols", &[]);
    ev.set_doc("LoadedPackages", "Assoc of loaded packages", &[]);
    ev.set_doc("RegisterExports", "Register exports for a package (internal)", &["name", "exports"]);
    ev.set_doc("PackageExports", "Get exports list for a package", &["name"]);
    // PM stubs
    ev.set_doc("BuildPackage", "Build a package (requires lyra-pm)", &["path?", "opts?"]);
    ev.set_doc("TestPackage", "Run package tests (requires lyra-pm)", &["path?", "opts?"]);
    ev.set_doc("LintPackage", "Lint a package (requires lyra-pm)", &["path?", "opts?"]);
    ev.set_doc("PackPackage", "Pack artifacts for distribution (requires lyra-pm)", &["path?", "opts?"]);
    ev.set_doc("GenerateSBOM", "Generate SBOM (requires lyra-pm)", &["path?", "opts?"]);
    ev.set_doc("SignPackage", "Sign package (requires lyra-pm)", &["path?", "opts?"]);
    ev.set_doc("PublishPackage", "Publish to registry (requires lyra-pm)", &["path?", "opts?"]);
    ev.set_doc("InstallPackage", "Install a package (requires lyra-pm)", &["name", "opts?"]);
    ev.set_doc("UpdatePackage", "Update a package (requires lyra-pm)", &["name", "opts?"]);
    ev.set_doc("RemovePackage", "Remove a package (requires lyra-pm)", &["name", "opts?"]);
    ev.set_doc("LoginRegistry", "Login to package registry (requires lyra-pm)", &["opts?"]);
    ev.set_doc("LogoutRegistry", "Logout from package registry (requires lyra-pm)", &["opts?"]);
    ev.set_doc("WhoAmI", "Show current registry identity (requires lyra-pm)", &[]);
    ev.set_doc("PackageAudit", "Audit dependencies (requires lyra-pm)", &["path?", "opts?"]);
    ev.set_doc("PackageVerify", "Verify signatures (requires lyra-pm)", &["path?", "opts?"]);
    ev.set_doc_examples("Using", &["Using[\"lyra/math\", <|\"Import\"->\"All\"|>]  ==> True"]);
    ev.set_doc_examples("BeginModule", &["BeginModule[\"Main\"]  ==> \"Main\""]);
    ev.set_doc_examples("Export", &["Export[{\"Foo\", \"Bar\"}]"]);
    ev.set_doc_examples("NewPackage", &["NewPackage[<|\"Name\"->\"acme.tools\", \"Path\"->\"./packages\"|>]  ==> <|path->...|>"]);
    ev.set_doc_examples("NewModule", &["NewModule[\"./packages/acme.tools\", \"Util\"]  ==> \".../src/Util.lyra\""]);

    // ---------------- Project ----------------
    ev.set_doc("ProjectDiscover", "Search upwards for project.lyra", &["start?"]);
    ev.set_doc("ProjectRoot", "Return project root path (or Null)", &[]);
    ev.set_doc("ProjectLoad", "Load and evaluate project.lyra (normalized)", &["root?"]);
    ev.set_doc("ProjectInfo", "Summarize project (name, version, paths)", &["root?"]);
    ev.set_doc("ProjectValidate", "Validate project manifest and structure", &["root?"]);
    ev.set_doc("ProjectInit", "Initialize new project (scaffold)", &["path", "opts?"]);
    ev.set_doc_examples("ProjectDiscover", &["ProjectDiscover[]  ==> \"/path/to/project\" | Null"]);
    ev.set_doc_examples("ProjectInfo", &["ProjectInfo[]  ==> <|name->..., version->..., modules->...|>"]);

    // ---------------- Workflow ----------------
    ev.set_doc("Workflow", "Run a list of steps sequentially (held)", &["steps"]);
    ev.set_doc_examples(
        "Workflow",
        &[
            "Workflow[{Print[\"a\"], Print[\"b\"]}]  ==> {Null, Null}",
            "Workflow[{<|\"name\"->\"echo\", \"run\"->Run[\"echo\", {\"hi\"}]|>}]  ==> {...}",
        ],
    );
    ev.set_doc("Task", "Define a task: <|name, run, dependsOn, when?, timeoutMs?, retries?|>", &["spec"]);
    ev.set_doc_examples(
        "Task",
        &[
            "Task[<|\"name\"->\"build\", \"run\"->\"build\"|>]  ==> <|name->..., run->..., dependsOn->{ }|>",
        ],
    );
    ev.set_doc("RunTasks", "Run tasks with deps; opts: maxConcurrency, onError, stream, hooks, onEvent", &["tasks","opts?"]);
    ev.set_doc_examples(
        "RunTasks",
        &[
            "b=Task[<|\"name\"->\"build\",\"run\"->\"build\"|>]; t=Task[<|\"name\"->\"test\",\"run\"->\"test\",\"dependsOn\"->{\"build\"}|>]; RunTasks[{b,t}, <|\"maxConcurrency\"->2|>]  ==> <|results->..., status->...|>",
            "RunTasks[{Task[<|\"name\"->\"a\",\"run\"->1|>]} , <|\"beforeAll\"->Set[x,1], \"afterAll\"->Set[y,1]|>]  ==> <|...|>",
            "RunTasks[{Task[<|\"name\"->\"a\",\"run\"->1|>]} , <|\"onEvent\"->(e)=>Print[e]|>]  ==> <|...|>",
        ],
    );
    ev.set_doc("ExplainTasks", "Explain task DAG: nodes, edges, topological order", &["tasks"]);
    ev.set_doc_examples(
        "ExplainTasks",
        &[
            "ExplainTasks[{Task[<|\"name\"->\"a\",\"run\"->1|>], Task[<|\"name\"->\"b\",\"run\"->2,\"dependsOn\"->{\"a\"}|>]}]  ==> <|\"order\"->{0,1}, ...|>",
        ],
    );
    // ---------------- Distributions ----------------
    ev.set_doc("Normal", "Normal distribution head (mean μ, stddev σ).", &["mu","sigma"]);
    ev.set_doc("Bernoulli", "Bernoulli distribution head (probability p).", &["p"]);
    ev.set_doc("BinomialDistribution", "Binomial distribution head (trials n, prob p).", &["n","p"]);
    ev.set_doc("Poisson", "Poisson distribution head (rate λ).", &["lambda"]);
    ev.set_doc("Exponential", "Exponential distribution head (rate λ).", &["lambda"]);
    ev.set_doc("Gamma", "Gamma distribution head (shape k, scale θ).", &["k","theta"]);
    ev.set_doc("PDF", "Probability density/mass for a distribution at x.", &["dist","x"]);
    ev.set_doc("CDF", "Cumulative distribution for a distribution at x.", &["dist","x"]);
    ev.set_doc("RandomVariate", "Sample from a distribution (optionally n samples).", &["dist","n?"]);
    ev.set_doc_examples("PDF", &["PDF[Normal[0,1], 0]  ==> 0.39894…", "PDF[BinomialDistribution[10, 0.5], 5]  ==> 0.24609375"]);
    ev.set_doc_examples("CDF", &["CDF[Exponential[2.0], 1.0]  ==> 1 - e^-2", "CDF[Poisson[2.0], 3]  ==> Σ_{k=0..3} e^-2 2^k/k!"]);
    ev.set_doc_examples("RandomVariate", &["RandomVariate[Normal[0,1], 3]  ==> {…}", "RandomVariate[Bernoulli[0.3], 5]  ==> {0,1,0,0,1}"]);

    // ---------------- Linalg ----------------
    ev.set_doc("Determinant", "Determinant of a square matrix (partial pivoting).", &["A"]);
    ev.set_doc("Inverse", "Inverse of a square matrix (Gauss–Jordan with pivoting).", &["A"]);
    ev.set_doc("LinearSolve", "Solve linear system A x = b (SPD via Cholesky; otherwise LU).", &["A","b"]);
    ev.set_doc("QR", "QR decomposition via Householder reflections. Use \"Reduced\" option for thin Q,R.", &["A","opts?"]);
    ev.set_doc("SVD", "Reduced singular value decomposition A = U S V^T (via AtA eigen).", &["A"]);
    ev.set_doc("EigenDecomposition", "Eigenvalues and eigenvectors. Symmetric: Jacobi; general: real QR + inverse iteration.", &["A"]);
    ev.set_doc("Dot", "Matrix multiplication and vector dot product (type-dispatched).", &["a","b"]);
    ev.set_doc("Transpose", "Transpose of a matrix (or NDTranspose for permutations).", &["A","perm?"]);
    ev.set_doc("Trace", "Trace of a square matrix (sum of diagonal).", &["A"]);
    ev.set_doc("Norm", "Vector p-norms; for matrices: Frobenius by default, 2-norm via SVD when p==2; also supports p=1 and Infinity.", &["x","p?"]);
    ev.set_doc("MatrixNorm", "Matrix p-norm: 2 (spectral via SVD), 1 (max column sum), Infinity (max row sum), \"Frobenius\" alias.", &["A","p"]);
    ev.set_doc("PseudoInverse", "Moore–Penrose pseudoinverse via reduced SVD (V S^+ U^T). Accepts Tolerance option or numeric.", &["A","optsOrTol?"]);
    ev.set_doc("Diagonal", "Main diagonal of a matrix as a vector.", &["A"]);
    ev.set_doc("DiagMatrix", "Diagonal matrix from a vector.", &["v"]);
    ev.set_doc("Rank", "Numerical matrix rank via reduced QR (tolerance-based).", &["A"]);
    ev.set_doc("ConditionNumber", "Estimated 2-norm condition number using power iterations on A^T A.", &["A"]);

    ev.set_doc_examples("QR", &[
        "r := QR[{{1,2},{3,4}}]  ==> <|Q->..., R->...|>",
        "r2 := QR[{{1,2,3},{4,5,6}}, \"Reduced\"]  ==> Q:(2x2), R:(2x3)",
    ]);
    ev.set_doc_examples("SVD", &[
        "sv := SVD[{{1,2},{3,4}}]  ==> <|U->(2x2), S->{..}, V->(2x2)|>",
        "Dot[sv[\"U\"], DiagMatrix[sv[\"S\"]], Transpose[sv[\"V\"]]]  ==> {{1,2},{3,4}}",
    ]);
    ev.set_doc_examples("EigenDecomposition", &[
        "EigenDecomposition[{{2,1},{1,2}}]  ==> <|Eigenvalues->{3,1}, Eigenvectors->...|>",
    ]);
    ev.set_doc("FFT", "Discrete Fourier transform of a 1D sequence (returns Complex list).", &["x","n?"]);
    ev.set_doc("IFFT", "Inverse DFT of a 1D sequence.", &["X"]);
    ev.set_doc("Convolve", "Linear convolution of two sequences. Modes: Full|Same|Valid.", &["a","b","mode?"]);
    ev.set_doc("Window", "Window weights vector by type and size (Hann|Hamming|Blackman).", &["type","n","opts?"]);
    ev.set_doc("STFT", "Short-time Fourier transform. STFT[x, size, hop?] or options.", &["x","size|opts","hop?"]);
    ev.set_doc_examples("FFT", &[
        "FFT[{1,0,0,0}]  ==> {1+0i, 1+0i, 1+0i, 1+0i}",
        "IFFT[%]  ==> {1,0,0,0}",
    ]);
    ev.set_doc_examples("Convolve", &[
        "Convolve[{1,2,1}, {1,1,1}]  ==> {1,3,4,3,1}",
        "Convolve[{1,2,1}, {1,1,1}, \"Same\"]  ==> {1,3,4}",
    ]);
    ev.set_doc_examples("Window", &[
        "Window[\"Hann\", 4]  ==> {0.0, 0.5, 0.5, 0.0}",
    ]);
    ev.set_doc_examples("STFT", &[
        "STFT[Range[0, 127], 64, 32]  ==> {{…}, {…}, …}",
    ]);
    ev.set_doc_examples("MatrixNorm", &[
        "MatrixNorm[{{1,2},{3,4}}, 2]  ==> largest singular value",
        "MatrixNorm[{{1,2},{-3,4}}, 1]  ==> max column sum",
        "MatrixNorm[{{1,2},{-3,4}}, Infinity]  ==> max row sum",
        "MatrixNorm[{{1,2},{3,4}}, \"Frobenius\"]  ==> same as Norm[A]",
    ]);
    ev.set_doc_examples("PseudoInverse", &[
        "PseudoInverse[{{1,2},{2,4}}]  ==> least-squares inverse (2x2 rank-1)",
        "PseudoInverse[A, <|\"Tolerance\"->1e-8|>]",
        "PseudoInverse[A, 1e-8]  (numeric shorthand)",
    ]);
    ev.set_doc("Determinant", "Determinant of a square matrix (partial pivoting).", &["A"]);
    ev.set_doc("Inverse", "Inverse of a square matrix (Gauss–Jordan with pivoting).", &["A"]);
    ev.set_doc("LinearSolve", "Solve linear system A x = b (SPD via Cholesky; otherwise LU).", &["A","b"]);
    ev.set_doc("QR", "QR decomposition via Householder reflections. Use \"Reduced\" option for thin Q,R.", &["A","opts?"]);
    ev.set_doc("SVD", "Reduced singular value decomposition A = U S V^T (via AtA eigen).", &["A"]);
    ev.set_doc("EigenDecomposition", "Eigenvalues and eigenvectors. Symmetric: Jacobi; general: real QR + inverse iteration.", &["A"]);
    ev.set_doc("Dot", "Matrix multiplication and vector dot product (type-dispatched).", &["a","b"]);
    ev.set_doc("Transpose", "Transpose of a matrix (or NDTranspose for permutations).", &["A","perm?"]);
    ev.set_doc("Trace", "Trace of a square matrix (sum of diagonal).", &["A"]);
    ev.set_doc("Norm", "Vector p-norms; for matrices: Frobenius by default, 2-norm via SVD when p==2.", &["x","p?"]);
    ev.set_doc("MatrixNorm", "Matrix p-norm: 2 (spectral via SVD), 1 (max column sum), Infinity (max row sum).", &["A","p"]);
    ev.set_doc("PseudoInverse", "Moore–Penrose pseudoinverse via reduced SVD (V S^+ U^T).", &["A"]);
    ev.set_doc("Diagonal", "Main diagonal of a matrix as a vector.", &["A"]);
    ev.set_doc("DiagMatrix", "Diagonal matrix from a vector.", &["v"]);
    ev.set_doc("Rank", "Numerical matrix rank via reduced QR (tolerance-based).", &["A"]);
    ev.set_doc("ConditionNumber", "Estimated 2-norm condition number using power iterations on A^T A.", &["A"]);

    ev.set_doc_examples("QR", &[
        "r := QR[{{1,2},{3,4}}]  ==> <|Q->..., R->...|>",
        "r2 := QR[{{1,2,3},{4,5,6}}, \"Reduced\"]  ==> Q:(2x2), R:(2x3)",
    ]);
    ev.set_doc_examples("SVD", &[
        "sv := SVD[{{1,2},{3,4}}]  ==> <|U->(2x2), S->{..}, V->(2x2)|>",
        "Dot[sv[\"U\"], DiagMatrix[sv[\"S\"]], Transpose[sv[\"V\"]]]  ==> {{1,2},{3,4}}",
    ]);
    ev.set_doc_examples("EigenDecomposition", &[
        "EigenDecomposition[{{2,1},{1,2}}]  ==> <|Eigenvalues->{3,1}, Eigenvectors->...|>",
    ]);
    ev.set_doc_examples("MatrixNorm", &[
        "MatrixNorm[{{1,2},{3,4}}, 2]  ==> largest singular value",
        "MatrixNorm[{{1,2},{-3,4}}, 1]  ==> max column sum",
        "MatrixNorm[{{1,2},{-3,4}}, Infinity]  ==> max row sum",
    ]);
    ev.set_doc_examples("PseudoInverse", &[
        "PseudoInverse[{{1,2},{2,4}}]  ==> least-squares inverse (2x2 rank-1)",
    ]);
}

// Internal glue and dispatchers: mark as internal so they don't count as missing
pub fn register_internal_docs(ev: &mut Evaluator) {
    // Database
    Evaluator::set_doc(ev, "__DBClose", "Internal: close DB cursor handle", &[]);
    // Dataset internal dispatch entry points
    Evaluator::set_doc(ev, "__DatasetSelect", "Internal: Dataset select dispatcher", &[]);
    Evaluator::set_doc(ev, "__DatasetDescribe", "Internal: Dataset describe dispatcher", &[]);
    Evaluator::set_doc(ev, "__DatasetSort", "Internal: Dataset sort dispatcher", &[]);
    Evaluator::set_doc(ev, "__DatasetOffset", "Internal: Dataset offset/skip dispatcher", &[]);
    Evaluator::set_doc(ev, "__DatasetHead", "Internal: Dataset head dispatcher", &[]);
    Evaluator::set_doc(ev, "__DatasetTail", "Internal: Dataset tail dispatcher", &[]);
    Evaluator::set_doc(ev, "__DatasetDistinct", "Internal: Dataset distinct dispatcher", &[]);
}
