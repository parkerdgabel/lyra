use lazy_static::lazy_static;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct Entry {
    pub features: &'static [&'static str],
    pub effects: &'static [&'static str],
}

// Central registry mapping Symbol head -> required Cargo features and capability tags.
lazy_static! {
    pub static ref REGISTRY: HashMap<&'static str, Entry> = {
        use Entry as E;
        let mut m: HashMap<&'static str, Entry> = HashMap::new();
        // Core string operations
        for s in [
            "StringLength","ToUpper","ToLower","StringJoin","StringJoinWith","StringTrim","StringTrimLeft",
            "StringTrimRight","StringTrimPrefix","StringTrimSuffix","StringTrimChars","StringContains","StringSplit",
            "SplitLines","JoinLines","StartsWith","EndsWith","StringReplace","StringReplaceFirst","StringReverse",
            "StringPadLeft","StringPadRight","StringSlice","IndexOf","LastIndexOf","StringRepeat","IsBlank",
            "Capitalize","TitleCase","EqualsIgnoreCase","StringChars","StringFromChars","StringInterpolate",
            "StringInterpolateWith","StringFormat","StringFormatMap","TemplateRender","HtmlEscape","HtmlUnescape",
            "UrlEncode","UrlDecode","JsonEscape","JsonUnescape","UrlFormEncode","UrlFormDecode","Slugify",
            "StringTruncate","CamelCase","SnakeCase","KebabCase","RegexMatch","RegexFind","RegexFindAll",
            "RegexReplace","ParseDate","FormatDate","DateDiff",
        ] { m.insert(s, E { features: &["string"], effects: &[] }); }

        // List/functional (best-effort common set)
        for s in [
            "Map","MapIndexed","Apply","Thread","Fold","FoldList","Total","Take","Drop","DropWhile","Append","Prepend","Join","Length","Reverse","Partition","Position","Range","Reduce","Reject","Scan","Slice","Sort","TakeWhile","Tally","Transpose","Unique","Unzip","Zip","Filter","Find","Flatten","PackedArray","PackedToList","PackedShape","Part",
        ] { m.insert(s, E { features: &["list"], effects: &[] }); }

        // Assoc/object basics
        for s in [
            "Keys","Values","Merge","Lookup","AssocSet","AssocGet","AssocDelete","AssocDrop","AssocContainsKeyQ","AssociationMap","AssociationMapKeys","AssociationMapPairs","AssociationMapKV","AssocInvert","AssocRenameKeys","AssocSelect","GroupBy","KeySort","SortBy",
        ] { m.insert(s, E { features: &["assoc"], effects: &[] }); }

        // Concurrency primitives
        for s in ["Future","Await","ParallelMap","ParallelTable","MapAsync","Gather","BoundedChannel","Send","Receive","CloseChannel"] {
            m.insert(s, E { features: &["concurrency"], effects: &["time"] });
        }

        // Networking
        for s in ["HttpGet","HttpPost","HttpPut","HttpPatch","HttpDelete","HttpHead","HttpOptions","HttpRequest","Download","DownloadStream","HttpServe","HttpServeRoutes","HttpServerStop","HttpServerAddr"] {
            m.insert(s, E { features: &["net"], effects: &["net"] });
        }
        m.insert("HttpServeTls", E { features: &["net_https"], effects: &["net"] });

        // Databases
        for s in ["Connect","Disconnect","Close","Begin","Commit","Rollback","Ping","Exec","Fetch","InsertRows","UpsertRows","WriteDataset","RegisterTable","ListTables","Table","SQL","SQLCursor","__SQLToRows"] {
            m.insert(s, E { features: &["db"], effects: &["db"] });
        }
        // Engine-specific features are enabled at build time via cargo features (db_sqlite, db_duckdb).
        // Function names are engine-agnostic (e.g., SQL/Exec/Connect), so static symbol analysis cannot
        // disambiguate engine use; CLI may add extra features via flags if needed.

        // Filesystem / IO
        for s in [
            "ReadFile","WriteFile","ReadLines","ReadCSV","WriteCSV","ListDirectory","FileExistsQ","Stat","CanonicalPath","CurrentDirectory","SetDirectory","PathJoin","PathSplit","Basename","Dirname","FileExtension","FileStem","RenderCSV","ParseCSV",
        ] { m.insert(s, E { features: &["io"], effects: &["fs"] }); }
        for s in ["GetEnv","SetEnv","ReadStdin"] { m.insert(s, E { features: &["io"], effects: &["process"] }); }
        for s in ["ToJson","FromJson","ExpandPath"] { m.insert(s, E { features: &["io"], effects: &[] }); }

        // Crypto, Image, Audio (coarse)
        for s in ["Blake3","Sha256","Argon2Hash","EncryptChaCha20Poly1305","DecryptChaCha20Poly1305"] {
            m.insert(s, E { features: &["crypto"], effects: &[] });
        }
        for s in ["ImageInfo","ImageDecode","ImageEncode","ImageResize","ImageCrop","ImagePad","ImageTransform","ImageThumbnail","ImageConvert","ImageCanvas"] { m.insert(s, E { features: &["image"], effects: &[] }); }
        for s in ["ImageSave"] { m.insert(s, E { features: &["image"], effects: &["fs"] }); }
        for s in ["AudioInfo","AudioDecode","AudioEncode","AudioConvert","AudioTrim","AudioGain","AudioResample","AudioConcat","AudioFade","AudioChannelMix"] { m.insert(s, E { features: &["audio"], effects: &[] }); }
        for s in ["AudioSave"] { m.insert(s, E { features: &["audio"], effects: &["fs"] }); }

        // Media (ffmpeg wrapper)
        for s in ["MediaProbe","MediaExtractAudio","MediaThumbnail","MediaTranscode","MediaConcat","MediaMux","MediaPipeline"] {
            m.insert(s, E { features: &["media"], effects: &["process","fs"] });
        }

        // Algebra, Math, Logic
        for s in [
            "Plus","Times","Power","Minus","Divide","Max","Min","Abs",
            "Floor","Ceiling","Round","Trunc","Mod","Quotient","Remainder","DivMod",
            "Sqrt","Exp","Log","Sin","Cos","Tan","ASin","ACos","ATan","ATan2","NthRoot",
            "Total","Mean","Median","Variance","StandardDeviation","GCD","LCM","Factorial","Binomial",
            "Clip","Signum","ToDegrees","ToRadians",
        ] { m.insert(s, E { features: &["math"], effects: &[] }); }
        for s in ["Simplify","Expand","ExpandAll","Factor","CollectTerms","CollectTermsBy","Solve","Roots","D","Apart","CancelRational"] { m.insert(s, E { features: &["algebra"], effects: &[] }); }
        for s in ["And","Or","Not","Equal","Less","LessEqual","Greater","GreaterEqual","If","Switch","When","Unless","EvenQ","OddQ"] { m.insert(s, E { features: &["logic"], effects: &[] }); }

        // Graphs
        for s in [
            "Graph","DropGraph","GraphInfo","AddNodes","AddEdges","RemoveNodes","RemoveEdges","UpsertNodes","UpsertEdges","ListNodes","ListEdges","Neighbors","IncidentEdges","HasNode","HasEdge","ShortestPaths","PageRank","KCore","KCoreDecomposition","MinimumSpanningTree","MaxFlow","ConnectedComponents","StronglyConnectedComponents","GlobalClustering","LocalClustering","TopologicalSort","BFS","DFS","SampleNodes","SampleEdges","ClosenessCentrality","DegreeCentrality",
        ] { m.insert(s, E { features: &["graphs"], effects: &[] }); }

        // Text & search
        for s in ["TextSearch","TextFind","TextReplace","TextLines","TextCount"] { m.insert(s, E { features: &["text"], effects: &[] }); }
        // Encoding detection benefits from optional detector feature
        m.insert("TextDetectEncoding", E { features: &["text","text_encoding_detect"], effects: &[] });
        // File search relies on glob walking when available
        m.insert("TextFilesWithMatch", E { features: &["text","text_glob"], effects: &["fs"] });
        for s in ["FuzzyFindInList","FuzzyFindInText"] { m.insert(s, E { features: &["text_fuzzy"], effects: &[] }); }
        for s in ["FuzzyFindInFiles"] { m.insert(s, E { features: &["text_fuzzy"], effects: &["fs"] }); }
        for s in ["Index","IndexAdd","IndexSearch","IndexInfo"] { m.insert(s, E { features: &["text_index"], effects: &["db","fs"] }); }

        // Package management (stdlib shims)
        for s in [
            "Unuse","ReloadPackage","WithPackage","BeginModule","EndModule","Export","Private","CurrentModule","ModulePath","SetModulePath","PackageVersion","PackagePath","ImportedSymbols","LoadedPackages","RegisterExports","PackageExports",
        ] { m.insert(s, E { features: &["package"], effects: &[] }); }
        for s in [
            "Using","PackageInfo","ListInstalledPackages","NewPackage","NewModule",
        ] { m.insert(s, E { features: &["package"], effects: &["fs"] }); }
        for s in [
            "BuildPackage","TestPackage","LintPackage","PackPackage","GenerateSBOM","SignPackage","UpdatePackage","RemovePackage","PackageVerify",
        ] { m.insert(s, E { features: &["package"], effects: &["fs"] }); }
        for s in [
            "PublishPackage","InstallPackage","LoginRegistry","LogoutRegistry","WhoAmI","PackageAudit",
        ] { m.insert(s, E { features: &["package"], effects: &["net"] }); }

        // Collections
        for s in [
            "HashSet","SetFromList","SetToList","SetInsert","SetRemove","SetMemberQ","SetUnion","SetIntersection","SetDifference","SetSubsetQ","SetEqualQ","ListUnion","ListIntersection","ListDifference","Bag","BagAdd","BagRemove","BagCount","BagUnion","BagIntersection","BagDifference","Queue","Enqueue","Dequeue","Peek","Stack","Push","Pop","Top","PriorityQueue","PQInsert","PQPop","PQPeek",
        ] { m.insert(s, E { features: &["collections"], effects: &[] }); }

        // NDArray
        for s in [
            "NDArray","NDShape","NDReshape","NDTranspose","NDConcat","NDSum","NDMean","NDArgMax","NDMatMul","NDType","NDAsType","NDSlice","NDPermuteDims","NDMap","NDReduce","NDAdd","NDSub","NDMul","NDDiv","NDEltwise","NDPow","NDClip","NDRelu","NDExp","NDSqrt","NDLog","NDSin","NDCos","NDTanh",
        ] { m.insert(s, E { features: &["ndarray"], effects: &[] }); }

        // ML & NN
        for s in [
            "Estimator","Classifier","Regressor","Clusterer",
            "Classify","Predict","Cluster","FeatureExtract","DimensionReduce","MLApply","MLProperty","ClassifyMeasurements","PredictMeasurements","MLCrossValidate","MLTune",
        ] { m.insert(s, E { features: &["ml"], effects: &[] }); }
        for s in [
            // internal NN entry points
            "NetChain","NetInitialize","NetTrain","NetApply","NetSummary","NetProperty","NetEncoder","NetDecoder","NetGraph",
            // old layer heads (kept for now)
            "LinearLayer","ActivationLayer","DropoutLayer","FlattenLayer","SoftmaxLayer","ConvolutionLayer","PoolingLayer","BatchNormLayer","ReshapeLayer","TransposeLayer","ConcatLayer","AddLayer","MulLayer","EmbeddingLayer","LayerNormLayer",
            // canonical heads
            "Network","Sequential","GraphNetwork","Initializer","Dense","Convolution1D","Convolution2D","DepthwiseConv2D","ConvTranspose2D","SeparableConv2D","Pooling","Pooling2D","GlobalAvgPool2D","BatchNorm","LayerNorm","GroupNorm","Residual","Upsample2D","ResidualBlock","Dropout","Flatten","Reshape","Embedding","__TransposeLayer","__ConcatLayer","__AddLayer","__MulLayer",
        ] { m.insert(s, E { features: &["nn"], effects: &[] }); }

        // Containers (Docker/etc.)
        for s in [
            "RuntimeInfo","RuntimeCapabilities","PingContainers","ListContainers","DescribeContainers","InspectContainer","StartContainer","StopContainer","RestartContainer","PauseContainer","UnpauseContainer","RemoveContainer","WaitContainer","Logs","ExecInContainer","Container","RunContainer","ConnectContainers","DisconnectContainers","ListNetworks","Network","RemoveNetwork","ListVolumes","Volume","RemoveVolume","CopyToContainer","CopyFromContainer","SearchImages","ListImages","InspectImage","PullImage","PushImage","SaveImage","LoadImage","RemoveImage","PruneImages","ImageHistory","ExportImages","TagImage","AddRegistryAuth","ListRegistryAuth","InspectRegistryImage","Events","Stats",
        ] { m.insert(s, E { features: &["containers"], effects: &["process","net","fs"] }); }

        // Dataset
        for s in [
            "DatasetFromRows","DatasetSchema","Describe","ExplainDataset","ExplainSQL","Columns","Select","SelectCols","RenameCols","WithColumns","Join","Concat","Union","UnionByPosition","FilterRows","GroupBy","Agg","Sort","Count","Distinct","DistinctOn","Offset","LimitRows","Head","Tail","Collect","ShowDataset","Cast","Coalesce","col",
        ] { m.insert(s, E { features: &["dataset"], effects: &[] }); }
        for s in ["ReadCSVDataset","ReadJsonLinesDataset","WriteDataset"] { m.insert(s, E { features: &["dataset"], effects: &["fs"] }); }

        m
    };
}

pub fn features_for(
    symbols: &std::collections::HashSet<String>,
) -> std::collections::HashSet<String> {
    let mut out = std::collections::HashSet::new();
    for s in symbols {
        if let Some(e) = REGISTRY.get(s.as_str()) {
            for f in e.features {
                out.insert(f.to_string());
            }
        }
    }
    out
}

pub fn capabilities_for(
    symbols: &std::collections::HashSet<String>,
) -> std::collections::HashSet<&'static str> {
    let mut out = std::collections::HashSet::new();
    for s in symbols {
        if let Some(e) = REGISTRY.get(s.as_str()) {
            for cap in e.effects {
                out.insert(*cap);
            }
        }
    }
    out
}
