# MISC

| Function | Usage | Summary |
|---|---|---|
| `ACos` | `ACos[x]` | Arc-cosine (inverse cosine) |
| `ASin` | `ASin[x]` | Arc-sine (inverse sine) |
| `ATan` | `ATan[x]` | Arc-tangent (inverse tangent) |
| `ATan2` | `ATan2[y, x]` | Arc-tangent of y/x (quadrant aware) |
| `ActivationLayer` | `ActivationLayer[kind, opts]` | Activation layer (Relu/Tanh/Sigmoid) |
| `Actor` | `Actor[handler]` | Create actor with handler (held) |
| `AddDuration` | `AddDuration[dt, dur]` | Add duration to DateTime/epochMs |
| `AddEdges` | `AddEdges[graph, edges]` | Add edges with optional keys and weights |
| `AddLayer` | `AddLayer[opts]` | Elementwise add layer |
| `AddNodes` | `AddNodes[graph, nodes]` | Add nodes by id and attributes |
| `AddRegistryAuth` | `AddRegistryAuth[server, user, password]` | Add registry credentials |
| `AlignCenter` | `AlignCenter[text, width, pad?]` | Pad on both sides to center |
| `AlignLeft` | `AlignLeft[text, width, pad?]` | Pad right to width |
| `AlignRight` | `AlignRight[text, width, pad?]` | Pad left to width |
| `AnsiEnabled` | `AnsiEnabled[]` | Are ANSI colors enabled? |
| `AnsiStyle` | `AnsiStyle[text, opts?]` | Style text with ANSI codes |
| `Apart` | `Apart[]` |  |
| `Apply` | `Apply[f, list]` | Apply head to list elements: Apply[f, {…}] |
| `ArgsParse` | `ArgsParse[list]` | Parse CLI-like args to assoc |
| `Ask` | `Ask[actor, msg]` | Request/response with actor (held) |
| `Await` | `Await[future]` | Wait for Future and return value |
| `BagAdd` | `BagAdd[bag, value]` | Add item to bag |
| `BagCount` | `BagCount[bag, value]` | Count occurrences of value |
| `BagCreate` | `BagCreate[]` | Create a multiset bag |
| `BagDifference` | `BagDifference[a, b]` | Difference of two bags |
| `BagIntersection` | `BagIntersection[a, b]` | Intersection of two bags |
| `BagRemove` | `BagRemove[bag, value]` | Remove one item from bag |
| `BagSize` | `BagSize[bag]` | Total items in bag |
| `BagUnion` | `BagUnion[a, b]` | Union of two bags |
| `BarChart` | `BarChart[data, opts]` | Render a bar chart |
| `BatchNormLayer` | `BatchNormLayer[opts]` | Batch normalization layer |
| `Binomial` | `Binomial[n, k]` | Binomial coefficient nCk |
| `BooleanQ` | `BooleanQ[x]` | Is value Boolean? |
| `BoundedChannel` | `BoundedChannel[n]` | Create bounded channel |
| `BoxText` | `BoxText[text, opts?]` | Draw a box around text |
| `BuildImage` | `BuildImage[context, opts?]` | Build image from context |
| `BuildPackage` | `BuildPackage[path?, opts?]` | Build a package (requires lyra-pm) |
| `BusyWait` | `BusyWait[]` |  |
| `CamelCase` | `CamelCase[s]` | Convert to camelCase |
| `Cancel` | `Cancel[future]` | Request cooperative cancellation |
| `CancelRational` | `CancelRational[]` |  |
| `CancelSchedule` | `CancelSchedule[token]` | Cancel scheduled task |
| `CancelScope` | `CancelScope[scope]` | Cancel running scope |
| `CancelWatch` | `CancelWatch[token]` | Cancel a directory watch |
| `Capitalize` | `Capitalize[s]` | Capitalize first letter |
| `Cast` | `Cast[]` |  |
| `Catch` | `Catch[body]` | Catch a thrown value (held) |
| `Ceiling` | `Ceiling[x]` | Smallest integer >= x |
| `ChangelogGenerate` | `ChangelogGenerate[]` |  |
| `Chart` | `Chart[spec, opts]` | Render a chart from a spec |
| `Chat` | `Chat[]` |  |
| `Citations` | `Citations[]` |  |
| `Cite` | `Cite[]` |  |
| `Classify` | `Classify[data, opts]` | Train a classifier (baseline/logistic) |
| `ClassifyMeasurements` | `ClassifyMeasurements[model, data, opts]` | Evaluate classifier metrics |
| `Clip` | `Clip[x, min, max]` | Clamp value to [min,max] |
| `Close` | `Close[cursor]` | Close a cursor |
| `CloseChannel` | `CloseChannel[ch]` | Close channel |
| `ClosenessCentrality` | `ClosenessCentrality[graph]` | Per-node closeness centrality |
| `Cluster` | `Cluster[data, opts]` | Cluster points (prototype) |
| `Coalesce` | `Coalesce[values…]` | First non-null value |
| `Collect` | `Collect[ds, limit?, opts?]` | Materialize dataset rows as a list |
| `CollectTerms` | `CollectTerms[]` |  |
| `CollectTermsBy` | `CollectTermsBy[]` |  |
| `Columnize` | `Columnize[lines, opts?]` | Align lines in columns |
| `Columns` | `Columns[ds]` | List column names for a dataset |
| `Complete` | `Complete[]` |  |
| `Compose` | `Compose[f, g, …]` | Compose functions left-to-right |
| `ConfigFind` | `ConfigFind[]` |  |
| `ConfigLoad` | `ConfigLoad[]` |  |
| `Confirm` | `Confirm[text, opts?]` | Ask yes/no question (TTY) |
| `ConstantFunction` | `ConstantFunction[c]` | Constant function returning c |
| `ContainersClose` | `ContainersClose[]` | Close open fetch handles |
| `ContainersFetch` | `ContainersFetch[paths, opts?]` | Fetch external resources for build |
| `ConvolutionLayer` | `ConvolutionLayer[opts]` | 2D convolution layer |
| `Cos` | `Cos[x]` | Cosine (radians) |
| `CostAdd` | `CostAdd[]` |  |
| `CostSoFar` | `CostSoFar[]` |  |
| `Count` | `Count[list, value|pred]` | Count elements equal to value or matching predicate |
| `CreateContainer` | `CreateContainer[image, opts?]` | Create a container |
| `CreateNetwork` | `CreateNetwork[opts?]` | Create network |
| `CreateVolume` | `CreateVolume[opts?]` | Create volume |
| `Cron` | `Cron[expr, body]` | Schedule with cron expression (held) |
| `CurrentModule` | `CurrentModule[]` | Current module path/name |
| `D` | `D[]` |  |
| `DateDiff` | `DateDiff[a, b]` | Difference between two DateTime in ms |
| `DateFormat` | `DateFormat[dt, fmt?]` | Format DateTime or epochMs to string |
| `DateParse` | `DateParse[s]` | Parse date/time string to epochMs |
| `DateTime` | `DateTime[spec]` | Build/parse DateTime assoc (UTC) |
| `DegreeCentrality` | `DegreeCentrality[graph]` | Per-node degree centrality |
| `Dequeue` | `Dequeue[queue]` | Dequeue value |
| `Describe` | `Describe[]` |  |
| `DescribeBuiltins` | `DescribeBuiltins[]` | List builtins with attributes (and docs when available). |
| `DescribeContainers` | `DescribeContainers[]` | Describe available container APIs |
| `DiffDuration` | `DiffDuration[a, b]` | Difference between DateTimes |
| `DimensionReduce` | `DimensionReduce[data, opts]` | Reduce dimensionality (PCA-like) |
| `Disconnect` | `Disconnect[conn]` | Close a database connection |
| `DisconnectContainers` | `DisconnectContainers[]` | Disconnect from container runtime |
| `DivMod` | `DivMod[a, n]` | Quotient and remainder |
| `Divide` | `Divide[a, b]` | Divide two numbers. |
| `Documentation` | `Documentation[name]` | Documentation card for a builtin. |
| `DotenvLoad` | `DotenvLoad[]` |  |
| `DropGraph` | `DropGraph[graph]` | Drop a graph handle |
| `DropoutLayer` | `DropoutLayer[p]` | Dropout probability p |
| `Duration` | `Duration[spec]` | Build Duration assoc from ms or fields |
| `DurationParse` | `DurationParse[s]` | Parse human duration (e.g., 1h30m) |
| `Embed` | `Embed[]` |  |
| `EmbeddingLayer` | `EmbeddingLayer[opts]` | Embedding lookup layer |
| `EmptyQ` | `EmptyQ[]` |  |
| `EndModule` | `EndModule[]` | End current module scope |
| `EndOf` | `EndOf[dt, unit]` | End of unit (day/week/month) |
| `EndScope` | `EndScope[scope]` | End scope and release resources |
| `EndsWith` | `EndsWith[]` |  |
| `Enqueue` | `Enqueue[queue, value]` | Enqueue value |
| `EnvExpand` | `EnvExpand[]` |  |
| `EqualsIgnoreCase` | `EqualsIgnoreCase[a, b]` | Case-insensitive string equality |
| `Events` | `Events[opts?]` | Subscribe to runtime events |
| `Exp` | `Exp[x]` | Natural exponential e^x |
| `Expand` | `Expand[]` |  |
| `ExpandAll` | `ExpandAll[]` |  |
| `Explain` | `Explain[expr]` | Explain evaluation; returns trace steps when enabled. |
| `ExplainContainers` | `ExplainContainers[]` | Explain container runtime configuration |
| `ExplainSQL` | `ExplainSQL[ds]` | Render SQL for pushdown-capable parts |
| `Export` | `Export[symbols]` | Mark symbol(s) as public |
| `ExportImages` | `ExportImages[refs, path]` | Export images to an archive |
| `Exported` | `Exported[]` |  |
| `Factor` | `Factor[]` |  |
| `Factorial` | `Factorial[n]` | n! (product 1..n) |
| `Fail` | `Fail[]` |  |
| `FeatureExtract` | `FeatureExtract[data, opts]` | Learn preprocessing (impute/encode/standardize) |
| `Figure` | `Figure[items, opts]` | Compose multiple charts in a grid |
| `Finally` | `Finally[body, cleanup]` | Ensure cleanup runs (held) |
| `FixedPoint` | `FixedPoint[f, x]` | Iterate f until convergence |
| `FixedPointList` | `FixedPointList[f, x]` | List of iterates until convergence |
| `FlattenLayer` | `FlattenLayer[opts]` | Flatten to 1D |
| `Floor` | `Floor[x]` | Largest integer <= x |
| `FoldList` | `FoldList[f, init, list]` | Cumulative fold producing intermediates |
| `FormatDate` | `FormatDate[dt, fmt]` | Format DateTime with strftime pattern |
| `FormatLyra` | `FormatLyra[]` |  |
| `FormatLyraFile` | `FormatLyraFile[]` |  |
| `FormatLyraText` | `FormatLyraText[]` |  |
| `Future` | `Future[expr]` | Create a Future from an expression (held) |
| `GCD` | `GCD[a, b, …]` | Greatest common divisor |
| `Gather` | `Gather[futures]` | Await Futures in same structure |
| `GenerateSBOM` | `GenerateSBOM[path?, opts?]` | Generate SBOM (requires lyra-pm) |
| `GetDownValues` | `GetDownValues[]` |  |
| `GetOwnValues` | `GetOwnValues[]` |  |
| `GetSubValues` | `GetSubValues[]` |  |
| `GetUpValues` | `GetUpValues[]` |  |
| `Gets` | `Gets[path?]` | Read entire stdin or file as string |
| `GitAdd` | `GitAdd[paths, opts?]` | Stage files for commit |
| `GitApply` | `GitApply[patch, opts?]` | Apply a patch (or check only) |
| `GitBranchCreate` | `GitBranchCreate[name, opts?]` | Create a new branch |
| `GitBranchList` | `GitBranchList[]` | List local branches |
| `GitCommit` | `GitCommit[message, opts?]` | Create a commit with message |
| `GitCreateFeatureBranch` | `GitCreateFeatureBranch[opts?]` | Create and switch to a feature branch |
| `GitCurrentBranch` | `GitCurrentBranch[]` | Current branch name |
| `GitDiff` | `GitDiff[opts?]` | Diff against base and optional paths |
| `GitEnsureRepo` | `GitEnsureRepo[opts?]` | Ensure Cwd is a git repo (init if needed) |
| `GitFetch` | `GitFetch[remote?]` | Fetch from remote |
| `GitInit` | `GitInit[opts?]` | Initialize a new git repository |
| `GitLog` | `GitLog[opts?]` | List commits with formatting options |
| `GitPull` | `GitPull[remote?, opts?]` | Pull from remote |
| `GitPush` | `GitPush[opts?]` | Push to remote |
| `GitRemoteList` | `GitRemoteList[]` | List remotes |
| `GitRoot` | `GitRoot[]` | Path to repository root (Null if absent) |
| `GitSmartCommit` | `GitSmartCommit[opts?]` | Stage + conventional commit (auto msg option) |
| `GitStatus` | `GitStatus[opts?]` | Status (porcelain) with branch/ahead/behind/changes |
| `GitStatusSummary` | `GitStatusSummary[opts?]` | Summarize status counts and branch |
| `GitSwitch` | `GitSwitch[name, opts?]` | Switch to branch (optionally create) |
| `GitSyncUpstream` | `GitSyncUpstream[opts?]` | Fetch, rebase (or merge), and push upstream |
| `GitVersion` | `GitVersion[]` | Get git client version string |
| `Gunzip` | `Gunzip[dataOrPath, opts?]` | Gunzip-decompress a string or a .gz file; optionally write to path. |
| `Gzip` | `Gzip[dataOrPath, opts?]` | Gzip-compress a string or a file; optionally write to path. |
| `HasEdge` | `HasEdge[graph, spec]` | Does graph contain edge? |
| `HasNode` | `HasNode[graph, id]` | Does graph contain node? |
| `Head` | `Head[ds, n]` | Take first n rows |
| `Histogram` | `Histogram[data, opts]` | Render a histogram |
| `HtmlEscape` | `HtmlEscape[s]` | Escape string for HTML |
| `HtmlUnescape` | `HtmlUnescape[s]` | Unescape HTML-escaped string |
| `HybridSearch` | `HybridSearch[]` |  |
| `IdempotencyKey` | `IdempotencyKey[]` |  |
| `Identity` | `Identity[x]` | Identity function: returns its argument |
| `If` | `If[cond, then, else?]` | Conditional: If[cond, then, else?] (held) |
| `ImageHistory` | `ImageHistory[]` |  |
| `ImportedSymbols` | `ImportedSymbols[]` | Assoc of package -> imported symbols |
| `InScope` | `InScope[scope, body]` | Run body inside a scope (held) |
| `IncidentEdges` | `IncidentEdges[graph, id, opts?]` | Edges incident to a node |
| `IndexOf` | `IndexOf[]` |  |
| `InspectContainer` | `InspectContainer[id]` | Inspect container |
| `InspectImage` | `InspectImage[ref]` | Inspect image details |
| `InspectNetwork` | `InspectNetwork[name]` | Inspect network |
| `InspectRegistryImage` | `InspectRegistryImage[ref, opts?]` | Inspect remote registry image |
| `InspectVolume` | `InspectVolume[name]` | Inspect volume |
| `InstallPackage` | `InstallPackage[name, opts?]` | Install a package (requires lyra-pm) |
| `IntegerQ` | `IntegerQ[x]` | Is value an integer? |
| `IsBlank` | `IsBlank[]` |  |
| `It` | `It[]` |  |
| `KebabCase` | `KebabCase[s]` | Convert to kebab-case |
| `KillProcess` | `KillProcess[proc, signal?]` | Send signal to process |
| `LCM` | `LCM[a, b, …]` | Least common multiple |
| `LastIndexOf` | `LastIndexOf[]` |  |
| `LayerNormLayer` | `LayerNormLayer[opts]` | Layer normalization layer |
| `LimitRows` | `LimitRows[ds, n]` | Limit number of rows |
| `LinePlot` | `LinePlot[data, opts]` | Render a line plot |
| `LinearLayer` | `LinearLayer[opts]` | Linear (fully-connected) layer |
| `LintLyra` | `LintLyra[]` |  |
| `LintLyraFile` | `LintLyraFile[]` |  |
| `LintLyraText` | `LintLyraText[]` |  |
| `LintPackage` | `LintPackage[path?, opts?]` | Lint a package (requires lyra-pm) |
| `ListContainers` | `ListContainers[opts?]` | List containers |
| `ListDifference` | `ListDifference[a, b]` | Elements in a not in b |
| `ListEdges` | `ListEdges[graph, opts?]` | List edges |
| `ListImages` | `ListImages[opts?]` | List local images |
| `ListInstalledPackages` | `ListInstalledPackages[]` | List packages available on $PackagePath |
| `ListIntersection` | `ListIntersection[a, b]` | Intersection of lists |
| `ListNetworks` | `ListNetworks[]` | List networks |
| `ListNodes` | `ListNodes[graph, opts?]` | List nodes |
| `ListQ` | `ListQ[x]` | Is value a list? |
| `ListRegistryAuth` | `ListRegistryAuth[]` | List stored registry credentials |
| `ListUnion` | `ListUnion[a, b]` | Union of lists (dedup) |
| `ListVolumes` | `ListVolumes[]` | List volumes |
| `LoadImage` | `LoadImage[path]` | Load image from tar |
| `LoadedPackages` | `LoadedPackages[]` | Assoc of loaded packages |
| `LocalClustering` | `LocalClustering[graph]` | Per-node clustering coefficient |
| `MLApply` | `MLApply[model, x, opts]` | Apply a trained model to input |
| `MLCrossValidate` | `MLCrossValidate[data, opts]` | Cross-validate with simple split |
| `MLProperty` | `MLProperty[model, prop]` | Inspect trained model properties |
| `MLTune` | `MLTune[data, opts]` | Parameter sweep with basic scoring |
| `MakeDirectory` | `MakeDirectory[path, opts?]` | Create a directory (Parents option) |
| `Map` | `Map[f, list]` | Map a function over a list. |
| `MapAsync` | `MapAsync[f, list]` | Map to Futures over list |
| `MatchQ` | `MatchQ[expr, pattern]` | Pattern match predicate (held) |
| `MaxFlow` | `MaxFlow[graph, src, dst]` | Maximum flow value and cut |
| `Mean` | `Mean[list]` | Arithmetic mean of list |
| `Median` | `Median[list]` | Median of list |
| `Metrics` | `Metrics[]` |  |
| `MetricsReset` | `MetricsReset[]` |  |
| `MinimumSpanningTree` | `MinimumSpanningTree[graph]` | Edges in a minimum spanning tree |
| `Minus` | `Minus[a, b?]` | Subtract or unary negate. |
| `Mod` | `Mod[a, n]` | Modulo remainder ((a mod n) >= 0) |
| `Model` | `Model[]` |  |
| `ModelsList` | `ModelsList[]` |  |
| `ModuleInfo` | `ModuleInfo[]` |  |
| `ModulePath` | `ModulePath[]` | Get module search path |
| `MonotonicNow` | `MonotonicNow[]` | Monotonic clock milliseconds since start |
| `MulLayer` | `MulLayer[opts]` | Elementwise multiply layer |
| `NegativeQ` | `NegativeQ[x]` | Is number < 0? |
| `Neighbors` | `Neighbors[graph, id, opts?]` | Neighbor node ids for a node |
| `Nest` | `Nest[f, x, n]` | Nest function n times: Nest[f, x, n] |
| `NestList` | `NestList[f, x, n]` | Nest and collect intermediate values |
| `NetApply` | `NetApply[net, x, opts]` | Apply network to input |
| `NetDecoder` | `NetDecoder[net]` | Get network output decoder |
| `NetEncoder` | `NetEncoder[net]` | Get network input encoder |
| `NetGraph` | `NetGraph[]` |  |
| `NetProperty` | `NetProperty[net, prop]` | Inspect network properties |
| `NetSummary` | `NetSummary[net]` | Summarize network structure |
| `NetTrain` | `NetTrain[net, data, opts]` | Train network on data |
| `NewModule` | `NewModule[pkgPath, name]` | Scaffold a new module file in a package |
| `NewPackage` | `NewPackage[name, opts?]` | Scaffold a new package directory |
| `NonEmptyQ` | `NonEmptyQ[x]` | Is list/string/assoc non-empty? |
| `NonNegativeQ` | `NonNegativeQ[x]` | Is number >= 0? |
| `NonPositiveQ` | `NonPositiveQ[x]` | Is number <= 0? |
| `NowMs` | `NowMs[]` | Current UNIX time in milliseconds |
| `NthRoot` | `NthRoot[]` |  |
| `NumberQ` | `NumberQ[x]` | Is value numeric (int/real)? |
| `Offset` | `Offset[ds, n]` | Skip first n rows |
| `OnFailure` | `OnFailure[body, handler]` | Handle Failure values (held) |
| `OpenApiGenerate` | `OpenApiGenerate[routes, opts?]` | Generate OpenAPI from routes |
| `PQCreate` | `PQCreate[]` | Create a priority queue |
| `PQEmptyQ` | `PQEmptyQ[pq]` | Is priority queue empty? |
| `PQInsert` | `PQInsert[pq, priority, value]` | Insert with priority |
| `PQPeek` | `PQPeek[pq]` | Peek min (or max) priority |
| `PQPop` | `PQPop[pq]` | Pop min (or max) priority |
| `PQSize` | `PQSize[pq]` | Size of priority queue |
| `PackPackage` | `PackPackage[path?, opts?]` | Pack artifacts for distribution (requires lyra-pm) |
| `PackageAudit` | `PackageAudit[path?, opts?]` | Audit dependencies (requires lyra-pm) |
| `PackageExports` | `PackageExports[name]` | Get exports list for a package |
| `PackageInfo` | `PackageInfo[name]` | Read package metadata (name, version, path) |
| `PackagePath` | `PackagePath[]` | Get current $PackagePath |
| `PackageVerify` | `PackageVerify[path?, opts?]` | Verify signatures (requires lyra-pm) |
| `PackageVersion` | `PackageVersion[pkgPath]` | Read version from manifest |
| `PackedArray` | `PackedArray[]` |  |
| `PackedShape` | `PackedShape[]` |  |
| `PackedToList` | `PackedToList[]` |  |
| `ParallelEvaluate` | `ParallelEvaluate[exprs, opts?]` | Evaluate expressions concurrently (held) |
| `ParallelMap` | `ParallelMap[f, list]` | Map in parallel over list |
| `ParallelTable` | `ParallelTable[exprs]` | Evaluate list of expressions in parallel (held) |
| `ParseDate` | `ParseDate[s]` | Parse date string into DateTime |
| `PasswordPrompt` | `PasswordPrompt[text, opts?]` | Prompt for password without echo |
| `PathExtname` | `PathExtname[path]` | File extension without dot |
| `PathNormalize` | `PathNormalize[path]` | Normalize path separators |
| `PathRelative` | `PathRelative[base, path]` | Relative path from base |
| `PathResolve` | `PathResolve[base, path]` | Resolve against base directory |
| `PatternQ` | `PatternQ[expr]` | Is value a pattern? (held) |
| `PauseContainer` | `PauseContainer[id]` | Pause a container |
| `Peek` | `Peek[handle]` | Peek top of stack/queue |
| `Ping` | `Ping[conn]` | Check connectivity for a database connection |
| `PingContainers` | `PingContainers[]` |  |
| `PoolingLayer` | `PoolingLayer[opts]` | Pooling layer (Max/Avg) |
| `Pop` | `Pop[stack]` | Pop from stack |
| `PositiveQ` | `PositiveQ[x]` | Is number > 0? |
| `Power` | `Power[a, b]` | Exponentiation (right-associative in parser). |
| `Predict` | `Predict[data, opts]` | Train a regressor (baseline/linear) |
| `PredictMeasurements` | `PredictMeasurements[model, data, opts]` | Evaluate regressor metrics |
| `Private` | `Private[symbols]` | Mark symbol(s) as private |
| `ProgressAdvance` | `ProgressAdvance[]` |  |
| `ProgressBar` | `ProgressBar[]` |  |
| `ProgressFinish` | `ProgressFinish[]` |  |
| `ProjectDiscover` | `ProjectDiscover[start?]` | Search upwards for project.lyra |
| `ProjectInfo` | `ProjectInfo[root?]` | Summarize project (name, version, paths) |
| `ProjectInit` | `ProjectInit[path, opts?]` | Initialize new project (scaffold) |
| `ProjectLoad` | `ProjectLoad[root?]` | Load and evaluate project.lyra (normalized) |
| `ProjectRoot` | `ProjectRoot[]` | Return project root path (or Null) |
| `ProjectValidate` | `ProjectValidate[root?]` | Validate project manifest and structure |
| `Prompt` | `Prompt[text, opts?]` | Prompt user for input (TTY) |
| `PruneImages` | `PruneImages[opts?]` | Remove unused images |
| `PublishPackage` | `PublishPackage[path?, opts?]` | Publish to registry (requires lyra-pm) |
| `PullImage` | `PullImage[ref, opts?]` | Pull an image |
| `Push` | `Push[stack, value]` | Push onto stack |
| `PushImage` | `PushImage[ref, opts?]` | Push an image to registry |
| `Puts` | `Puts[content, path]` | Write string to file (overwrite) |
| `PutsAppend` | `PutsAppend[content, path]` | Append string to file |
| `QueueCreate` | `QueueCreate[]` | Create a FIFO queue |
| `QueueEmptyQ` | `QueueEmptyQ[queue]` | Is queue empty? |
| `QueueSize` | `QueueSize[queue]` | Size of queue |
| `Quotient` | `Quotient[a, n]` | Integer division quotient |
| `RAGAnswer` | `RAGAnswer[]` |  |
| `RAGAssembleContext` | `RAGAssembleContext[]` |  |
| `RAGChunk` | `RAGChunk[]` |  |
| `RAGIndex` | `RAGIndex[]` |  |
| `RAGRetrieve` | `RAGRetrieve[]` |  |
| `RealQ` | `RealQ[x]` | Is value a real number? |
| `Recall` | `Recall[]` |  |
| `Receive` | `Receive[ch, opts?]` | Receive value from channel |
| `RegexFind` | `RegexFind[s, pattern]` | Find first regex capture groups |
| `RegexFindAll` | `RegexFindAll[s, pattern]` | Find all regex capture groups |
| `RegexIsMatch` | `RegexIsMatch[s, pattern]` | Test if regex matches string |
| `RegexMatch` | `RegexMatch[s, pattern]` | Return first regex match |
| `RegexReplace` | `RegexReplace[s, pattern, repl]` | Replace matches using regex |
| `RegisterExports` | `RegisterExports[name, exports]` | Register exports for a package (internal) |
| `RegisterTable` | `RegisterTable[conn, name, rows]` | Register in-memory rows as a table (mock) |
| `ReleaseTag` | `ReleaseTag[]` |  |
| `ReloadPackage` | `ReloadPackage[name]` | Reload a package |
| `Remainder` | `Remainder[a, n]` | Integer division remainder |
| `Remember` | `Remember[]` |  |
| `RenameCols` | `RenameCols[ds, mapping]` | Rename columns via mapping |
| `RenameContainer` | `RenameContainer[id, name]` | Rename a container |
| `Replace` | `Replace[expr, rules]` | Replace first match by rule(s). |
| `ReplaceAll` | `ReplaceAll[expr, rules]` | Replace all matches by rule(s). |
| `ReplaceFirst` | `ReplaceFirst[expr, rule]` | Replace first element(s) matching pattern. |
| `ReplaceRepeated` | `ReplaceRepeated[]` |  |
| `ReshapeLayer` | `ReshapeLayer[shape]` | Reshape to given shape |
| `ResolveRelative` | `ResolveRelative[]` |  |
| `RestartContainer` | `RestartContainer[id]` | Restart a container |
| `Reverse` | `Reverse[list]` | Reverse a list |
| `RightCompose` | `RightCompose[f, g, …]` | Compose functions right-to-left |
| `Roots` | `Roots[]` |  |
| `Round` | `Round[x]` | Round to nearest integer |
| `Rule` | `Rule[k, v]` | Format a rule (k->v) as a string |
| `SampleEdges` | `SampleEdges[graph, k]` | Sample k edges uniformly |
| `SampleNodes` | `SampleNodes[graph, k]` | Sample k nodes uniformly |
| `SaveImage` | `SaveImage[ref, path]` | Save image to tar |
| `ScatterPlot` | `ScatterPlot[data, opts]` | Render a scatter plot |
| `ScheduleEvery` | `ScheduleEvery[ms, body]` | Schedule recurring task (held) |
| `Schema` | `Schema[value]` | Return a minimal schema for a value/association. |
| `Scope` | `Scope[opts, body]` | Run body with resource limits (held) |
| `SearchImages` | `SearchImages[query, opts?]` | Search registry images |
| `SecretsGet` | `SecretsGet[]` |  |
| `Send` | `Send[ch, value]` | Send value to channel (held) |
| `SessionClear` | `SessionClear[]` |  |
| `Set` | `Set[symbol, value]` | Assignment: Set[symbol, value]. |
| `SetCreate` | `SetCreate[values]` | Create a set from values |
| `SetDelayed` | `SetDelayed[symbol, expr]` | Delayed assignment evaluated on use. |
| `SetDifference` | `SetDifference[a, b]` | Elements in a not in b |
| `SetDownValues` | `SetDownValues[]` |  |
| `SetEmptyQ` | `SetEmptyQ[set]` | Is set empty? |
| `SetEqualQ` | `SetEqualQ[a, b]` | Are two sets equal? |
| `SetFromList` | `SetFromList[list]` | Create set from list |
| `SetInsert` | `SetInsert[set, value]` | Insert value into set |
| `SetIntersection` | `SetIntersection[a, b]` | Intersection of two sets |
| `SetMemberQ` | `SetMemberQ[set, value]` | Is value a member of set? |
| `SetModulePath` | `SetModulePath[path]` | Set module search path |
| `SetOwnValues` | `SetOwnValues[]` |  |
| `SetRemove` | `SetRemove[set, value]` | Remove value from set |
| `SetSize` | `SetSize[set]` | Number of elements in set |
| `SetSubValues` | `SetSubValues[]` |  |
| `SetSubsetQ` | `SetSubsetQ[a, b]` | Is a subset of b? |
| `SetToList` | `SetToList[set]` | Convert set to list |
| `SetUnion` | `SetUnion[a, b]` | Union of two sets |
| `SetUpValues` | `SetUpValues[]` |  |
| `ShortestPaths` | `ShortestPaths[graph, start, opts?]` | Shortest path distances from start |
| `ShowDataset` | `ShowDataset[ds, opts?]` | Pretty-print a dataset table to string |
| `SignPackage` | `SignPackage[path?, opts?]` | Sign package (requires lyra-pm) |
| `Signum` | `Signum[x]` | Sign of number (-1,0,1) |
| `Simplify` | `Simplify[]` |  |
| `Sin` | `Sin[x]` | Sine (radians) |
| `Sleep` | `Sleep[ms]` | Sleep for N milliseconds |
| `Slugify` | `Slugify[s]` | Slugify for URLs |
| `SnakeCase` | `SnakeCase[s]` | Convert to snake_case |
| `SoftmaxLayer` | `SoftmaxLayer[opts]` | Softmax over last dimension |
| `Solve` | `Solve[]` |  |
| `Span` | `Span[]` |  |
| `SpanEnd` | `SpanEnd[]` |  |
| `SplitLines` | `SplitLines[s]` | Split string on 
 into lines |
| `Sqrt` | `Sqrt[x]` | Square root |
| `StackCreate` | `StackCreate[]` | Create a stack |
| `StackEmptyQ` | `StackEmptyQ[stack]` | Is stack empty? |
| `StackSize` | `StackSize[stack]` | Size of a stack |
| `StandardDeviation` | `StandardDeviation[list]` | Standard deviation of list |
| `StartContainer` | `StartContainer[id]` | Start a container |
| `StartOf` | `StartOf[dt, unit]` | Start of unit (day/week/month) |
| `StartScope` | `StartScope[opts, body]` | Start a managed scope (held) |
| `StartsWith` | `StartsWith[]` |  |
| `Stats` | `Stats[id, opts?]` | Stream container stats |
| `StopActor` | `StopActor[actor]` | Stop actor |
| `StopContainer` | `StopContainer[id, opts?]` | Stop a container |
| `StripAnsi` | `StripAnsi[text]` | Remove ANSI escape codes from string |
| `StronglyConnectedComponents` | `StronglyConnectedComponents[graph]` | Strongly connected components |
| `Subgraph` | `Subgraph[graph, ids]` | Induced subgraph from node set |
| `Switch` | `Switch[expr, rules…]` | Multi-way conditional by equals (held) |
| `SymbolQ` | `SymbolQ[x]` | Is value a symbol? |
| `Symlink` | `Symlink[src, dst]` | Create a symbolic link |
| `TagImage` | `TagImage[src, dest]` | Tag an image |
| `Tail` | `Tail[ds, n]` | Take last n rows |
| `Tan` | `Tan[x]` | Tangent (radians) |
| `TarCreate` | `TarCreate[dest, inputs, opts?]` | Create a .tar (optionally .tar.gz) archive from inputs. |
| `TarExtract` | `TarExtract[src, dest]` | Extract a .tar or .tar.gz archive into a directory. |
| `Tell` | `Tell[actor, msg]` | Send message to actor (held) |
| `TermSize` | `TermSize[]` | Current terminal width/height |
| `TestPackage` | `TestPackage[path?, opts?]` | Run package tests (requires lyra-pm) |
| `TextCount` | `TextCount[]` |  |
| `TextDetectEncoding` | `TextDetectEncoding[]` |  |
| `TextFilesWithMatch` | `TextFilesWithMatch[]` |  |
| `TextFind` | `TextFind[]` |  |
| `TextLines` | `TextLines[]` |  |
| `TextReplace` | `TextReplace[]` |  |
| `TextSearch` | `TextSearch[]` |  |
| `Thread` | `Thread[expr]` | Thread Sequence and lists into arguments. |
| `Through` | `Through[fs, x]` | Through[{f,g}, x] applies each to x |
| `Throw` | `Throw[x]` | Throw a value for Catch |
| `TimeZoneConvert` | `TimeZoneConvert[dt, tz]` | Convert DateTime to another timezone |
| `TitleCase` | `TitleCase[s]` | Convert to Title Case |
| `TraceExport` | `TraceExport[]` |  |
| `TraceGet` | `TraceGet[]` |  |
| `TransposeLayer` | `TransposeLayer[perm]` | Transpose dimensions |
| `Trunc` | `Trunc[]` |  |
| `Truncate` | `Truncate[text, width, ellipsis?]` | Truncate to width with ellipsis |
| `Try` | `Try[body]` | Try body; capture failures (held) |
| `TryOr` | `TryOr[body, default]` | Try body else default (held) |
| `TryReceive` | `TryReceive[ch]` | Non-blocking receive |
| `TrySend` | `TrySend[ch, value]` | Non-blocking send (held) |
| `Unless` | `Unless[cond, body]` | Evaluate body when condition is False (held) |
| `UnpauseContainer` | `UnpauseContainer[id]` | Unpause a container |
| `Unset` | `Unset[symbol]` | Clear definition: Unset[symbol]. |
| `Unuse` | `Unuse[name]` | Unload a package; hide imported symbols |
| `UpdatePackage` | `UpdatePackage[name, opts?]` | Update a package (requires lyra-pm) |
| `UrlDecode` | `UrlDecode[s]` | Decode percent-encoded string |
| `UrlEncode` | `UrlEncode[s]` | Percent-encode string for URLs |
| `UrlFormDecode` | `UrlFormDecode[s]` | Parse form-encoded string to assoc |
| `UrlFormEncode` | `UrlFormEncode[params]` | application/x-www-form-urlencoded from assoc |
| `Using` | `Using[name, opts?]` | Load a package by name with import options |
| `UuidV4` | `UuidV4[]` |  |
| `UuidV7` | `UuidV7[]` |  |
| `Variance` | `Variance[list]` | Variance of list |
| `VectorCount` | `VectorCount[store]` | Count items in store |
| `VectorDelete` | `VectorDelete[store, ids]` | Delete items by ids |
| `VectorReset` | `VectorReset[store]` | Clear all items in store |
| `VectorSearch` | `VectorSearch[store, query, opts]` | Search by vector or text (hybrid supported) |
| `VectorStore` | `VectorStore[optsOrDsn]` | Create/open a vector store (memory or DSN) |
| `VectorUpsert` | `VectorUpsert[store, rows]` | Insert or update vectors with metadata |
| `VersionBump` | `VersionBump[]` |  |
| `WaitContainer` | `WaitContainer[id, opts?]` | Wait for container to stop |
| `WaitProcess` | `WaitProcess[proc]` | Wait for process to exit |
| `When` | `When[cond, body]` | Evaluate body when condition is True (held) |
| `WhoAmI` | `WhoAmI[]` | Show current registry identity (requires lyra-pm) |
| `With` | `With[<|vars|>, body]` | Lexically bind symbols within a body. |
| `WithColumns` | `WithColumns[ds, defs]` | Add/compute new columns (held) |
| `WithPackage` | `WithPackage[name, expr]` | Temporarily add a path to $PackagePath |
| `WithPolicy` | `WithPolicy[]` |  |
| `Workflow` | `Workflow[steps]` | Run a list of steps sequentially (held) |
| `Wrap` | `Wrap[text, width]` | Wrap text to width |
| `XdgDirs` | `XdgDirs[]` |  |
| `ZipCreate` | `ZipCreate[dest, inputs]` | Create a .zip archive from files/directories. |
| `ZipExtract` | `ZipExtract[src, dest]` | Extract a .zip archive into a directory. |
| `__DatasetFromDbTable` | `__DatasetFromDbTable[]` |  |
| `__SQLToRows` | `__SQLToRows[]` |  |
| `col` | `col[]` |  |

## `AddEdges`

- Usage: `AddEdges[graph, edges]`
- Summary: Add edges with optional keys and weights
- Examples:
  - `AddEdges[g, {{"a","b"}}]`

## `AlignCenter`

- Usage: `AlignCenter[text, width, pad?]`
- Summary: Pad on both sides to center
- Examples:
  - `AlignCenter["ok", 6, "-"]  ==> "--ok--"`

## `AnsiStyle`

- Usage: `AnsiStyle[text, opts?]`
- Summary: Style text with ANSI codes
- Examples:
  - `AnsiStyle["hi", <|"Color"->"green", "Bold"->True|>]`

## `Apply`

- Usage: `Apply[f, list]`
- Summary: Apply head to list elements: Apply[f, {…}]
- Examples:
  - `Apply[Plus, {1,2,3}]  ==> 6`

## `Binomial`

- Usage: `Binomial[n, k]`
- Summary: Binomial coefficient nCk
- Examples:
  - `Binomial[5, 2]  ==> 10`

## `BoundedChannel`

- Usage: `BoundedChannel[n]`
- Summary: Create bounded channel
- Examples:
  - `ch := BoundedChannel[2]; Send[ch, 1]; Receive[ch]  ==> 1`

## `CamelCase`

- Usage: `CamelCase[s]`
- Summary: Convert to camelCase
- Examples:
  - `CamelCase["hello world"]  ==> "helloWorld"`

## `Clip`

- Usage: `Clip[x, min, max]`
- Summary: Clamp value to [min,max]
- Examples:
  - `Clip[10, 0, 5]  ==> 5`

## `Coalesce`

- Usage: `Coalesce[values…]`
- Summary: First non-null value
- Examples:
  - `Coalesce[Null, 0, 42]  ==> 0`

## `Compose`

- Usage: `Compose[f, g, …]`
- Summary: Compose functions left-to-right
- Examples:
  - `Compose[f,g][x]  ==> f[g[x]]`

## `Confirm`

- Usage: `Confirm[text, opts?]`
- Summary: Ask yes/no question (TTY)
- Examples:
  - `Confirm["Proceed?"]  ==> True|False`

## `Count`

- Usage: `Count[list, value|pred]`
- Summary: Count elements equal to value or matching predicate
- Examples:
  - `Count[{1,2,1,1}, 1]  ==> 3`

## `CreateContainer`

- Usage: `CreateContainer[image, opts?]`
- Summary: Create a container
- Examples:
  - `cid := CreateContainer["alpine", <|"cmd"->"echo hi"|>]`
  - `StartContainer[cid]`

## `DateFormat`

- Usage: `DateFormat[dt, fmt?]`
- Summary: Format DateTime or epochMs to string
- Examples:
  - `DateFormat[DateTime[<|"Year"->2024,"Month"->8,"Day"->1|>], "%Y-%m-%d"]  ==> "2024-08-01"`

## `DateTime`

- Usage: `DateTime[spec]`
- Summary: Build/parse DateTime assoc (UTC)
- Examples:
  - `DateTime["2024-08-01T00:00:00Z"]  ==> <|"epochMs"->...|>`

## `DivMod`

- Usage: `DivMod[a, n]`
- Summary: Quotient and remainder
- Examples:
  - `DivMod[7, 3]  ==> {2, 1}`

## `Divide`

- Usage: `Divide[a, b]`
- Summary: Divide two numbers.
- Examples:
  - `Divide[6, 3]  ==> 2`
  - `Divide[7, 2]  ==> 3.5`

## `DurationParse`

- Usage: `DurationParse[s]`
- Summary: Parse human duration (e.g., 1h30m)
- Examples:
  - `DurationParse["2h30m"]  ==> <|...|>`

## `Explain`

- Usage: `Explain[expr]`
- Summary: Explain evaluation; returns trace steps when enabled.
- Examples:
  - `Explain[Plus[1,2]]  ==> <|steps->...|>`

## `Export`

- Usage: `Export[symbols]`
- Summary: Mark symbol(s) as public
- Examples:
  - `Export[{"Foo", "Bar"}]`

## `Factorial`

- Usage: `Factorial[n]`
- Summary: n! (product 1..n)
- Examples:
  - `Factorial[5]  ==> 120`

## `FixedPoint`

- Usage: `FixedPoint[f, x]`
- Summary: Iterate f until convergence
- Examples:
  - `FixedPoint[Cos, 1.0]  ==> 0.739... `

## `FoldList`

- Usage: `FoldList[f, init, list]`
- Summary: Cumulative fold producing intermediates
- Examples:
  - `FoldList[Plus, 0, {1,2,3}]  ==> {0,1,3,6}`

## `Future`

- Usage: `Future[expr]`
- Summary: Create a Future from an expression (held)
- Examples:
  - `f := Future[Range[1,1_000]]; Await[f]  ==> {1,2,...}`

## `GCD`

- Usage: `GCD[a, b, …]`
- Summary: Greatest common divisor
- Examples:
  - `GCD[18, 24]  ==> 6`

## `Gets`

- Usage: `Gets[path?]`
- Summary: Read entire stdin or file as string
- Examples:
  - `Gets["/tmp/hi.txt"]  ==> "hello!"`

## `GitAdd`

- Usage: `GitAdd[paths, opts?]`
- Summary: Stage files for commit
- Examples:
  - `GitAdd["src/main.rs"]  ==> True`

## `GitBranchCreate`

- Usage: `GitBranchCreate[name, opts?]`
- Summary: Create a new branch
- Examples:
  - `GitBranchCreate["feature/x"]  ==> True`

## `GitCommit`

- Usage: `GitCommit[message, opts?]`
- Summary: Create a commit with message
- Examples:
  - `GitCommit["feat: add api"]  ==> <|Sha->..., Message->...|>`

## `GitDiff`

- Usage: `GitDiff[opts?]`
- Summary: Diff against base and optional paths
- Examples:
  - `GitDiff[<|"Base"->"HEAD~1"|>]  ==> "diff..."`

## `GitLog`

- Usage: `GitLog[opts?]`
- Summary: List commits with formatting options
- Examples:
  - `GitLog[<|"Limit"->5|>]  ==> {"<sha>|<author>|...", ...}`

## `GitRoot`

- Usage: `GitRoot[]`
- Summary: Path to repository root (Null if absent)
- Examples:
  - `GitRoot[]  ==> "/path/to/repo" | Null`

## `GitStatus`

- Usage: `GitStatus[opts?]`
- Summary: Status (porcelain) with branch/ahead/behind/changes
- Examples:
  - `GitStatus[]  ==> <|Branch->..., Ahead->0, Behind->0, Changes->{...}|>`

## `GitSwitch`

- Usage: `GitSwitch[name, opts?]`
- Summary: Switch to branch (optionally create)
- Examples:
  - `GitSwitch["feature/x"]  ==> True`

## `GitVersion`

- Usage: `GitVersion[]`
- Summary: Get git client version string
- Examples:
  - `GitVersion[]  ==> "git version ..."`

## `Gunzip`

- Usage: `Gunzip[dataOrPath, opts?]`
- Summary: Gunzip-decompress a string or a .gz file; optionally write to path.
- Examples:
  - `Gunzip[Gzip["hello"]]  ==> "hello"`
  - `Gunzip["/tmp/a.txt.gz", <|"Out"->"/tmp/a.txt"|>]  ==> <|"path"->"/tmp/a.txt", "bytes_written"->...|>`

## `Gzip`

- Usage: `Gzip[dataOrPath, opts?]`
- Summary: Gzip-compress a string or a file; optionally write to path.
- Examples:
  - `Gzip["hello"]  ==> <compressed bytes as string>`
  - `Gzip["/tmp/a.txt", <|"Out"->"/tmp/a.txt.gz"|>]  ==> <|"path"->"/tmp/a.txt.gz", "bytes_written"->...|>`

## `Identity`

- Usage: `Identity[x]`
- Summary: Identity function: returns its argument
- Examples:
  - `Identity[42]  ==> 42`

## `If`

- Usage: `If[cond, then, else?]`
- Summary: Conditional: If[cond, then, else?] (held)
- Examples:
  - `If[1<2, "yes", "no"]  ==> "yes"`

## `KebabCase`

- Usage: `KebabCase[s]`
- Summary: Convert to kebab-case
- Examples:
  - `KebabCase["HelloWorld"]  ==> "hello-world"`

## `LCM`

- Usage: `LCM[a, b, …]`
- Summary: Least common multiple
- Examples:
  - `LCM[6, 8]  ==> 24`

## `Map`

- Usage: `Map[f, list]`
- Summary: Map a function over a list.
- Examples:
  - `Map[ToUpper, {"a", "b"}]  ==> {"A", "B"}`
  - `Map[#^2 &, {1,2,3}]  ==> {1,4,9}`

## `Minus`

- Usage: `Minus[a, b?]`
- Summary: Subtract or unary negate.
- Examples:
  - `Minus[5, 2]  ==> 3`
  - `Minus[5]  ==> -5`

## `Mod`

- Usage: `Mod[a, n]`
- Summary: Modulo remainder ((a mod n) >= 0)
- Examples:
  - `Mod[7, 3]  ==> 1`

## `Nest`

- Usage: `Nest[f, x, n]`
- Summary: Nest function n times: Nest[f, x, n]
- Examples:
  - `Nest[#*2 &, 1, 3]  ==> 8`

## `NewModule`

- Usage: `NewModule[pkgPath, name]`
- Summary: Scaffold a new module file in a package
- Examples:
  - `NewModule["./packages/acme.tools", "Util"]  ==> ".../src/Util.lyra"`

## `NewPackage`

- Usage: `NewPackage[name, opts?]`
- Summary: Scaffold a new package directory
- Examples:
  - `NewPackage[<|"Name"->"acme.tools", "Path"->"./packages"|>]  ==> <|path->...|>`

## `NowMs`

- Usage: `NowMs[]`
- Summary: Current UNIX time in milliseconds
- Examples:
  - `NowMs[]  ==> 1710000000000`

## `ParallelMap`

- Usage: `ParallelMap[f, list]`
- Summary: Map in parallel over list
- Examples:
  - `ParallelMap[#^2 &, Range[1,4]]  ==> {1,4,9,16}`

## `Power`

- Usage: `Power[a, b]`
- Summary: Exponentiation (right-associative in parser).
- Examples:
  - `Power[2, 8]  ==> 256`
  - `2^3^2 parses as 2^(3^2)`

## `ProjectDiscover`

- Usage: `ProjectDiscover[start?]`
- Summary: Search upwards for project.lyra
- Examples:
  - `ProjectDiscover[]  ==> "/path/to/project" | Null`

## `ProjectInfo`

- Usage: `ProjectInfo[root?]`
- Summary: Summarize project (name, version, paths)
- Examples:
  - `ProjectInfo[]  ==> <|name->..., version->..., modules->...|>`

## `PullImage`

- Usage: `PullImage[ref, opts?]`
- Summary: Pull an image
- Examples:
  - `PullImage["alpine:latest"]  ==> <|id->..., size->...|>`

## `Puts`

- Usage: `Puts[content, path]`
- Summary: Write string to file (overwrite)
- Examples:
  - `Puts["hello", "/tmp/hi.txt"]  ==> True`

## `PutsAppend`

- Usage: `PutsAppend[content, path]`
- Summary: Append string to file
- Examples:
  - `PutsAppend["!", "/tmp/hi.txt"]  ==> True`

## `RegexFindAll`

- Usage: `RegexFindAll[s, pattern]`
- Summary: Find all regex capture groups
- Examples:
  - `RegexFindAll[\"a1 b22\", \"\\d+\"]  ==> {\"1\",\"22\"}`

## `RegexIsMatch`

- Usage: `RegexIsMatch[s, pattern]`
- Summary: Test if regex matches string
- Examples:
  - `RegexIsMatch[\"abc123\", \"\\d+\"]  ==> True`

## `RegisterTable`

- Usage: `RegisterTable[conn, name, rows]`
- Summary: Register in-memory rows as a table (mock)
- Examples:
  - `rows := {<|"id"->1, "name"->"a"|>, <|"id"->2, "name"->"b"|>}`
  - `conn := Connect["mock://"]; RegisterTable[conn, "t", rows]  ==> True`

## `Replace`

- Usage: `Replace[expr, rules]`
- Summary: Replace first match by rule(s).
- Examples:
  - `Replace[{1,2,1,3}, 1->9]  ==> {9,2,1,3}`
  - `Replace["2024-08-01", DigitCharacter.. -> "#"]`

## `ReplaceAll`

- Usage: `ReplaceAll[expr, rules]`
- Summary: Replace all matches by rule(s).
- Examples:
  - `ReplaceAll[{1,2,1,3}, 1->9]  ==> {9,2,9,3}`
  - `ReplaceAll["a-b-a", "a"->"x"]  ==> "x-b-x"`

## `Round`

- Usage: `Round[x]`
- Summary: Round to nearest integer
- Examples:
  - `Round[2.6]  ==> 3`

## `Schema`

- Usage: `Schema[value]`
- Summary: Return a minimal schema for a value/association.
- Examples:
  - `Schema[<|"a"->1, "b"->"x"|>]  ==> <|a->Integer, b->String|>`

## `Set`

- Usage: `Set[symbol, value]`
- Summary: Assignment: Set[symbol, value].
- Examples:
  - `Set[x, 10]; x  ==> 10`

## `SetDelayed`

- Usage: `SetDelayed[symbol, expr]`
- Summary: Delayed assignment evaluated on use.
- Examples:
  - `SetDelayed[now, CurrentTime[]]; now  ==> 1690000000 (changes)`

## `Sleep`

- Usage: `Sleep[ms]`
- Summary: Sleep for N milliseconds
- Examples:
  - `Sleep[100]  ==> Null`

## `Slugify`

- Usage: `Slugify[s]`
- Summary: Slugify for URLs
- Examples:
  - `Slugify["Hello, World!"]  ==> "hello-world"`

## `SnakeCase`

- Usage: `SnakeCase[s]`
- Summary: Convert to snake_case
- Examples:
  - `SnakeCase["HelloWorld"]  ==> "hello_world"`

## `TarCreate`

- Usage: `TarCreate[dest, inputs, opts?]`
- Summary: Create a .tar (optionally .tar.gz) archive from inputs.
- Examples:
  - `TarCreate["/tmp/bundle.tar", {"/tmp/data"}]  ==> <|"path"->"/tmp/bundle.tar"|>`
  - `TarCreate["/tmp/bundle.tar.gz", {"/tmp/data"}, <|"Gzip"->True|>]  ==> <|"path"->...|>`

## `TarExtract`

- Usage: `TarExtract[src, dest]`
- Summary: Extract a .tar or .tar.gz archive into a directory.
- Examples:
  - `TarExtract["/tmp/bundle.tar", "/tmp/untar"]  ==> <|"path"->"/tmp/untar"|>`

## `Thread`

- Usage: `Thread[expr]`
- Summary: Thread Sequence and lists into arguments.
- Examples:
  - `Thread[f[{1,2}, {3,4}]]  ==> {f[1,3], f[2,4]}`
  - `Thread[Plus[Sequence[{1,2},{3,4}]]]  ==> {4,6}`

## `TitleCase`

- Usage: `TitleCase[s]`
- Summary: Convert to Title Case
- Examples:
  - `TitleCase["hello world"]  ==> "Hello World"`

## `Truncate`

- Usage: `Truncate[text, width, ellipsis?]`
- Summary: Truncate to width with ellipsis
- Examples:
  - `Truncate["abcdef", 4]  ==> "abc…"`

## `TryOr`

- Usage: `TryOr[body, default]`
- Summary: Try body else default (held)
- Examples:
  - `TryOr[1/0, "fallback"]  ==> "fallback"`

## `Unless`

- Usage: `Unless[cond, body]`
- Summary: Evaluate body when condition is False (held)
- Examples:
  - `Unless[False, Print["ok"]]`

## `Unset`

- Usage: `Unset[symbol]`
- Summary: Clear definition: Unset[symbol].
- Examples:
  - `Set[x, 10]; Unset[x]; x  ==> x`

## `UrlEncode`

- Usage: `UrlEncode[s]`
- Summary: Percent-encode string for URLs
- Examples:
  - `UrlEncode["a b"]  ==> "a%20b"`

## `UrlFormEncode`

- Usage: `UrlFormEncode[params]`
- Summary: application/x-www-form-urlencoded from assoc
- Examples:
  - `UrlFormEncode[<|"a"->"b"|>]  ==> "a=b"`

## `Using`

- Usage: `Using[name, opts?]`
- Summary: Load a package by name with import options
- Examples:
  - `Using["lyra/math", <|"Import"->"All"|>]  ==> True`

## `When`

- Usage: `When[cond, body]`
- Summary: Evaluate body when condition is True (held)
- Examples:
  - `When[True, Print["ok"]]`

## `With`

- Usage: `With[<|vars|>, body]`
- Summary: Lexically bind symbols within a body.
- Examples:
  - `With[<|"x"->2|>, x^3]  ==> 8`

## `Workflow`

- Usage: `Workflow[steps]`
- Summary: Run a list of steps sequentially (held)
- Examples:
  - `Workflow[{Print["a"], Print["b"]}]  ==> {Null, Null}`
  - `Workflow[{<|"name"->"echo", "run"->Run["echo", {"hi"}]|>}]  ==> {...}`

## `Wrap`

- Usage: `Wrap[text, width]`
- Summary: Wrap text to width
- Examples:
  - `Wrap["aaaa bbbb cccc", 5]  ==> "aaaa\nbbbb\ncccc"`

## `ZipCreate`

- Usage: `ZipCreate[dest, inputs]`
- Summary: Create a .zip archive from files/directories.
- Examples:
  - `ZipCreate["/tmp/bundle.zip", {"/tmp/a.txt", "/tmp/dir"}]  ==> <|"path"->"/tmp/bundle.zip", ...|>`

## `ZipExtract`

- Usage: `ZipExtract[src, dest]`
- Summary: Extract a .zip archive into a directory.
- Examples:
  - `ZipExtract["/tmp/bundle.zip", "/tmp/unzipped"]  ==> <|"path"->"/tmp/unzipped", "files"->...|>`
