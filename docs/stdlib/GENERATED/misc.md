# MISC

| Function | Usage | Summary |
|---|---|---|
| `ACos` | `ACos[x]` | Arc-cosine (inverse cosine) |
| `ASin` | `ASin[x]` | Arc-sine (inverse sine) |
| `ATan` | `ATan[x]` | Arc-tangent (inverse tangent) |
| `ATan2` | `ATan2[y, x]` | Arc-tangent of y/x (quadrant aware) |
| `ActivationLayer` | `ActivationLayer[name|opts]` | Activation layer (e.g., ReLU, Tanh) |
| `Actor` | `Actor[handler]` | Create actor with handler (held) |
| `Add` | `Add[target, value]` | Add value to a collection (alias of Insert for some types) |
| `AddLayer` | `AddLayer[opts?]` | Elementwise addition layer |
| `AddRegistryAuth` | `AddRegistryAuth[server, user, password]` | Add registry credentials |
| `AlignCenter` | `AlignCenter[text, width, pad?]` | Pad on both sides to center |
| `AlignLeft` | `AlignLeft[text, width, pad?]` | Pad right to width |
| `AlignRight` | `AlignRight[text, width, pad?]` | Pad left to width |
| `AnsiEnabled` | `AnsiEnabled[]` | Are ANSI colors enabled? |
| `AnsiStyle` | `AnsiStyle[text, opts?]` | Style text with ANSI codes |
| `Apart` | `Apart[expr, var?]` | Partial fraction decomposition. |
| `Apply` | `Apply[f, list, level?]` | Apply head to list elements: Apply[f, {…}] |
| `ArgsParse` | `ArgsParse[list]` | Parse CLI-like args to assoc |
| `Ask` | `Ask[actor, msg]` | Request/response with actor (held) |
| `Await` | `Await[future]` | Wait for Future and return value |
| `Bag` | `Bag[]` | Create a multiset bag |
| `BagAdd` | `BagAdd[bag, value]` | Add item to bag |
| `BagCount` | `BagCount[bag, value]` | Count occurrences of value |
| `BagDifference` | `BagDifference[a, b]` | Difference of two bags |
| `BagIntersection` | `BagIntersection[a, b]` | Intersection of two bags |
| `BagRemove` | `BagRemove[bag, value]` | Remove one item from bag |
| `BagSize` | `BagSize[bag]` | Total items in bag |
| `BagUnion` | `BagUnion[a, b]` | Union of two bags |
| `BarChart` | `BarChart[data, opts]` | Render a bar chart |
| `BatchNorm` | `BatchNorm[opts?]` | Batch normalization layer |
| `BatchNormLayer` | `BatchNormLayer[opts?]` | Batch normalization layer |
| `Bernoulli` | `Bernoulli[p]` | Bernoulli distribution head (probability p). |
| `Binomial` | `Binomial[n, k]` | Binomial coefficient nCk |
| `BinomialDistribution` | `BinomialDistribution[n, p]` | Binomial distribution head (trials n, prob p). |
| `BlankQ` | `BlankQ[s]` | Alias: IsBlank string predicate |
| `BooleanQ` | `BooleanQ[x]` | Is value Boolean? |
| `BoundedChannel` | `BoundedChannel[n]` | Create bounded channel |
| `BoxText` | `BoxText[text, opts?]` | Draw a box around text |
| `BuildImage` | `BuildImage[context, opts?]` | Build image from context |
| `BuildPackage` | `BuildPackage[path?, opts?]` | Build a package (requires lyra-pm) |
| `BusyWait` | `BusyWait[ms]` | Block for n milliseconds (testing only) |
| `CDF` | `CDF[dist, x]` | Cumulative distribution for a distribution at x. |
| `CamelCase` | `CamelCase[s]` | Convert to camelCase |
| `Cancel` | `Cancel[future]` | Request cooperative cancellation |
| `CancelRational` | `CancelRational[expr]` | Cancel common factors in a rational expression. |
| `CancelScope` | `CancelScope[scope]` | Cancel running scope |
| `Capitalize` | `Capitalize[s]` | Capitalize first letter |
| `Catch` | `Catch[body, handlers]` | Catch a thrown value (held) |
| `CausalSelfAttention` | `CausalSelfAttention[opts?]` | Self-attention with causal mask |
| `Ceiling` | `Ceiling[x]` | Smallest integer >= x |
| `CellCreate` | `CellCreate[type, input, opts?]` | Create a new cell association |
| `CellDelete` | `CellDelete[notebook, id]` | Delete a cell by UUID |
| `CellInsert` | `CellInsert[notebook, cell, pos]` | Insert a cell into a notebook at position |
| `CellMove` | `CellMove[notebook, id, toIndex]` | Move a cell to index |
| `CellUpdate` | `CellUpdate[notebook, id, updates]` | Update fields on a cell by UUID |
| `Cells` | `Cells[notebook]` | Get list of cells in a notebook |
| `ChangelogGenerate` | `ChangelogGenerate[range?]` | Generate CHANGELOG entries from git log. |
| `Chart` | `Chart[spec, opts]` | Render a chart from a spec |
| `Chat` | `Chat[model?, opts]` | Chat completion with messages; supports tools and streaming. |
| `ChineseRemainder` | `ChineseRemainder[residues, moduli]` | Solve x ≡ r_i (mod m_i) for coprime moduli. |
| `Cholesky` | `Cholesky[A]` | Cholesky factorization for SPD matrices |
| `Citations` | `Citations[matchesOrAnswer]` | Normalize citations from matches or answer. |
| `Cite` | `Cite[matchesOrAnswer, opts?]` | Format citations from retrieval matches or answers. |
| `Classifier` | `Classifier[opts?]` | Create classifier spec (Logistic/Baseline) |
| `Classify` | `Classify[data, opts]` | Train a classifier (baseline/logistic) |
| `ClassifyMeasurements` | `ClassifyMeasurements[model, data, opts]` | Evaluate classifier metrics |
| `ClearOutputs` | `ClearOutputs[notebook]` | Clear output of code cells |
| `Clip` | `Clip[x, min, max]` | Clamp value to [min,max]. Tensor-aware: elementwise on tensors. |
| `Close` | `Close[handle]` | Close an open handle (cursor, channel) |
| `CloseChannel` | `CloseChannel[ch]` | Close channel |
| `ClosenessCentrality` | `ClosenessCentrality[graph]` | Per-node closeness centrality |
| `Cluster` | `Cluster[data, opts]` | Cluster points (prototype) |
| `Clusterer` | `Clusterer[opts?]` | Create clusterer spec |
| `CollectTerms` | `CollectTerms[expr]` | Collect like terms in a sum. |
| `CollectTermsBy` | `CollectTermsBy[expr, by]` | Collect terms by function or key. |
| `Columnize` | `Columnize[lines, opts?]` | Align lines in columns |
| `Combinations` | `Combinations[listOrN, k]` | All k-combinations (subsets) of a list or 1..n. |
| `CombinationsCount` | `CombinationsCount[n, k]` | Count of combinations; Binomial[n,k]. |
| `Complete` | `Complete[model?, opts|prompt]` | Text completion from prompt or options. |
| `Compose` | `Compose[f, g, …]` | Compose functions left-to-right |
| `ConditionNumber` | `ConditionNumber[A]` | Estimated 2-norm condition number using power iterations on A^T A. |
| `ConfigFind` | `ConfigFind[names?, startDir?]` | Search upwards for config files (e.g., .env, lyra.toml). |
| `ConfigLoad` | `ConfigLoad[opts?]` | Load project config and environment. |
| `Confirm` | `Confirm[text, opts?]` | Ask yes/no question (TTY) |
| `ConstantFunction` | `ConstantFunction[c]` | Constant function returning c |
| `Container` | `Container[image, opts?]` | Create a container |
| `ContainersClose` | `ContainersClose[]` | Close open fetch handles |
| `ContainersFetch` | `ContainersFetch[paths, opts?]` | Fetch external resources for build |
| `Contains` | `Contains[container, item]` | Membership test for strings/lists/sets/assocs |
| `ContainsKeyQ` | `ContainsKeyQ[subject, key]` | Key membership for assoc/rows/Dataset/Frame |
| `ContainsQ` | `ContainsQ[container, item]` | Alias: membership predicate |
| `ConvTranspose2D` | `ConvTranspose2D[opts?]` | Transposed 2D convolution (deconv) |
| `Convolution1D` | `Convolution1D[opts?]` | 1D convolution layer |
| `Convolution2D` | `Convolution2D[opts?]` | 2D convolution layer (uses InputChannels/Height/Width for forward) |
| `ConvolutionLayer` | `ConvolutionLayer[outChannels, kernelSize, opts?]` | 1D convolution layer |
| `Convolve` | `Convolve[a, b, mode?]` | Linear convolution of two sequences. Modes: Full\|Same\|Valid. |
| `CoprimeQ` | `CoprimeQ[a, b]` | Predicate: are integers coprime? |
| `Cos` | `Cos[x]` | Cosine (radians). Tensor-aware: elementwise on tensors. |
| `CostAdd` | `CostAdd[amount]` | Add delta to accumulated USD cost; returns total. |
| `CostSoFar` | `CostSoFar[]` | Return accumulated USD cost. |
| `Count` | `Count[x]` | Count items/elements (lists, assocs, Bag/VectorStore) |
| `CrossAttention` | `CrossAttention[opts?]` | Cross-attention over Memory (seq x dim) |
| `CrossValidate` | `CrossValidate[obj, data, opts?]` | Cross-validate estimator + data (dispatched) |
| `CurrentModule` | `CurrentModule[]` | Current module path/name |
| `D` | `D[expr, var]` | Differentiate expression w.r.t. variable. |
| `DateDiff` | `DateDiff[a, b]` | Difference between two DateTime in ms |
| `DegreeCentrality` | `DegreeCentrality[graph]` | Per-node degree centrality |
| `Delete` | `Delete[assoc, keys]` | Delete keys from association |
| `Dense` | `Dense[opts?]` | Linear (fully-connected) layer |
| `DepthwiseConv2D` | `DepthwiseConv2D[opts?]` | Depthwise 2D convolution (per-channel) |
| `Dequeue` | `Dequeue[queue]` | Dequeue value |
| `Describe` | `Describe[name, items, opts?]` | Define a test suite (held). |
| `DescribeBuiltins` | `DescribeBuiltins[]` | List builtins with attributes (and docs when available). |
| `DescribeContainers` | `DescribeContainers[]` | Describe available container APIs |
| `Determinant` | `Determinant[A]` | Determinant of a square matrix (partial pivoting). |
| `DiagMatrix` | `DiagMatrix[v]` | Diagonal matrix from a vector. |
| `Diagonal` | `Diagonal[A]` | Main diagonal of a matrix as a vector. |
| `Difference` | `Difference[a, b]` | Difference for lists or sets (dispatched) |
| `DimensionReduce` | `DimensionReduce[data, opts]` | Reduce dimensionality (PCA-like) |
| `DisconnectContainers` | `DisconnectContainers[]` | Disconnect from container runtime |
| `DivMod` | `DivMod[a, n]` | Quotient and remainder |
| `Divide` | `Divide[a, b]` | Divide two numbers. |
| `DividesQ` | `DividesQ[a, b]` | Predicate: does a divide b? |
| `Documentation` | `Documentation[name]` | Documentation card for a builtin. |
| `Dot` | `Dot[a, b]` | Matrix multiplication and vector dot product (type-dispatched). |
| `DotenvLoad` | `DotenvLoad[path?, opts?]` | Load .env variables into process env. |
| `DropGraph` | `DropGraph[graph]` | Drop a graph handle |
| `DropoutLayer` | `DropoutLayer[p|opts]` | Dropout layer |
| `EigenDecomposition` | `EigenDecomposition[A]` | Eigenvalues and eigenvectors. Symmetric: Jacobi; general: real QR + inverse iteration. |
| `Embed` | `Embed[opts]` | Compute embeddings for text using a provider. |
| `Embedding` | `Embedding[vocab, dim]` | Embedding lookup layer |
| `EmbeddingLayer` | `EmbeddingLayer[vocab, dim, opts?]` | Embedding lookup layer |
| `EmptyQ` | `EmptyQ[x]` | Is the subject empty? (lists, strings, assocs, handles) |
| `EndModule` | `EndModule[]` | End current module scope |
| `EndScope` | `EndScope[scope]` | End scope and release resources |
| `EndsWithQ` | `EndsWithQ[s, suffix]` | Alias: EndsWith predicate |
| `Enqueue` | `Enqueue[queue, value]` | Enqueue value |
| `EnvExpand` | `EnvExpand[text, opts?]` | Expand $VAR or %VAR% style environment variables in text. |
| `EqualQ` | `EqualQ[a, b]` | Structural equality for sets and handles |
| `EqualsIgnoreCase` | `EqualsIgnoreCase[a, b]` | Case-insensitive string equality |
| `Estimator` | `Estimator[opts]` | Create ML estimator spec (Task/Method) |
| `EulerPhi` | `EulerPhi[n]` | Euler's totient function: count of 1<=k<=n with CoprimeQ[k,n]. |
| `Evaluate` | `Evaluate[model, data, opts?]` | Evaluate an ML model on data (dispatched) |
| `Events` | `Events[opts?]` | Subscribe to runtime events |
| `Exp` | `Exp[x]` | Natural exponential e^x. Tensor-aware: elementwise on tensors. |
| `Expand` | `Expand[expr]` | Distribute products over sums once. |
| `ExpandAll` | `ExpandAll[expr]` | Fully expand products over sums. |
| `Explain` | `Explain[expr]` | Explain evaluation; returns trace steps when enabled. |
| `ExplainContainers` | `ExplainContainers[]` | Explain container runtime configuration |
| `Exponential` | `Exponential[lambda]` | Exponential distribution head (rate λ). |
| `ExportImages` | `ExportImages[refs, path]` | Export images to an archive |
| `Exported` | `Exported[symbols]` | Mark symbols as exported from current module. |
| `ExtendedGCD` | `ExtendedGCD[a, b]` | Extended GCD: returns {g, x, y} with a x + b y = g. |
| `FFN` | `FFN[opts?]` | Position-wise feed-forward (supports SwiGLU/GEGLU) |
| `FFT` | `FFT[x, n?]` | Discrete Fourier transform of a 1D sequence (returns Complex list). |
| `Factor` | `Factor[expr]` | Factor a polynomial expression. |
| `FactorInteger` | `FactorInteger[n]` | Prime factorization as {{p1,e1},{p2,e2},…}. |
| `Factorial` | `Factorial[n]` | n! (product 1..n) |
| `Fail` | `Fail[message?]` | Construct a failure value (optionally with message) |
| `FeatureExtract` | `FeatureExtract[data, opts]` | Learn preprocessing (impute/encode/standardize) |
| `Figure` | `Figure[items, opts]` | Compose multiple charts in a grid |
| `Finally` | `Finally[body, cleanup]` | Ensure cleanup runs (held) |
| `FindByTag` | `FindByTag[notebook, tag]` | Find cell ids by tag |
| `FindCells` | `FindCells[notebook, spec]` | Find cell ids by spec |
| `Fit` | `Fit[net, data, opts?]` | Train a network (alias of NetTrain) |
| `FixedPoint` | `FixedPoint[f, x]` | Iterate f until convergence |
| `FixedPointList` | `FixedPointList[f, x]` | List of iterates until convergence |
| `Flatten` | `Flatten[list, levels?]` | Flatten by levels (default 1) |
| `FlattenLayer` | `FlattenLayer[opts?]` | Flatten leading dims to vector |
| `Floor` | `Floor[x]` | Largest integer <= x |
| `FoldList` | `FoldList[f, init, list]` | Cumulative fold producing intermediates |
| `FormatDate` | `FormatDate[dt, fmt]` | Format DateTime with strftime pattern |
| `FormatLyra` | `FormatLyra[x]` | Format Lyra from text or file path. |
| `FormatLyraFile` | `FormatLyraFile[path]` | Format a Lyra source file in place. |
| `FormatLyraText` | `FormatLyraText[text]` | Format Lyra source text (pretty printer). |
| `FrameColumns` | `FrameColumns[frame]` | List column names for a Frame |
| `FrameDescribe` | `FrameDescribe[frame, opts?]` | Quick stats by columns |
| `FrameDistinct` | `FrameDistinct[frame, cols?]` | Distinct rows in Frame (optional columns) |
| `FrameFilter` | `FrameFilter[frame, pred]` | Filter rows in a Frame |
| `FrameFromRows` | `FrameFromRows[rows]` | Create a Frame from assoc rows |
| `FrameHead` | `FrameHead[frame, n?]` | Take first n rows from Frame |
| `FrameJoin` | `FrameJoin[left, right, on?, opts?]` | Join two Frames by keys |
| `FrameOffset` | `FrameOffset[frame, n]` | Skip first n rows of Frame |
| `FrameSelect` | `FrameSelect[frame, spec]` | Select/compute columns in Frame |
| `FrameSort` | `FrameSort[frame, by]` | Sort Frame by columns |
| `FrameTail` | `FrameTail[frame, n?]` | Take last n rows from Frame |
| `FrameUnion` | `FrameUnion[frames…]` | Union Frames by columns (schema union) |
| `Future` | `Future[expr]` | Create a Future from an expression (held) |
| `GCD` | `GCD[a, b, …]` | Greatest common divisor |
| `Gamma` | `Gamma[k, theta]` | Gamma distribution head (shape k, scale θ). |
| `Gather` | `Gather[futures]` | Await Futures in same structure |
| `Gelu` | `Gelu[x?]` | GELU activation (tanh approx): tensor op or zero-arg layer |
| `GenerateSBOM` | `GenerateSBOM[path?, opts?]` | Generate SBOM (requires lyra-pm) |
| `Get` | `Get[subject, key, default?]` | Get value by key/index from a structure (dispatch) |
| `GetDownValues` | `GetDownValues[symbol]` | Return DownValues for a symbol |
| `GetOwnValues` | `GetOwnValues[symbol]` | Return OwnValues for a symbol |
| `GetSubValues` | `GetSubValues[symbol]` | Return SubValues for a symbol |
| `GetUiBackend` | `GetUiBackend[]` | Get current UI backend mode |
| `GetUiTheme` | `GetUiTheme[]` | Get current UI theme |
| `GetUpValues` | `GetUpValues[symbol]` | Return UpValues for a symbol |
| `Gets` | `Gets[path?]` | Read entire stdin or file as string |
| `GitAdd` | `GitAdd[paths, opts?]` | Stage files for commit |
| `GitApply` | `GitApply[patch, opts?]` | Apply a patch (or check only) |
| `GitBranch` | `GitBranch[name, opts?]` | Create a new branch |
| `GitBranchList` | `GitBranchList[]` | List local branches |
| `GitCommit` | `GitCommit[message, opts?]` | Create a commit with message |
| `GitCurrentBranch` | `GitCurrentBranch[]` | Current branch name |
| `GitDiff` | `GitDiff[opts?]` | Diff against base and optional paths |
| `GitEnsureRepo` | `GitEnsureRepo[opts?]` | Ensure Cwd is a git repo (init if needed) |
| `GitFeatureBranch` | `GitFeatureBranch[opts?]` | Create and switch to a feature branch |
| `GitFetch` | `GitFetch[remote?]` | Fetch from remote |
| `GitInit` | `GitInit[opts?]` | Initialize a new git repository |
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
| `GroupNorm` | `GroupNorm[opts?]` | Group normalization over channels (NumGroups) |
| `HasEdge` | `HasEdge[graph, spec]` | Does graph contain edge? |
| `HasKeyQ` | `HasKeyQ[subject, key]` | Alias: key membership predicate |
| `HasNode` | `HasNode[graph, id]` | Does graph contain node? |
| `HashSet` | `HashSet[values]` | Create a set from values |
| `Head` | `Head[ds, n]` | Take first n rows |
| `Histogram` | `Histogram[data, opts]` | Render a histogram |
| `HybridSearch` | `HybridSearch[store, query, opts?]` | Combine keyword and vector search for retrieval. |
| `IFFT` | `IFFT[X]` | Inverse DFT of a 1D sequence. |
| `IdempotencyKey` | `IdempotencyKey[]` | Generate a unique idempotency key. |
| `Identity` | `Identity[x]` | Identity function: returns its argument |
| `If` | `If[cond, then, else?]` | Conditional: If[cond, then, else?] (held) |
| `ImageHistory` | `ImageHistory[ref]` | Show history/metadata for an image. |
| `ImportedSymbols` | `ImportedSymbols[]` | Assoc of package -> imported symbols |
| `InScope` | `InScope[scope, body]` | Run body inside a scope (held) |
| `IncidentEdges` | `IncidentEdges[graph, id, opts?]` | Edges incident to a node |
| `IndexOf` | `IndexOf[s, substr, from?]` | Index of substring (0-based; -1 if not found) |
| `Info` | `Info[target]` | Information about a handle (Logger, Graph, etc.) |
| `Initializer` | `Initializer[opts?]` | Initializer spec for layer parameters |
| `InspectContainer` | `InspectContainer[id]` | Inspect container |
| `InspectImage` | `InspectImage[ref]` | Inspect image details |
| `InspectNetwork` | `InspectNetwork[name]` | Inspect network |
| `InspectRegistryImage` | `InspectRegistryImage[ref, opts?]` | Inspect remote registry image |
| `InspectVolume` | `InspectVolume[name]` | Inspect volume |
| `InstallPackage` | `InstallPackage[name, opts?]` | Install a package (requires lyra-pm) |
| `IntegerQ` | `IntegerQ[x]` | Is value an integer? |
| `Intersection` | `Intersection[args]` | Intersection for lists or sets (dispatched) |
| `Inverse` | `Inverse[A]` | Inverse of a square matrix (Gauss–Jordan with pivoting). |
| `Invert` | `Invert[assoc]` | Invert mapping values -> list of keys |
| `IsBlank` | `IsBlank[s]` | True if string is empty or whitespace |
| `It` | `It[name, body, opts?]` | Define a test case (held). |
| `KebabCase` | `KebabCase[s]` | Convert to kebab-case |
| `LCM` | `LCM[a, b, …]` | Least common multiple |
| `LU` | `LU[A]` | LU factorization: returns <\|L,U,P\|> |
| `LastIndexOf` | `LastIndexOf[s, substr, from?]` | Last index of substring (0-based; -1 if not found) |
| `LayerNorm` | `LayerNorm[opts?]` | Layer normalization layer |
| `LayerNormLayer` | `LayerNormLayer[opts?]` | Layer normalization layer |
| `Length` | `Length[x]` | Length of a list or string. |
| `LimitRows` | `LimitRows[ds, n]` | Limit number of rows |
| `LinePlot` | `LinePlot[data, opts]` | Render a line plot |
| `LinearLayer` | `LinearLayer[out|opts]` | Fully-connected linear layer |
| `LinearSolve` | `LinearSolve[A, b]` | Solve linear system A x = b (SPD via Cholesky; otherwise LU). |
| `LintLyra` | `LintLyra[x]` | Lint Lyra from text or file path. |
| `LintLyraFile` | `LintLyraFile[path]` | Lint a Lyra source file; returns diagnostics. |
| `LintLyraText` | `LintLyraText[text]` | Lint Lyra source text; returns diagnostics. |
| `LintPackage` | `LintPackage[path?, opts?]` | Lint a package (requires lyra-pm) |
| `ListContainers` | `ListContainers[opts?]` | List containers |
| `ListImages` | `ListImages[opts?]` | List local images |
| `ListInstalledPackages` | `ListInstalledPackages[]` | List packages available on $PackagePath |
| `ListNetworks` | `ListNetworks[]` | List networks |
| `ListQ` | `ListQ[x]` | Is value a list? |
| `ListRegistryAuth` | `ListRegistryAuth[]` | List stored registry credentials |
| `ListVolumes` | `ListVolumes[]` | List volumes |
| `LoadImage` | `LoadImage[path]` | Load image from tar |
| `LoadedPackages` | `LoadedPackages[]` | Assoc of loaded packages |
| `LocalClustering` | `LocalClustering[graph]` | Per-node clustering coefficient |
| `MLApply` | `MLApply[model, x, opts]` | Apply a trained model to input |
| `MLCrossValidate` | `MLCrossValidate[data, opts]` | Cross-validate with simple split |
| `MLProperty` | `MLProperty[model, prop]` | Inspect trained model properties |
| `MLTune` | `MLTune[data, opts]` | Parameter sweep with basic scoring |
| `Map` | `Map[f, list]` | Map a function over a list. |
| `MapAsync` | `MapAsync[f, list]` | Map to Futures over list |
| `MapKeyValues` | `MapKeyValues[f, assoc]` | Map over (k,v) pairs to new values or pairs |
| `MapKeys` | `MapKeys[f, assoc]` | Map keys with f[k] |
| `MapPairs` | `MapPairs[f, assoc]` | Map over (k,v) pairs |
| `MapValues` | `MapValues[f, assoc]` | Map values with f[v] |
| `MatchQ` | `MatchQ[expr, pattern]` | Pattern match predicate (held) |
| `Mean` | `Mean[list]` | Arithmetic mean of list |
| `Median` | `Median[list]` | Median of list |
| `MemberQ` | `MemberQ[container, item]` | Alias: membership predicate (Contains) |
| `Metrics` | `Metrics[]` | Return counters for tools/models/tokens/cost. |
| `MetricsReset` | `MetricsReset[]` | Reset metrics counters to zero. |
| `Minus` | `Minus[a, b?]` | Subtract or unary negate. |
| `MobiusMu` | `MobiusMu[n]` | Möbius mu function: 0 if non-square-free, else (-1)^k. |
| `Mod` | `Mod[a, n]` | Modulo remainder ((a mod n) >= 0) |
| `ModInverse` | `ModInverse[a, m]` | Multiplicative inverse of a modulo m (if coprime). |
| `Model` | `Model[id|spec]` | Construct a model handle by id or spec. |
| `ModelsList` | `ModelsList[]` | List available model providers/ids. |
| `ModuleInfo` | `ModuleInfo[]` | Information about the current module (path, package). |
| `ModulePath` | `ModulePath[]` | Get module search path |
| `MulLayer` | `MulLayer[opts?]` | Elementwise multiplication layer |
| `MultiHeadAttention` | `MultiHeadAttention[opts?]` | Self-attention with NumHeads (single-batch) |
| `NegativeQ` | `NegativeQ[x]` | Is number < 0? |
| `Nest` | `Nest[f, x, n]` | Nest function n times: Nest[f, x, n] |
| `NestList` | `NestList[f, x, n]` | Nest and collect intermediate values |
| `NetApply` | `NetApply[net, x, opts?]` | Apply network to input(s) |
| `NetChain` | `NetChain[layers, opts?]` | Compose layers sequentially |
| `NetDecoder` | `NetDecoder[spec|auto]` | Construct an output decoder |
| `NetEncoder` | `NetEncoder[spec|auto]` | Construct an input encoder |
| `NetGraph` | `NetGraph[nodes, edges, opts?]` | Construct a simple network graph from layers and edges. |
| `NetInitialize` | `NetInitialize[net, opts?]` | Initialize a network (weights/state) |
| `NetProperty` | `NetProperty[net, key]` | Get network property |
| `NetSummary` | `NetSummary[net]` | Human-readable network summary |
| `NetTrain` | `NetTrain[net, data, opts?]` | Train a network on data |
| `Network` | `Network[opts?]` | Create network |
| `NewModule` | `NewModule[pkgPath, name]` | Scaffold a new module file in a package |
| `NewPackage` | `NewPackage[name, opts?]` | Scaffold a new package directory |
| `NextPrime` | `NextPrime[n]` | Next prime greater than n (or 2 if n<2). |
| `NonEmptyQ` | `NonEmptyQ[x]` | Is list/string/assoc non-empty? |
| `NonNegativeQ` | `NonNegativeQ[x]` | Is number >= 0? |
| `NonPositiveQ` | `NonPositiveQ[x]` | Is number <= 0? |
| `Norm` | `Norm[x, p?]` | Vector p-norms; for matrices: Frobenius by default, 2-norm via SVD when p==2. |
| `Normal` | `Normal[mu, sigma]` | Normal distribution head (mean μ, stddev σ). |
| `NotebookCreate` | `NotebookCreate[]` | Create a new notebook association |
| `NotebookMetadata` | `NotebookMetadata[notebook]` | Get notebook-level metadata |
| `NotebookRead` | `NotebookRead[path]` | Read a .lynb file into an association |
| `NotebookSetMetadata` | `NotebookSetMetadata[notebook, updates]` | Set notebook-level metadata |
| `NotebookValidate` | `NotebookValidate[notebook]` | Validate notebook structure |
| `NotebookWrite` | `NotebookWrite[notebook, path, opts?]` | Write a notebook association to .lynb |
| `Notify` | `Notify[text, opts?]` | Show a notification/message to the user |
| `NthRoot` | `NthRoot[x, n]` | Principal nth root of a number. |
| `NumberQ` | `NumberQ[x]` | Is value numeric (int/real)? |
| `Offset` | `Offset[ds, n]` | Skip first n rows |
| `OnFailure` | `OnFailure[body, handler]` | Handle Failure values (held) |
| `OpenApiGenerate` | `OpenApiGenerate[routes, opts?]` | Generate OpenAPI from routes |
| `PDF` | `PDF[dist, x]` | Probability density/mass for a distribution at x. |
| `PQInsert` | `PQInsert[pq, priority, value]` | Insert with priority |
| `PQPeek` | `PQPeek[pq]` | Peek min (or max) priority |
| `PQPop` | `PQPop[pq]` | Pop min (or max) priority |
| `PackPackage` | `PackPackage[path?, opts?]` | Pack artifacts for distribution (requires lyra-pm) |
| `PackageAudit` | `PackageAudit[path?, opts?]` | Audit dependencies (requires lyra-pm) |
| `PackageExports` | `PackageExports[name]` | Get exports list for a package |
| `PackageInfo` | `PackageInfo[name]` | Read package metadata (name, version, path) |
| `PackagePath` | `PackagePath[]` | Get current $PackagePath |
| `PackageVerify` | `PackageVerify[path?, opts?]` | Verify signatures (requires lyra-pm) |
| `PackageVersion` | `PackageVersion[pkgPath]` | Read version from manifest |
| `ParallelEvaluate` | `ParallelEvaluate[exprs, opts?]` | Evaluate expressions concurrently (held) |
| `ParallelMap` | `ParallelMap[f, list]` | Map in parallel over list |
| `ParallelTable` | `ParallelTable[exprs]` | Evaluate list of expressions in parallel (held) |
| `ParseDate` | `ParseDate[s]` | Parse date string into DateTime |
| `PasswordPrompt` | `PasswordPrompt[text, opts?]` | Prompt for password without echo |
| `PatchEmbedding2D` | `PatchEmbedding2D[opts?]` | 2D to tokens via patch conv |
| `PathExtname` | `PathExtname[path]` | File extension without dot |
| `PathNormalize` | `PathNormalize[path]` | Normalize path separators |
| `PathRelative` | `PathRelative[base, path]` | Relative path from base |
| `PathResolve` | `PathResolve[base, path]` | Resolve against base directory |
| `PatternQ` | `PatternQ[expr]` | Is value a pattern? (held) |
| `PauseContainer` | `PauseContainer[id]` | Pause a container |
| `Peek` | `Peek[handle]` | Peek top of stack/queue |
| `Permutations` | `Permutations[listOrN, k?]` | All permutations (or k-permutations) of a list or range 1..n. |
| `PermutationsCount` | `PermutationsCount[n, k?]` | Count of permutations; n! or nPk. |
| `PingContainers` | `PingContainers[]` | Check if container engine is reachable. |
| `Poisson` | `Poisson[lambda]` | Poisson distribution head (rate λ). |
| `Pooling` | `Pooling[kind, size, opts?]` | Pooling layer (Max/Avg) |
| `Pooling2D` | `Pooling2D[kind, size, opts?]` | 2D pooling layer (Max/Avg; requires InputChannels/Height/Width) |
| `PoolingLayer` | `PoolingLayer[kind, size, opts?]` | 1D pooling layer (Max/Avg) |
| `Pop` | `Pop[stack]` | Pop from stack |
| `PositionalEmbedding` | `PositionalEmbedding[opts?]` | Learnable positional embeddings (adds to input) |
| `PositionalEncoding` | `PositionalEncoding[opts?]` | Sinusoidal positional encoding (adds to input) |
| `PositiveQ` | `PositiveQ[x]` | Is number > 0? |
| `Power` | `Power[a, b]` | Exponentiation (right-associative). Tensor-aware: elementwise when any tensor. |
| `PowerMod` | `PowerMod[a, b, m]` | Modular exponentiation: a^b mod m (supports negative b if invertible). |
| `Predict` | `Predict[obj, x, opts?]` | Predict using a network or estimator (dispatched) |
| `PredictMeasurements` | `PredictMeasurements[model, data, opts]` | Evaluate regressor metrics |
| `PrimeFactors` | `PrimeFactors[n]` | List of prime factors with multiplicity. |
| `PrimeQ` | `PrimeQ[n]` | Predicate: is integer prime? |
| `PriorityQueue` | `PriorityQueue[]` | Create a priority queue |
| `Private` | `Private[symbols]` | Mark symbol(s) as private |
| `ProgressAdvance` | `ProgressAdvance[id, n?]` | Advance progress bar by n (default 1). |
| `ProgressBar` | `ProgressBar[total]` | Create a progress bar; returns id. |
| `ProgressFinish` | `ProgressFinish[id]` | Finish and remove a progress bar. |
| `ProjectDiscover` | `ProjectDiscover[start?]` | Search upwards for project.lyra |
| `ProjectInfo` | `ProjectInfo[root?]` | Summarize project (name, version, paths) |
| `ProjectInit` | `ProjectInit[path, opts?]` | Initialize new project (scaffold) |
| `ProjectLoad` | `ProjectLoad[root?]` | Load and evaluate project.lyra (normalized) |
| `ProjectRoot` | `ProjectRoot[]` | Return project root path (or Null) |
| `ProjectValidate` | `ProjectValidate[root?]` | Validate project manifest and structure |
| `Prompt` | `Prompt[text, opts?]` | Prompt user for input (TTY) |
| `PromptSelect` | `PromptSelect[text, items, opts?]` | Prompt user to select one item from a list. |
| `PromptSelectMany` | `PromptSelectMany[text, items, opts?]` | Prompt user to select many items from a list |
| `Property` | `Property[obj, key]` | Property of a network or ML model (dispatch) |
| `PruneImages` | `PruneImages[opts?]` | Remove unused images |
| `PseudoInverse` | `PseudoInverse[A]` | Moore–Penrose pseudoinverse via reduced SVD (V S^+ U^T). |
| `PublishPackage` | `PublishPackage[path?, opts?]` | Publish to registry (requires lyra-pm) |
| `PullImage` | `PullImage[ref, opts?]` | Pull an image |
| `Push` | `Push[stack, value]` | Push onto stack |
| `PushImage` | `PushImage[ref, opts?]` | Push an image to registry |
| `Puts` | `Puts[content, path]` | Write string to file (overwrite) |
| `PutsAppend` | `PutsAppend[content, path]` | Append string to file |
| `QR` | `QR[A, opts?]` | QR decomposition via Householder reflections. Use "Reduced" option for thin Q,R. |
| `Queue` | `Queue[]` | Create a FIFO queue |
| `Quotient` | `Quotient[a, n]` | Integer division quotient |
| `RAGAnswer` | `RAGAnswer[store, query, opts?]` | Answer a question using retrieved context and a model. |
| `RAGAssembleContext` | `RAGAssembleContext[matches, opts?]` | Assemble a context string from matches. |
| `RAGChunk` | `RAGChunk[text, opts?]` | Split text into overlapping chunks for indexing. |
| `RAGIndex` | `RAGIndex[store, docs, opts?]` | Embed and upsert documents into a vector store. |
| `RAGRetrieve` | `RAGRetrieve[store, query, opts?]` | Retrieve similar chunks for a query. |
| `RMSNorm` | `RMSNorm[opts?]` | RMS normalization (seq x dim) |
| `RandomInteger` | `RandomInteger[spec?]` | Random integer; supports {min,max}. |
| `RandomReal` | `RandomReal[spec?]` | Random real; supports {min,max}. |
| `RandomVariate` | `RandomVariate[dist, n?]` | Sample from a distribution (optionally n samples). |
| `Rank` | `Rank[A]` | Numerical matrix rank via reduced QR (tolerance-based). |
| `RealQ` | `RealQ[x]` | Is value a real number? |
| `Recall` | `Recall[session, query?, opts?]` | Return recent items from session (with optional query). |
| `Receive` | `Receive[ch, opts?]` | Receive value from channel |
| `RegexIsMatch` | `RegexIsMatch[s, pattern]` | Test if regex matches string |
| `RegisterExports` | `RegisterExports[name, exports]` | Register exports for a package (internal) |
| `Regressor` | `Regressor[opts?]` | Create regressor spec (Linear/Baseline) |
| `ReleaseTag` | `ReleaseTag[version, opts?]` | Create annotated git tag (and optionally push). |
| `ReloadPackage` | `ReloadPackage[name]` | Reload a package |
| `Relu` | `Relu[x]` | Rectified Linear Unit: max(0, x). Tensor-aware: elementwise on tensors. |
| `Remainder` | `Remainder[a, n]` | Integer division remainder |
| `Remember` | `Remember[session, item]` | Append item to named session buffer. |
| `RenameContainer` | `RenameContainer[id, name]` | Rename a container |
| `RenameKeys` | `RenameKeys[assoc, map|f]` | Rename keys by mapping or function |
| `Replace` | `Replace[expr, rules]` | Replace first match by rule(s). |
| `ReplaceAll` | `ReplaceAll[expr, rules]` | Replace all matches by rule(s). |
| `ReplaceFirst` | `ReplaceFirst[expr, rule]` | Replace first element(s) matching pattern. |
| `ReplaceRepeated` | `ReplaceRepeated[expr, rules]` | Repeatedly apply rules until fixed point (held) |
| `Reset` | `Reset[handle]` | Reset or clear a handle (e.g., VectorStore) |
| `Reshape` | `Reshape[tensor, dims]` | Reshape a tensor to new dims (supports -1) |
| `ReshapeLayer` | `ReshapeLayer[shape|opts]` | Reshape tensor to new shape |
| `Residual` | `Residual[layers]` | Residual wrapper with inner layers (adds skip) |
| `ResidualBlock` | `ResidualBlock[opts?]` | Two convs + skip (MVP no norm) |
| `ResolveRelative` | `ResolveRelative[path]` | Resolve a path relative to current file/module. |
| `RestartContainer` | `RestartContainer[id]` | Restart a container |
| `Reverse` | `Reverse[list]` | Reverse a list |
| `RightCompose` | `RightCompose[f, g, …]` | Compose functions right-to-left |
| `Roots` | `Roots[poly, var?]` | Polynomial roots for univariate polynomial. |
| `Round` | `Round[x]` | Round to nearest integer |
| `Rule` | `Rule[k, v]` | Format a rule (k->v) as a string |
| `STFT` | `STFT[x, size|opts, hop?]` | Short-time Fourier transform. STFT[x, size, hop?] or options. |
| `SVD` | `SVD[A]` | Reduced singular value decomposition A = U S V^T (via AtA eigen). |
| `SampleEdges` | `SampleEdges[graph, k]` | Sample k edges uniformly |
| `SampleNodes` | `SampleNodes[graph, k]` | Sample k nodes uniformly |
| `SaveImage` | `SaveImage[ref, path]` | Save image to tar |
| `ScatterPlot` | `ScatterPlot[data, opts]` | Render a scatter plot |
| `Schema` | `Schema[value]` | Return a minimal schema for a value/association. |
| `Scope` | `Scope[opts, body]` | Run body with resource limits (held) |
| `Search` | `Search[target, query, opts?]` | Search within a store or index (VectorStore, Index) |
| `SearchImages` | `SearchImages[query, opts?]` | Search registry images |
| `SecretsGet` | `SecretsGet[key, provider]` | Get secret by key from provider (Env or File). |
| `SeedRandom` | `SeedRandom[seed?]` | Seed deterministic RNG scoped to this evaluator. |
| `Send` | `Send[ch, value]` | Send value to channel (held) |
| `SeparableConv2D` | `SeparableConv2D[opts?]` | Depthwise + 1x1 pointwise convolution |
| `Sequential` | `Sequential[layers, opts?]` | Construct a sequential network from layers |
| `SessionClear` | `SessionClear[session]` | Clear a named session buffer. |
| `Set` | `Set[symbol, value]` | Assignment: Set[symbol, value]. |
| `SetDelayed` | `SetDelayed[symbol, expr]` | Delayed assignment evaluated on use. |
| `SetDownValues` | `SetDownValues[symbol, defs]` | Attach DownValues to a symbol (held) |
| `SetEqualQ` | `SetEqualQ[a, b]` | Are two sets equal? |
| `SetFromList` | `SetFromList[list]` | Create set from list |
| `SetInsert` | `SetInsert[set, value]` | Insert value into set |
| `SetLevel` | `SetLevel[logger, level]` | Set logger level (trace\|debug\|info\|warn\|error). |
| `SetMemberQ` | `SetMemberQ[set, value]` | Is value a member of set? |
| `SetModulePath` | `SetModulePath[path]` | Set module search path |
| `SetOwnValues` | `SetOwnValues[symbol, defs]` | Attach OwnValues to a symbol (held) |
| `SetRemove` | `SetRemove[set, value]` | Remove value from set |
| `SetSubValues` | `SetSubValues[symbol, defs]` | Attach SubValues to a symbol (held) |
| `SetSubsetQ` | `SetSubsetQ[a, b]` | Is a subset of b? |
| `SetToList` | `SetToList[set]` | Convert set to list |
| `SetUiBackend` | `SetUiBackend[mode]` | Set UI backend: terminal \| null \| auto \| gui (requires ui_egui) |
| `SetUiTheme` | `SetUiTheme[mode, opts?]` | Set UI theme: system \| light \| dark. Optional opts: <\|AccentColor->color, Rounding->px, FontSize->pt, Compact->True\|False, SpacingScale->num, Palette-><\|Primary->color, Success->color, Warning->color, Error->color, Info->color, Background->color, Surface->color, Text->color\|>\|> |
| `SetUpValues` | `SetUpValues[symbol, defs]` | Attach UpValues to a symbol (held) |
| `Shape` | `Shape[tensor]` | Shape of a tensor |
| `ShowDataset` | `ShowDataset[ds, opts?]` | Pretty-print a dataset table to string |
| `Sigmoid` | `Sigmoid[x?]` | Sigmoid activation: tensor op or zero-arg layer |
| `SignPackage` | `SignPackage[path?, opts?]` | Sign package (requires lyra-pm) |
| `Signum` | `Signum[x]` | Sign of number (-1,0,1) |
| `Simplify` | `Simplify[expr]` | Simplify algebraic expression. |
| `Sin` | `Sin[x]` | Sine (radians). Tensor-aware: elementwise on tensors. |
| `SnakeCase` | `SnakeCase[s]` | Convert to snake_case |
| `Softmax` | `Softmax[x?, opts?]` | Softmax activation: zero-arg layer (tensor variant TBD) |
| `SoftmaxLayer` | `SoftmaxLayer[opts?]` | Softmax over features |
| `Solve` | `Solve[eqns, vars?]` | Solve equations for variables. |
| `Sort` | `Sort[list|ds, by?, opts?]` | Sort a list or dataset (dispatched). Overloads: Sort[list]; Sort[ds, by, opts?] |
| `SortBy` | `SortBy[f, subject]` | Sort list by key or association by derived key. |
| `Span` | `Span[name, opts?]` | Start a trace span and return its id. |
| `SpanEnd` | `SpanEnd[id?]` | End the last span or the given span id. |
| `SpinnerStart` | `SpinnerStart[text?, opts?]` | Start an indeterminate spinner; returns id |
| `SpinnerStop` | `SpinnerStop[id]` | Stop a spinner by id |
| `SplitLines` | `SplitLines[s]` | Split string on 
 into lines |
| `Sqrt` | `Sqrt[x]` | Square root. Tensor-aware: elementwise on tensors. |
| `StableKey` | `StableKey[x]` | Canonical stable key string for ordering/dedup. |
| `Stack` | `Stack[]` | Create a stack |
| `StandardDeviation` | `StandardDeviation[list]` | Standard deviation of list |
| `StartContainer` | `StartContainer[id]` | Start a container |
| `StartScope` | `StartScope[opts, body]` | Start a managed scope (held) |
| `StartsWithQ` | `StartsWithQ[s, prefix]` | Alias: StartsWith predicate |
| `Stats` | `Stats[id, opts?]` | Stream container stats |
| `StopActor` | `StopActor[actor]` | Stop actor |
| `StopContainer` | `StopContainer[id, opts?]` | Stop a container |
| `StripAnsi` | `StripAnsi[text]` | Remove ANSI escape codes from string |
| `StronglyConnectedComponents` | `StronglyConnectedComponents[graph]` | Strongly connected components |
| `Subgraph` | `Subgraph[graph, ids]` | Induced subgraph from node set |
| `Summary` | `Summary[obj]` | Summary of a network (dispatch to NetSummary) |
| `Switch` | `Switch[expr, rules…]` | Multi-way conditional by equals (held) |
| `SymbolQ` | `SymbolQ[x]` | Is value a symbol? |
| `TagImage` | `TagImage[src, dest]` | Tag an image |
| `Tail` | `Tail[ds, n]` | Take last n rows |
| `Tan` | `Tan[x]` | Tangent (radians) |
| `Tanh` | `Tanh[x]` | Hyperbolic tangent (Listable) |
| `Tell` | `Tell[actor, msg]` | Send message to actor (held) |
| `Tensor` | `Tensor[spec]` | Create a tensor (alias of NDArray) |
| `TermSize` | `TermSize[]` | Current terminal width/height |
| `TestPackage` | `TestPackage[path?, opts?]` | Run package tests (requires lyra-pm) |
| `TextCount` | `TextCount[input, pattern, opts?]` | Count regex matches per file and total. |
| `TextDetectEncoding` | `TextDetectEncoding[input]` | Detect likely text encoding for files. |
| `TextFilesWithMatch` | `TextFilesWithMatch[input, pattern, opts?]` | List files that contain the pattern. |
| `TextFind` | `TextFind[input, pattern, opts?]` | Find regex matches across files or text. |
| `TextLines` | `TextLines[input, pattern, opts?]` | Return matching lines with positions for a pattern. |
| `TextReplace` | `TextReplace[input, pattern, replacement, opts?]` | Replace pattern across files; supports dry-run and backups. |
| `TextSearch` | `TextSearch[input, query, opts?]` | Search text via regex, fuzzy, or index engine. |
| `Thread` | `Thread[expr]` | Thread Sequence and lists into arguments. |
| `Through` | `Through[fs, x]` | Through[{f,g}, x] applies each to x |
| `Throw` | `Throw[x]` | Throw a value for Catch |
| `TitleCase` | `TitleCase[s]` | Convert to Title Case |
| `Trace` | `Trace[A]` | Trace of a square matrix (sum of diagonal). |
| `TraceExport` | `TraceExport[format, opts?]` | Export spans to a file (json). |
| `TraceGet` | `TraceGet[]` | Return collected spans as a list of assoc. |
| `Train` | `Train[net, data, opts?]` | Train a network (dispatch to NetTrain) |
| `TransformerDecoder` | `TransformerDecoder[opts?]` | Decoder block: self-attn + cross-attn + FFN (single-batch) |
| `TransformerDecoderStack` | `TransformerDecoderStack[opts?]` | Stack N decoder blocks (returns Sequential network) |
| `TransformerEncoder` | `TransformerEncoder[opts?]` | Encoder block: MHA + FFN with residuals and layer norms (single-batch) |
| `TransformerEncoderDecoder` | `TransformerEncoderDecoder[opts?]` | Convenience: builds Encoder/Decoder stacks |
| `TransformerEncoderStack` | `TransformerEncoderStack[opts?]` | Stack N encoder blocks (returns Sequential network) |
| `Transpose` | `Transpose[A, perm?]` | Transpose of a matrix (or NDTranspose for permutations). |
| `TransposeLayer` | `TransposeLayer[perm|opts]` | Transpose/permute axes |
| `Trunc` | `Trunc[x]` | Truncate toward zero (Listable). |
| `Truncate` | `Truncate[text, width, ellipsis?]` | Truncate to width with ellipsis |
| `Try` | `Try[body, handler?]` | Try body; capture failures (held) |
| `TryOr` | `TryOr[body, default]` | Try body else default (held) |
| `TryReceive` | `TryReceive[ch]` | Non-blocking receive |
| `TrySend` | `TrySend[ch, value]` | Non-blocking send (held) |
| `Tune` | `Tune[obj, data, opts?]` | Hyperparameter search for estimator (dispatched) |
| `Unless` | `Unless[cond, body]` | Evaluate body when condition is False (held) |
| `UnpauseContainer` | `UnpauseContainer[id]` | Unpause a container |
| `Unset` | `Unset[symbol]` | Clear definition: Unset[symbol]. |
| `Unuse` | `Unuse[name]` | Unload a package; hide imported symbols |
| `UpdatePackage` | `UpdatePackage[name, opts?]` | Update a package (requires lyra-pm) |
| `Upsample2D` | `Upsample2D[opts?]` | Upsample HxW (Nearest/Bilinear) |
| `UrlFormDecode` | `UrlFormDecode[s]` | Parse form-encoded string to assoc |
| `UrlFormEncode` | `UrlFormEncode[params]` | application/x-www-form-urlencoded from assoc |
| `Using` | `Using[name, opts?]` | Load a package by name with import options |
| `UsingFile` | `UsingFile[path, fn]` | Open a file and pass handle to a function (ensures close) |
| `UuidV4` | `UuidV4[]` | Generate a random UUID v4 string. |
| `UuidV7` | `UuidV7[]` | Generate a time-ordered UUID v7 string. |
| `Variance` | `Variance[list]` | Variance of list |
| `VectorCount` | `VectorCount[store]` | Count items in store |
| `VectorDelete` | `VectorDelete[store, ids]` | Delete items by ids |
| `VectorReset` | `VectorReset[store]` | Clear all items in store |
| `VectorSearch` | `VectorSearch[store, query, opts]` | Search by vector or text (hybrid supported) |
| `VectorStore` | `VectorStore[optsOrDsn]` | Create/open a vector store (memory or DSN) |
| `VectorUpsert` | `VectorUpsert[store, rows]` | Insert or update vectors with metadata |
| `VersionBump` | `VersionBump[level, paths]` | Bump semver in files: major/minor/patch. |
| `Volume` | `Volume[opts?]` | Create volume |
| `WaitContainer` | `WaitContainer[id, opts?]` | Wait for container to stop |
| `When` | `When[cond, body]` | Evaluate body when condition is True (held) |
| `WhoAmI` | `WhoAmI[]` | Show current registry identity (requires lyra-pm) |
| `Window` | `Window[type, n, opts?]` | Window weights vector by type and size (Hann\|Hamming\|Blackman). |
| `With` | `With[<|vars|>, body]` | Lexically bind symbols within a body. |
| `WithPackage` | `WithPackage[name, expr]` | Temporarily add a path to $PackagePath |
| `WithPolicy` | `WithPolicy[opts, body]` | Evaluate body with temporary tool capabilities. |
| `Workflow` | `Workflow[steps]` | Run a list of steps sequentially (held) |
| `Wrap` | `Wrap[text, width]` | Wrap text to width |
| `XdgDirs` | `XdgDirs[]` | Return XDG base directories (data, cache, config). |

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

- Usage: `Apply[f, list, level?]`
- Summary: Apply head to list elements: Apply[f, {…}]
- Tags: functional, apply
- Examples:
  - `Apply[Plus, {1,2,3}]  ==> 6`

## `Binomial`

- Usage: `Binomial[n, k]`
- Summary: Binomial coefficient nCk
- Examples:
  - `Binomial[5, 2]  ==> 10`

## `BlankQ`

- Usage: `BlankQ[s]`
- Summary: Alias: IsBlank string predicate
- Examples:
  - `BlankQ["   "]  ==> True`
  - `BlankQ["x"]  ==> False`

## `BoundedChannel`

- Usage: `BoundedChannel[n]`
- Summary: Create bounded channel
- Tags: concurrency, channel
- Examples:
  - `ch := BoundedChannel[2]; Send[ch, 1]; Receive[ch]  ==> 1`

## `CDF`

- Usage: `CDF[dist, x]`
- Summary: Cumulative distribution for a distribution at x.
- Tags: stats, dist
- Examples:
  - `CDF[Exponential[2.0], 1.0]  ==> 1 - e^-2`
  - `CDF[Poisson[2.0], 3]  ==> Σ_{k=0..3} e^-2 2^k/k!`

## `CamelCase`

- Usage: `CamelCase[s]`
- Summary: Convert to camelCase
- Examples:
  - `CamelCase["hello world"]  ==> "helloWorld"`

## `CausalSelfAttention`

- Usage: `CausalSelfAttention[opts?]`
- Summary: Self-attention with causal mask
- Tags: nn, layer, transformer
- Examples:
  - `Sequential[{CausalSelfAttention[<|SeqLen->8, ModelDim->64, NumHeads->8|>]}]`

## `ChineseRemainder`

- Usage: `ChineseRemainder[residues, moduli]`
- Summary: Solve x ≡ r_i (mod m_i) for coprime moduli.
- Examples:
  - `ChineseRemainder[{2,3,2}, {3,5,7}]  ==> 23`
  - `23 mod 3==2, mod 5==3, mod 7==2`

## `Cholesky`

- Usage: `Cholesky[A]`
- Summary: Cholesky factorization for SPD matrices
- Examples:
  - `Cholesky[{{4,1},{1,3}}]  ==> <|L->...|>`
  - `L := Cholesky[A]["L"]; L.Transpose[L] == A`

## `Classifier`

- Usage: `Classifier[opts?]`
- Summary: Create classifier spec (Logistic/Baseline)
- Tags: ml, classification
- Examples:
  - `Classifier[<|Method->"Logistic"|>]`

## `Clip`

- Usage: `Clip[x, min, max]`
- Summary: Clamp value to [min,max]. Tensor-aware: elementwise on tensors.
- Examples:
  - `Clip[10, 0, 5]  ==> 5`
  - `Clip[Tensor[{-1,2,7}], 0, 5]  ==> Tensor[...]`

## `Combinations`

- Usage: `Combinations[listOrN, k]`
- Summary: All k-combinations (subsets) of a list or 1..n.
- Examples:
  - `Combinations[{1,2,3}, 2]  ==> {{1,2},{1,3},{2,3}}`
  - `Length[Combinations[4, 2]]  ==> 6`

## `CombinationsCount`

- Usage: `CombinationsCount[n, k]`
- Summary: Count of combinations; Binomial[n,k].
- Examples:
  - `CombinationsCount[5, 2]  ==> 10`

## `Compose`

- Usage: `Compose[f, g, …]`
- Summary: Compose functions left-to-right
- Tags: functional, compose
- Examples:
  - `Compose[f,g][x]  ==> f[g[x]]`

## `ConfigFind`

- Usage: `ConfigFind[names?, startDir?]`
- Summary: Search upwards for config files (e.g., .env, lyra.toml).
- Examples:
  - `ConfigFind["lyra.toml"]  ==> <|"path"->".../lyra.toml"|>`

## `Confirm`

- Usage: `Confirm[text, opts?]`
- Summary: Ask yes/no question (TTY)
- Examples:
  - `Confirm["Proceed?"]  ==> True|False`

## `Container`

- Usage: `Container[image, opts?]`
- Summary: Create a container
- Examples:
  - `cid := Container["alpine", <|"cmd"->"echo hi"|>]`
  - `StartContainer[cid]`

## `Contains`

- Usage: `Contains[container, item]`
- Summary: Membership test for strings/lists/sets/assocs
- Tags: generic, predicate
- Examples:
  - `Contains["foobar", "bar"]  ==> True`
  - `Contains[{1,2,3}, 2]  ==> True`
  - `Contains[<|a->1|>, "a"]  ==> True`
  - `s := HashSet[{1,2}]; Contains[s, 3]  ==> False`

## `ContainsKeyQ`

- Usage: `ContainsKeyQ[subject, key]`
- Summary: Key membership for assoc/rows/Dataset/Frame
- Tags: generic, predicate, schema
- Examples:
  - `ContainsKeyQ[<|a->1|>, "a"]  ==> True`
  - `ContainsKeyQ[{<|a->1|>,<|b->2|>}, "b"]  ==> True`

## `ContainsQ`

- Usage: `ContainsQ[container, item]`
- Summary: Alias: membership predicate
- Tags: generic, predicate
- Examples:
  - `ContainsQ[{1,2,3}, 2]  ==> True`

## `ConvTranspose2D`

- Usage: `ConvTranspose2D[opts?]`
- Summary: Transposed 2D convolution (deconv)
- Tags: nn, layer
- Examples:
  - `Sequential[{ConvTranspose2D[<|Output->1, KernelSize->2, Stride->2, InputChannels->1, Height->1, Width->1|>]}]`
  - `(* Upsample feature map 16x16 -> 32x32 *) Sequential[{ConvTranspose2D[<|Output->8, KernelSize->4, Stride->2, Padding->1, InputChannels->16, Height->16, Width->16|>], Relu[]}]`

## `Convolution2D`

- Usage: `Convolution2D[opts?]`
- Summary: 2D convolution layer (uses InputChannels/Height/Width for forward)
- Tags: nn, layer
- Examples:
  - `Sequential[{Convolution2D[<|Output->1, KernelSize->{2,2}, InputChannels->1, Height->2, Width->2, W->{{{{1,1},{1,1}}}}, b->{0}|>}]]`

## `Convolve`

- Usage: `Convolve[a, b, mode?]`
- Summary: Linear convolution of two sequences. Modes: Full|Same|Valid.
- Examples:
  - `Convolve[{1,2,1}, {1,1,1}]  ==> {1,3,4,3,1}`
  - `Convolve[{1,2,1}, {1,1,1}, "Same"]  ==> {1,3,4}`

## `CoprimeQ`

- Usage: `CoprimeQ[a, b]`
- Summary: Predicate: are integers coprime?
- Examples:
  - `CoprimeQ[12, 35]  ==> True`
  - `CoprimeQ[12, 18]  ==> False`

## `Cos`

- Usage: `Cos[x]`
- Summary: Cosine (radians). Tensor-aware: elementwise on tensors.
- Examples:
  - `Cos[0]  ==> 1`
  - `Cos[Tensor[{0, Pi}]]  ==> Tensor[...]`

## `Count`

- Usage: `Count[x]`
- Summary: Count items/elements (lists, assocs, Bag/VectorStore)
- Tags: generic, aggregate
- Examples:
  - `Count[{1,2,1,1}, 1]  ==> 3`

## `CrossAttention`

- Usage: `CrossAttention[opts?]`
- Summary: Cross-attention over Memory (seq x dim)
- Tags: nn, layer, transformer
- Examples:
  - `Sequential[{CrossAttention[<|SeqLen->8, ModelDim->64, NumHeads->8, Memory->{{...}}|>]}]`

## `CrossValidate`

- Usage: `CrossValidate[obj, data, opts?]`
- Summary: Cross-validate estimator + data (dispatched)
- Tags: ml, cv
- Examples:
  - `CrossValidate[Classifier[<|Method->"Logistic"|>], train, <|k->5|>]`

## `Describe`

- Usage: `Describe[name, items, opts?]`
- Summary: Define a test suite (held).
- Tags: generic, introspection, stats, testing
- Examples:
  - `Describe["Math", {It["adds", 1+1==2]}]  ==> <|"type"->"suite"|>`

## `Difference`

- Usage: `Difference[a, b]`
- Summary: Difference for lists or sets (dispatched)
- Examples:
  - `Difference[{1,2,3},{2}]  ==> {1,3}`

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

## `DividesQ`

- Usage: `DividesQ[a, b]`
- Summary: Predicate: does a divide b?
- Examples:
  - `DividesQ[3, 12]  ==> True`
  - `DividesQ[5, 12]  ==> False`

## `DotenvLoad`

- Usage: `DotenvLoad[path?, opts?]`
- Summary: Load .env variables into process env.
- Examples:
  - `DotenvLoad[]  ==> <|"path"->".../.env", "loaded"->n|>`

## `EigenDecomposition`

- Usage: `EigenDecomposition[A]`
- Summary: Eigenvalues and eigenvectors. Symmetric: Jacobi; general: real QR + inverse iteration.
- Examples:
  - `EigenDecomposition[{{2,1},{1,2}}]  ==> <|Eigenvalues->{3,1}, Eigenvectors->...|>`

## `EmptyQ`

- Usage: `EmptyQ[x]`
- Summary: Is the subject empty? (lists, strings, assocs, handles)
- Tags: generic, predicate
- Examples:
  - `EmptyQ[{}]  ==> True`
  - `EmptyQ[""]  ==> True`
  - `EmptyQ[Queue[]]  ==> True`
  - `EmptyQ[DatasetFromRows[{}]]  ==> True`

## `EndsWithQ`

- Usage: `EndsWithQ[s, suffix]`
- Summary: Alias: EndsWith predicate
- Examples:
  - `EndsWithQ["hello", "lo"]  ==> True`

## `EnvExpand`

- Usage: `EnvExpand[text, opts?]`
- Summary: Expand $VAR or %VAR% style environment variables in text.
- Examples:
  - `EnvExpand["Hello $USER"]  ==> "Hello alice"`
  - `EnvExpand["%HOME%\tmp", <|"Style"->"windows"|>]  ==> "/home/alice/tmp"`

## `EqualQ`

- Usage: `EqualQ[a, b]`
- Summary: Structural equality for sets and handles
- Tags: generic, set
- Examples:
  - `EqualQ[HashSet[{1,2}], HashSet[{2,1}]]  ==> True`

## `Estimator`

- Usage: `Estimator[opts]`
- Summary: Create ML estimator spec (Task/Method)
- Tags: ml, estimator
- Examples:
  - `Estimator[<|Task->"Classification", Method->"Logistic"|>]`

## `EulerPhi`

- Usage: `EulerPhi[n]`
- Summary: Euler's totient function: count of 1<=k<=n with CoprimeQ[k,n].
- Examples:
  - `EulerPhi[36]  ==> 12`

## `Evaluate`

- Usage: `Evaluate[model, data, opts?]`
- Summary: Evaluate an ML model on data (dispatched)
- Tags: ml, metrics
- Examples:
  - `Evaluate[model, val, <|Metrics->{Accuracy}|>]`

## `Exp`

- Usage: `Exp[x]`
- Summary: Natural exponential e^x. Tensor-aware: elementwise on tensors.
- Examples:
  - `Exp[1]  ==> 2.71828...`
  - `Exp[Tensor[{0,1}]]  ==> Tensor[...]`

## `Explain`

- Usage: `Explain[expr]`
- Summary: Explain evaluation; returns trace steps when enabled.
- Examples:
  - `Explain[Plus[1,2]]  ==> <|steps->...|>`

## `ExtendedGCD`

- Usage: `ExtendedGCD[a, b]`
- Summary: Extended GCD: returns {g, x, y} with a x + b y = g.
- Examples:
  - `ExtendedGCD[240, 46]  ==> {2, -9, 47}`

## `FFN`

- Usage: `FFN[opts?]`
- Summary: Position-wise feed-forward (supports SwiGLU/GEGLU)
- Tags: nn, layer, transformer
- Examples:
  - `Sequential[{FFN[<|SeqLen->8, ModelDim->64, HiddenDim->256, Variant->"SwiGLU"|>]}]`

## `FFT`

- Usage: `FFT[x, n?]`
- Summary: Discrete Fourier transform of a 1D sequence (returns Complex list).
- Examples:
  - `FFT[{1,0,0,0}]  ==> {1+0i, 1+0i, 1+0i, 1+0i}`
  - `IFFT[%]  ==> {1,0,0,0}`

## `FactorInteger`

- Usage: `FactorInteger[n]`
- Summary: Prime factorization as {{p1,e1},{p2,e2},…}.
- Examples:
  - `FactorInteger[84]  ==> {{2,2},{3,1},{7,1}}`

## `Factorial`

- Usage: `Factorial[n]`
- Summary: n! (product 1..n)
- Examples:
  - `Factorial[5]  ==> 120`

## `Fit`

- Usage: `Fit[net, data, opts?]`
- Summary: Train a network (alias of NetTrain)
- Tags: nn, train
- Examples:
  - `Fit[net, data, <|Epochs->1, BatchSize->32|>]`

## `FixedPoint`

- Usage: `FixedPoint[f, x]`
- Summary: Iterate f until convergence
- Tags: functional, fixedpoint
- Examples:
  - `FixedPoint[Cos, 1.0]  ==> 0.739... `

## `Flatten`

- Usage: `Flatten[list, levels?]`
- Summary: Flatten by levels (default 1)
- Tags: nn, layer
- Examples:
  - `Flatten[{{1},{2,3}}]  ==> {1,2,3}`

## `FoldList`

- Usage: `FoldList[f, init, list]`
- Summary: Cumulative fold producing intermediates
- Tags: functional, fold
- Examples:
  - `FoldList[Plus, 0, {1,2,3}]  ==> {0,1,3,6}`

## `FrameColumns`

- Usage: `FrameColumns[frame]`
- Summary: List column names for a Frame
- Tags: frame, schema
- Examples:
  - `FrameColumns[f]`

## `FrameDescribe`

- Usage: `FrameDescribe[frame, opts?]`
- Summary: Quick stats by columns
- Tags: frame, stats
- Examples:
  - `FrameDescribe[f]`

## `FrameDistinct`

- Usage: `FrameDistinct[frame, cols?]`
- Summary: Distinct rows in Frame (optional columns)
- Tags: frame, distinct
- Examples:
  - `FrameDistinct[f, {"a"}]`

## `FrameFilter`

- Usage: `FrameFilter[frame, pred]`
- Summary: Filter rows in a Frame
- Tags: frame, transform, filter
- Examples:
  - `FrameFilter[f, #a>1 &]`

## `FrameFromRows`

- Usage: `FrameFromRows[rows]`
- Summary: Create a Frame from assoc rows
- Tags: frame, create
- Examples:
  - `f := FrameFromRows[{<|a->1|>,<|a->2|>}]`

## `FrameHead`

- Usage: `FrameHead[frame, n?]`
- Summary: Take first n rows from Frame
- Tags: frame, inspect
- Examples:
  - `FrameHead[f, 5]`

## `FrameJoin`

- Usage: `FrameJoin[left, right, on?, opts?]`
- Summary: Join two Frames by keys
- Tags: frame, join
- Examples:
  - `FrameJoin[f1, f2, {"id"}]`

## `FrameOffset`

- Usage: `FrameOffset[frame, n]`
- Summary: Skip first n rows of Frame
- Tags: frame, transform
- Examples:
  - `FrameOffset[f, 10]`

## `FrameSelect`

- Usage: `FrameSelect[frame, spec]`
- Summary: Select/compute columns in Frame
- Tags: frame, transform, select
- Examples:
  - `FrameSelect[f, {"a"}]`

## `FrameSort`

- Usage: `FrameSort[frame, by]`
- Summary: Sort Frame by columns
- Tags: frame, sort
- Examples:
  - `FrameSort[f, {"a"}]`

## `FrameTail`

- Usage: `FrameTail[frame, n?]`
- Summary: Take last n rows from Frame
- Tags: frame, inspect
- Examples:
  - `FrameTail[f, 5]`

## `FrameUnion`

- Usage: `FrameUnion[frames…]`
- Summary: Union Frames by columns (schema union)
- Tags: frame, set
- Examples:
  - `FrameUnion[f1, f2]`

## `Future`

- Usage: `Future[expr]`
- Summary: Create a Future from an expression (held)
- Tags: concurrency, async
- Examples:
  - `f := Future[Range[1,1_000]]; Await[f]  ==> {1,2,...}`

## `GCD`

- Usage: `GCD[a, b, …]`
- Summary: Greatest common divisor
- Examples:
  - `GCD[18, 24]  ==> 6`

## `Gelu`

- Usage: `Gelu[x?]`
- Summary: GELU activation (tanh approx): tensor op or zero-arg layer
- Tags: tensor, nn, activation
- Examples:
  - `Gelu[]  (* layer spec *)`

## `Get`

- Usage: `Get[subject, key, default?]`
- Summary: Get value by key/index from a structure (dispatch)
- Examples:
  - `Get[<|"a"->1|>, "a"]  ==> 1`
  - `Get[<|"a"->1|>, "b", 9]  ==> 9`

## `Gets`

- Usage: `Gets[path?]`
- Summary: Read entire stdin or file as string
- Examples:
  - `Gets["/tmp/hi.txt"]  ==> "hello!"`

## `GitAdd`

- Usage: `GitAdd[paths, opts?]`
- Summary: Stage files for commit
- Tags: git, index
- Examples:
  - `GitAdd["src/main.rs"]  ==> True`

## `GitBranch`

- Usage: `GitBranch[name, opts?]`
- Summary: Create a new branch
- Tags: git, branch
- Examples:
  - `GitBranch["feature/x"]  ==> True`

## `GitCommit`

- Usage: `GitCommit[message, opts?]`
- Summary: Create a commit with message
- Tags: git, commit
- Examples:
  - `GitCommit["feat: add api"]  ==> <|Sha->..., Message->...|>`

## `GitDiff`

- Usage: `GitDiff[opts?]`
- Summary: Diff against base and optional paths
- Tags: git, diff
- Examples:
  - `GitDiff[<|"Base"->"HEAD~1"|>]  ==> "diff..."`

## `GitRoot`

- Usage: `GitRoot[]`
- Summary: Path to repository root (Null if absent)
- Tags: git, vcs
- Examples:
  - `GitRoot[]  ==> "/path/to/repo" | Null`

## `GitStatus`

- Usage: `GitStatus[opts?]`
- Summary: Status (porcelain) with branch/ahead/behind/changes
- Tags: git, status
- Examples:
  - `GitStatus[]  ==> <|Branch->..., Ahead->0, Behind->0, Changes->{...}|>`

## `GitSwitch`

- Usage: `GitSwitch[name, opts?]`
- Summary: Switch to branch (optionally create)
- Tags: git, branch
- Examples:
  - `GitSwitch["feature/x"]  ==> True`

## `GitVersion`

- Usage: `GitVersion[]`
- Summary: Get git client version string
- Tags: git, vcs
- Examples:
  - `GitVersion[]  ==> "git version ..."`

## `GroupNorm`

- Usage: `GroupNorm[opts?]`
- Summary: Group normalization over channels (NumGroups)
- Tags: nn, layer
- Examples:
  - `Sequential[{Convolution2D[<|Output->32, KernelSize->3, Padding->1, InputChannels->3, Height->32, Width->32|>], GroupNorm[<|NumGroups->4, InputChannels->32, Height->32, Width->32|>], Relu[]}]`

## `Identity`

- Usage: `Identity[x]`
- Summary: Identity function: returns its argument
- Tags: functional
- Examples:
  - `Identity[42]  ==> 42`

## `If`

- Usage: `If[cond, then, else?]`
- Summary: Conditional: If[cond, then, else?] (held)
- Examples:
  - `If[1<2, "yes", "no"]  ==> "yes"`

## `IndexOf`

- Usage: `IndexOf[s, substr, from?]`
- Summary: Index of substring (0-based; -1 if not found)
- Examples:
  - `IndexOf["banana", "na"]  ==> 2`
  - `IndexOf["banana", "x"]  ==> -1`

## `Info`

- Usage: `Info[target]`
- Summary: Information about a handle (Logger, Graph, etc.)
- Tags: generic, introspection
- Examples:
  - `Info[Graph[]]  ==> <|nodes->..., edges->...|>`
  - `Info[DatasetFromRows[{<|a->1|>}]]  ==> <|Type->"Dataset", Rows->1, Columns->{"a"}|>`
  - `Info[VectorStore[<|name->"vs"|>]]  ==> <|Type->"VectorStore", Name->"vs", Count->0|>`
  - `Info[HashSet[{1,2,3}]]  ==> <|Type->"Set", Size->3|>`
  - `Info[Queue[]]  ==> <|Type->"Queue", Size->0|>`
  - `Info[Index["/tmp/idx.db"]]  ==> <|indexPath->..., numDocs->...|>`
  - `conn := Connect["mock://"]; Info[conn]  ==> <|Type->"Connection", ...|>`

## `Initializer`

- Usage: `Initializer[opts?]`
- Summary: Initializer spec for layer parameters
- Tags: nn, init
- Examples:
  - `Initializer[<|Type->"Xavier"|>]`

## `Intersection`

- Usage: `Intersection[args]`
- Summary: Intersection for lists or sets (dispatched)
- Examples:
  - `Intersection[{1,2,3},{2,4}]  ==> {2}`

## `IsBlank`

- Usage: `IsBlank[s]`
- Summary: True if string is empty or whitespace
- Examples:
  - `IsBlank["   "]  ==> True`
  - `IsBlank["a"]  ==> False`

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

## `LU`

- Usage: `LU[A]`
- Summary: LU factorization: returns <|L,U,P|>
- Examples:
  - `LU[{{1,2},{3,4}}]  ==> <|L->..., U->..., P->...|>`
  - `{L,U,P} := Values[LU[A]]; L.U ≈ P.A`

## `LastIndexOf`

- Usage: `LastIndexOf[s, substr, from?]`
- Summary: Last index of substring (0-based; -1 if not found)
- Examples:
  - `LastIndexOf["banana", "na"]  ==> 4`

## `Length`

- Usage: `Length[x]`
- Summary: Length of a list or string.
- Tags: generic
- Examples:
  - `Length[{1,2,3}]  ==> 3`
  - `Length["ok"]  ==> 2`

## `Map`

- Usage: `Map[f, list]`
- Summary: Map a function over a list.
- Examples:
  - `Map[ToUpper, {"a", "b"}]  ==> {"A", "B"}`
  - `Map[#^2 &, {1,2,3}]  ==> {1,4,9}`

## `MapKeyValues`

- Usage: `MapKeyValues[f, assoc]`
- Summary: Map over (k,v) pairs to new values or pairs
- Examples:
  - `MapKeyValues[(k,v)=>k<>ToString[v], <|"a"->1, "b"->2|>]  ==> {"a1", "b2"}`

## `MapValues`

- Usage: `MapValues[f, assoc]`
- Summary: Map values with f[v]
- Examples:
  - `MapValues[ToUpper, <|"a"->"x"|>]  ==> <|"a"->"X"|>`

## `MemberQ`

- Usage: `MemberQ[container, item]`
- Summary: Alias: membership predicate (Contains)
- Tags: generic, predicate
- Examples:
  - `MemberQ[{1,2,3}, 2]  ==> True`
  - `MemberQ["foobar", "bar"]  ==> True`

## `Minus`

- Usage: `Minus[a, b?]`
- Summary: Subtract or unary negate.
- Examples:
  - `Minus[5, 2]  ==> 3`
  - `Minus[5]  ==> -5`

## `MobiusMu`

- Usage: `MobiusMu[n]`
- Summary: Möbius mu function: 0 if non-square-free, else (-1)^k.
- Examples:
  - `MobiusMu[36]  ==> 0`
  - `MobiusMu[35]  ==> 1`
  - `MobiusMu[30]  ==> -1`

## `Mod`

- Usage: `Mod[a, n]`
- Summary: Modulo remainder ((a mod n) >= 0)
- Examples:
  - `Mod[7, 3]  ==> 1`

## `ModInverse`

- Usage: `ModInverse[a, m]`
- Summary: Multiplicative inverse of a modulo m (if coprime).
- Examples:
  - `ModInverse[3, 11]  ==> 4`

## `MultiHeadAttention`

- Usage: `MultiHeadAttention[opts?]`
- Summary: Self-attention with NumHeads (single-batch)
- Tags: nn, layer, transformer
- Examples:
  - `Sequential[{MultiHeadAttention[<|SeqLen->4, ModelDim->32, NumHeads->4|>]}]`
  - `Sequential[{MultiHeadAttention[<|SeqLen->4, ModelDim->32, NumHeads->4, Mask->"Causal"|>]}]`

## `Nest`

- Usage: `Nest[f, x, n]`
- Summary: Nest function n times: Nest[f, x, n]
- Tags: functional, iteration
- Examples:
  - `Nest[#*2 &, 1, 3]  ==> 8`

## `NetApply`

- Usage: `NetApply[net, x, opts?]`
- Summary: Apply network to input(s)
- Examples:
  - `NetApply[neti, {1.0, 2.0}]  ==> {...}`

## `NetChain`

- Usage: `NetChain[layers, opts?]`
- Summary: Compose layers sequentially
- Examples:
  - `net := NetChain[{LinearLayer[8], ActivationLayer["relu"], LinearLayer[1]}]`

## `NetInitialize`

- Usage: `NetInitialize[net, opts?]`
- Summary: Initialize a network (weights/state)
- Examples:
  - `neti := NetInitialize[net]  ==> net' (initialized)`

## `NetSummary`

- Usage: `NetSummary[net]`
- Summary: Human-readable network summary
- Examples:
  - `NetSummary[neti]  ==> "Layer (out) ..."`

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

## `NextPrime`

- Usage: `NextPrime[n]`
- Summary: Next prime greater than n (or 2 if n<2).
- Examples:
  - `NextPrime[10]  ==> 11`
  - `NextPrime[-5]  ==> 2`

## `Notify`

- Usage: `Notify[text, opts?]`
- Summary: Show a notification/message to the user
- Examples:
  - `Notify["Saved successfully", <|Level->"Success"|>]`
  - `Notify["Low disk space", <|Level->"Warning", timeoutMs->5000|>]`
  - `Notify["Build failed", <|Level->"Error", Title->"CI"|>]`
  - `Notify["Heads up", <|Level->"Info", AccentColor->"cyan"|>]`
  - `Notify["Tap anywhere to dismiss", <|CloseOnClick->True, ShowDismiss->False|>]`

## `PDF`

- Usage: `PDF[dist, x]`
- Summary: Probability density/mass for a distribution at x.
- Tags: stats, dist
- Examples:
  - `PDF[Normal[0,1], 0]  ==> 0.39894…`
  - `PDF[BinomialDistribution[10, 0.5], 5]  ==> 0.24609375`

## `ParallelMap`

- Usage: `ParallelMap[f, list]`
- Summary: Map in parallel over list
- Tags: concurrency, parallel
- Examples:
  - `ParallelMap[#^2 &, Range[1,4]]  ==> {1,4,9,16}`

## `PatchEmbedding2D`

- Usage: `PatchEmbedding2D[opts?]`
- Summary: 2D to tokens via patch conv
- Tags: nn, layer, vision, transformer
- Examples:
  - `Sequential[{PatchEmbedding2D[<|PatchSize->{4,4}, ModelDim->64, InputChannels->3, Height->32, Width->32|>]}]`

## `Permutations`

- Usage: `Permutations[listOrN, k?]`
- Summary: All permutations (or k-permutations) of a list or range 1..n.
- Examples:
  - `Permutations[{1,2}]  ==> {{1,2},{2,1}}`
  - `Length[Permutations[{1,2,3}]]  ==> 6`
  - `Permutations[3, 2]  ==> {{1,2},{1,3},{2,1},{2,3},{3,1},{3,2}}`

## `PermutationsCount`

- Usage: `PermutationsCount[n, k?]`
- Summary: Count of permutations; n! or nPk.
- Examples:
  - `PermutationsCount[5]  ==> 120`
  - `PermutationsCount[5, 2]  ==> 20`

## `Pooling2D`

- Usage: `Pooling2D[kind, size, opts?]`
- Summary: 2D pooling layer (Max/Avg; requires InputChannels/Height/Width)
- Tags: nn, layer
- Examples:
  - `Sequential[{Pooling2D["Max", 2, <|InputChannels->1, Height->2, Width->2|>]}]`

## `PositionalEmbedding`

- Usage: `PositionalEmbedding[opts?]`
- Summary: Learnable positional embeddings (adds to input)
- Tags: nn, layer, transformer
- Examples:
  - `Sequential[{PositionalEmbedding[<|SeqLen->8, ModelDim->64|>]}]`

## `PositionalEncoding`

- Usage: `PositionalEncoding[opts?]`
- Summary: Sinusoidal positional encoding (adds to input)
- Tags: nn, layer, transformer
- Examples:
  - `Sequential[{PositionalEncoding[<|SeqLen->4, ModelDim->32|>]}]`

## `Power`

- Usage: `Power[a, b]`
- Summary: Exponentiation (right-associative). Tensor-aware: elementwise when any tensor.
- Examples:
  - `Power[2, 8]  ==> 256`
  - `Power[Tensor[{2,3}], 2]  ==> Tensor[...]`
  - `2^3^2 parses as 2^(3^2)`

## `PowerMod`

- Usage: `PowerMod[a, b, m]`
- Summary: Modular exponentiation: a^b mod m (supports negative b if invertible).
- Examples:
  - `PowerMod[2, 10, 1000]  ==> 24`
  - `PowerMod[3, -1, 11]  ==> 4`

## `Predict`

- Usage: `Predict[obj, x, opts?]`
- Summary: Predict using a network or estimator (dispatched)
- Tags: nn, ml, inference
- Examples:
  - `Predict[Sequential[{Relu[]}], {1,2,3}]`
  - `(* Tensor input *) net := Sequential[{Convolution2D[<|Output->4, KernelSize->3, Padding->1, InputChannels->1, Height->28, Width->28|>]}]; Shape[Predict[net, Tensor[(* 1x28x28 data *)]]]  ==> {4,28,28}`

## `PrimeFactors`

- Usage: `PrimeFactors[n]`
- Summary: List of prime factors with multiplicity.
- Examples:
  - `PrimeFactors[84]  ==> {2,2,3,7}`

## `PrimeQ`

- Usage: `PrimeQ[n]`
- Summary: Predicate: is integer prime?
- Examples:
  - `PrimeQ[17]  ==> True`
  - `PrimeQ[1]  ==> False`

## `ProgressBar`

- Usage: `ProgressBar[total]`
- Summary: Create a progress bar; returns id.
- Examples:
  - `pb := ProgressBar[100]`
  - `ProgressAdvance[pb, 10]  ==> True`
  - `ProgressFinish[pb]  ==> True`

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

## `PromptSelect`

- Usage: `PromptSelect[text, items, opts?]`
- Summary: Prompt user to select one item from a list.
- Examples:
  - `PromptSelect["Pick one", {"A","B","C"}]  ==> "B"`
  - `PromptSelect["Pick one", {<|"name"->"Human", "value"->"human"|>, <|"name"->"AI", "value"->"ai"|>}]  ==> "ai"`

## `PromptSelectMany`

- Usage: `PromptSelectMany[text, items, opts?]`
- Summary: Prompt user to select many items from a list
- Examples:
  - `PromptSelectMany["Pick some", {"A","B","C"}]  ==> {"A","C"}`

## `PseudoInverse`

- Usage: `PseudoInverse[A]`
- Summary: Moore–Penrose pseudoinverse via reduced SVD (V S^+ U^T).
- Examples:
  - `PseudoInverse[{{1,2},{2,4}}]  ==> least-squares inverse (2x2 rank-1)`

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

## `QR`

- Usage: `QR[A, opts?]`
- Summary: QR decomposition via Householder reflections. Use "Reduced" option for thin Q,R.
- Examples:
  - `r := QR[{{1,2},{3,4}}]  ==> <|Q->..., R->...|>`
  - `r2 := QR[{{1,2,3},{4,5,6}}, "Reduced"]  ==> Q:(2x2), R:(2x3)`

## `RMSNorm`

- Usage: `RMSNorm[opts?]`
- Summary: RMS normalization (seq x dim)
- Tags: nn, layer, transformer
- Examples:
  - `Sequential[{RMSNorm[<|SeqLen->8, ModelDim->64|>]}]`

## `RandomInteger`

- Usage: `RandomInteger[spec?]`
- Summary: Random integer; supports {min,max}.
- Tags: random
- Examples:
  - `SeedRandom[1]; RandomInteger[{1,3}]  ==> 2`

## `RandomReal`

- Usage: `RandomReal[spec?]`
- Summary: Random real; supports {min,max}.
- Tags: random
- Examples:
  - `SeedRandom[1]; RandomReal[{0.0,1.0}]  ==> 0.3...`

## `RandomVariate`

- Usage: `RandomVariate[dist, n?]`
- Summary: Sample from a distribution (optionally n samples).
- Tags: stats, random
- Examples:
  - `RandomVariate[Normal[0,1], 3]  ==> {…}`
  - `RandomVariate[Bernoulli[0.3], 5]  ==> {0,1,0,0,1}`

## `RegexIsMatch`

- Usage: `RegexIsMatch[s, pattern]`
- Summary: Test if regex matches string
- Examples:
  - `RegexIsMatch[\"abc123\", \"\\d+\"]  ==> True`

## `Regressor`

- Usage: `Regressor[opts?]`
- Summary: Create regressor spec (Linear/Baseline)
- Tags: ml, regression
- Examples:
  - `Regressor[<|Method->"Linear"|>]`

## `Relu`

- Usage: `Relu[x]`
- Summary: Rectified Linear Unit: max(0, x). Tensor-aware: elementwise on tensors.
- Tags: tensor, nn, activation
- Examples:
  - `Relu[-2]  ==> 0`
  - `Relu[Tensor[{-1,0,2}]]  ==> Tensor[...]`

## `Replace`

- Usage: `Replace[expr, rules]`
- Summary: Replace first match by rule(s).
- Examples:
  - `Replace["foo bar", "o", "0"]  ==> "f00 bar"`

## `ReplaceAll`

- Usage: `ReplaceAll[expr, rules]`
- Summary: Replace all matches by rule(s).
- Examples:
  - `ReplaceAll[{1,2,1,3}, 1->9]  ==> {9,2,9,3}`
  - `ReplaceAll["a-b-a", "a"->"x"]  ==> "x-b-x"`

## `ResidualBlock`

- Usage: `ResidualBlock[opts?]`
- Summary: Two convs + skip (MVP no norm)
- Tags: nn, layer
- Examples:
  - `Sequential[{Convolution2D[<|Output->8, KernelSize->3, Padding->1, InputChannels->3, Height->32, Width->32|>], ResidualBlock[<|Output->8, KernelSize->3, Padding->1, Activation->"Relu"|>]}]`

## `Round`

- Usage: `Round[x]`
- Summary: Round to nearest integer
- Examples:
  - `Round[2.6]  ==> 3`

## `STFT`

- Usage: `STFT[x, size|opts, hop?]`
- Summary: Short-time Fourier transform. STFT[x, size, hop?] or options.
- Examples:
  - `STFT[Range[0, 127], 64, 32]  ==> {{…}, {…}, …}`

## `SVD`

- Usage: `SVD[A]`
- Summary: Reduced singular value decomposition A = U S V^T (via AtA eigen).
- Examples:
  - `sv := SVD[{{1,2},{3,4}}]  ==> <|U->(2x2), S->{..}, V->(2x2)|>`
  - `Dot[sv["U"], DiagMatrix[sv["S"]], Transpose[sv["V"]]]  ==> {{1,2},{3,4}}`

## `Schema`

- Usage: `Schema[value]`
- Summary: Return a minimal schema for a value/association.
- Examples:
  - `Schema[<|"a"->1, "b"->"x"|>]  ==> <|a->Integer, b->String|>`

## `Search`

- Usage: `Search[target, query, opts?]`
- Summary: Search within a store or index (VectorStore, Index)
- Tags: generic, search
- Examples:
  - `Search[VectorStore[<|name->"vs"|>], {0.1,0.2,0.3}]  ==> {...}`
  - `idx := Index["/tmp/idx.db"]; Search[idx, "foo"]`

## `SeedRandom`

- Usage: `SeedRandom[seed?]`
- Summary: Seed deterministic RNG scoped to this evaluator.
- Tags: random
- Examples:
  - `SeedRandom[1]  ==> True`

## `SeparableConv2D`

- Usage: `SeparableConv2D[opts?]`
- Summary: Depthwise + 1x1 pointwise convolution
- Tags: nn, layer
- Examples:
  - `Sequential[{SeparableConv2D[<|Output->2, KernelSize->3, InputChannels->1, Height->8, Width->8|>]}]`

## `Sequential`

- Usage: `Sequential[layers, opts?]`
- Summary: Construct a sequential network from layers
- Tags: nn, network
- Examples:
  - `Sequential[{Dense[<|Output->4|>], Relu[]}]`
  - `(* MLP *) Sequential[{Flatten[], Dense[<|Output->64|>], Relu[], Dense[<|Output->10|>], Softmax[]}]`
  - `(* Conv block *) Sequential[{Convolution2D[<|Output->8, KernelSize->3, Padding->1, InputChannels->1, Height->28, Width->28|>], Relu[], Pooling2D["Max", 2, <|InputChannels->8, Height->28, Width->28|>]}]`
  - `(* End-to-end conv → dense *) net := Sequential[{Convolution2D[<|Output->8, KernelSize->3, Padding->1, InputChannels->1, Height->28, Width->28|>], Relu[], Pooling2D["Max", 2], Flatten[], Dense[<|Output->10|>], Softmax[]}]; Shape[Predict[net, Tensor[(* 1x28x28 image tensor here *)]]]  ==> {10}`

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

## `Sigmoid`

- Usage: `Sigmoid[x?]`
- Summary: Sigmoid activation: tensor op or zero-arg layer
- Tags: tensor, nn, activation
- Examples:
  - `Sigmoid[{-1,0,1}]  ==> {~0.2689, 0.5, ~0.7311}`
  - `Sigmoid[]  (* layer spec *)`

## `Sin`

- Usage: `Sin[x]`
- Summary: Sine (radians). Tensor-aware: elementwise on tensors.
- Examples:
  - `Sin[0]  ==> 0`
  - `Sin[Tensor[{0, Pi/2}]]  ==> Tensor[...]`

## `SnakeCase`

- Usage: `SnakeCase[s]`
- Summary: Convert to snake_case
- Examples:
  - `SnakeCase["HelloWorld"]  ==> "hello_world"`

## `Softmax`

- Usage: `Softmax[x?, opts?]`
- Summary: Softmax activation: zero-arg layer (tensor variant TBD)
- Tags: nn, activation
- Examples:
  - `Softmax[]  (* layer spec *)`

## `SortBy`

- Usage: `SortBy[f, subject]`
- Summary: Sort list by key or association by derived key.
- Tags: generic, sort
- Examples:
  - `SortBy[Length, {"a","bbb","cc"}]  ==> {"a","cc","bbb"}`

## `Span`

- Usage: `Span[name, opts?]`
- Summary: Start a trace span and return its id.
- Examples:
  - `id := Span["work", <|"Attrs"-><|"module"->"demo"|>|>]`
  - `SpanEnd[id]  ==> True`
  - `TraceGet[]  ==> {<|"Name"->"work", ...|>, ...}`

## `Sqrt`

- Usage: `Sqrt[x]`
- Summary: Square root. Tensor-aware: elementwise on tensors.
- Examples:
  - `Sqrt[9]  ==> 3`
  - `Sqrt[Tensor[{1,4,9}]]  ==> Tensor[...]`

## `StableKey`

- Usage: `StableKey[x]`
- Summary: Canonical stable key string for ordering/dedup.
- Tags: generic, key
- Examples:
  - `StableKey[<|a->1|>]  ==> "6:<|a=>0:00000000000000000001|>"`

## `StartsWithQ`

- Usage: `StartsWithQ[s, prefix]`
- Summary: Alias: StartsWith predicate
- Examples:
  - `StartsWithQ["hello", "he"]  ==> True`

## `Tanh`

- Usage: `Tanh[x]`
- Summary: Hyperbolic tangent (Listable)
- Examples:
  - `Tanh[0]  ==> 0`
  - `Tanh[1]  ==> 0.76159...`
  - `Tanh[{0,1}]  ==> {0, 0.76159...}`

## `TextCount`

- Usage: `TextCount[input, pattern, opts?]`
- Summary: Count regex matches per file and total.
- Examples:
  - `TextCount["a b a", "a"]  ==> <|"total"->2, ...|>`

## `TextDetectEncoding`

- Usage: `TextDetectEncoding[input]`
- Summary: Detect likely text encoding for files.
- Examples:
  - `TextDetectEncoding[{"file1.txt"}]  ==> <|"files"->{<|"file"->..., "encoding"->...|>}|>`

## `TextFilesWithMatch`

- Usage: `TextFilesWithMatch[input, pattern, opts?]`
- Summary: List files that contain the pattern.
- Examples:
  - `TextFilesWithMatch["src", "TODO"]  ==> <|"files"->{...}|>`

## `TextFind`

- Usage: `TextFind[input, pattern, opts?]`
- Summary: Find regex matches across files or text.
- Examples:
  - `TextFind["hello world", "\w+"]  ==> <|"matches"->...|>`

## `TextLines`

- Usage: `TextLines[input, pattern, opts?]`
- Summary: Return matching lines with positions for a pattern.
- Examples:
  - `TextLines["a
TODO b", "TODO"]  ==> <|"lines"->{<|"lineNumber"->2,...|>}|>`

## `TextReplace`

- Usage: `TextReplace[input, pattern, replacement, opts?]`
- Summary: Replace pattern across files; supports dry-run and backups.
- Examples:
  - `TextReplace["src", "foo", "bar", <|"dryRun"->True|>]  ==> <|...|>`

## `TextSearch`

- Usage: `TextSearch[input, query, opts?]`
- Summary: Search text via regex, fuzzy, or index engine.
- Examples:
  - `TextSearch["hello", "hell"]  ==> <|"engine"->"fuzzy", ...|>`

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

## `TraceExport`

- Usage: `TraceExport[format, opts?]`
- Summary: Export spans to a file (json).
- Examples:
  - `Span["build"]; SpanEnd[]; TraceExport["json", <|"Path"->"/tmp/spans.json"|>]  ==> True`

## `TransformerDecoder`

- Usage: `TransformerDecoder[opts?]`
- Summary: Decoder block: self-attn + cross-attn + FFN (single-batch)
- Tags: nn, layer, transformer
- Examples:
  - `Sequential[{TransformerDecoder[<|SeqLen->8, ModelDim->64, NumHeads->8, HiddenDim->256, Causal->True|>]}]`
  - `(* With fixed memory/context *) Sequential[{TransformerDecoder[<|SeqLen->4, ModelDim->32, NumHeads->4, HiddenDim->128, Memory->{{1,0,0,0},{0,1,0,0}}|>]}]`

## `TransformerDecoderStack`

- Usage: `TransformerDecoderStack[opts?]`
- Summary: Stack N decoder blocks (returns Sequential network)
- Tags: nn, network, transformer
- Examples:
  - `TransformerDecoderStack[<|Layers->2, SeqLen->8, ModelDim->64, NumHeads->8, HiddenDim->256, Causal->True|>]`

## `TransformerEncoder`

- Usage: `TransformerEncoder[opts?]`
- Summary: Encoder block: MHA + FFN with residuals and layer norms (single-batch)
- Tags: nn, layer, transformer
- Examples:
  - `Sequential[{TransformerEncoder[<|SeqLen->8, ModelDim->64, NumHeads->8, HiddenDim->256|>]}]`
  - `Sequential[{TransformerEncoder[<|SeqLen->8, ModelDim->64, NumHeads->8, HiddenDim->256, Mask->"Causal"|>]}]`

## `TransformerEncoderDecoder`

- Usage: `TransformerEncoderDecoder[opts?]`
- Summary: Convenience: builds Encoder/Decoder stacks
- Tags: nn, network, transformer
- Examples:
  - `TransformerEncoderDecoder[<|EncLayers->4, DecLayers->4, SeqLen->8, ModelDim->64, NumHeads->8, HiddenDim->256, Causal->True|>]`

## `TransformerEncoderStack`

- Usage: `TransformerEncoderStack[opts?]`
- Summary: Stack N encoder blocks (returns Sequential network)
- Tags: nn, network, transformer
- Examples:
  - `TransformerEncoderStack[<|Layers->2, SeqLen->8, ModelDim->64, NumHeads->8, HiddenDim->256|>]`

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

## `Tune`

- Usage: `Tune[obj, data, opts?]`
- Summary: Hyperparameter search for estimator (dispatched)
- Tags: ml, tune
- Examples:
  - `Tune[Regressor[<|Method->"Linear"|>], train, <|SearchSpace-><|L2->{0.0,1e-2}|>|>]`

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

## `Upsample2D`

- Usage: `Upsample2D[opts?]`
- Summary: Upsample HxW (Nearest/Bilinear)
- Tags: nn, layer
- Examples:
  - `Sequential[{Upsample2D[<|Scale->2, Mode->"Nearest", InputChannels->1, Height->2, Width->2|>]}]`
  - `Sequential[{Upsample2D[<|Scale->2, Mode->"Bilinear", InputChannels->1, Height->2, Width->2|>]}]`

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

## `UsingFile`

- Usage: `UsingFile[path, fn]`
- Summary: Open a file and pass handle to a function (ensures close)
- Examples:
  - `UsingFile["/tmp/x.txt", (f)=>Puts[f, "hi"]]`
  - `ReadFile["/tmp/x.txt"]  ==> "hi"`

## `UuidV4`

- Usage: `UuidV4[]`
- Summary: Generate a random UUID v4 string.
- Examples:
  - `UuidV4[]  ==> "xxxxxxxx-xxxx-4xxx-..."`

## `UuidV7`

- Usage: `UuidV7[]`
- Summary: Generate a time-ordered UUID v7 string.
- Examples:
  - `UuidV7[]  ==> "xxxxxxxx-xxxx-7xxx-..."`

## `VectorCount`

- Usage: `VectorCount[store]`
- Summary: Count items in store
- Tags: vector, info
- Examples:
  - `VectorCount[vs]`

## `VectorDelete`

- Usage: `VectorDelete[store, ids]`
- Summary: Delete items by ids
- Tags: vector, delete
- Examples:
  - `VectorDelete[vs, {"a"}]`

## `VectorReset`

- Usage: `VectorReset[store]`
- Summary: Clear all items in store
- Tags: vector, admin
- Examples:
  - `VectorReset[vs]`

## `VectorSearch`

- Usage: `VectorSearch[store, query, opts]`
- Summary: Search by vector or text (hybrid supported)
- Tags: vector, search
- Examples:
  - `VectorSearch[vs, {0.1,0.2,0.3}]`

## `VectorStore`

- Usage: `VectorStore[optsOrDsn]`
- Summary: Create/open a vector store (memory or DSN)
- Tags: vector, store
- Examples:
  - `vs := VectorStore[<|name->"vs", dims->3|>]`

## `VectorUpsert`

- Usage: `VectorUpsert[store, rows]`
- Summary: Insert or update vectors with metadata
- Tags: vector, upsert
- Examples:
  - `VectorUpsert[vs, {<|id->"a", vec->{0.1,0.2,0.3}|>}]`

## `When`

- Usage: `When[cond, body]`
- Summary: Evaluate body when condition is True (held)
- Examples:
  - `When[True, Print["ok"]]`

## `Window`

- Usage: `Window[type, n, opts?]`
- Summary: Window weights vector by type and size (Hann|Hamming|Blackman).
- Examples:
  - `Window["Hann", 4]  ==> {0.0, 0.5, 0.5, 0.0}`

## `With`

- Usage: `With[<|vars|>, body]`
- Summary: Lexically bind symbols within a body.
- Examples:
  - `With[<|"x"->2|>, x^3]  ==> 8`

## `WithPolicy`

- Usage: `WithPolicy[opts, body]`
- Summary: Evaluate body with temporary tool capabilities.
- Examples:
  - `WithPolicy[<|"Capabilities"->{"net"}|>, HttpGet["https://example.com"]]`

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
