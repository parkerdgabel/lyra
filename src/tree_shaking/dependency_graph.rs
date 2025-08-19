//! Dependency Graph Data Structures
//!
//! Core data structures for representing function dependencies in the stdlib hierarchy.

use std::collections::{HashMap, HashSet, VecDeque};
use serde::{Serialize, Deserialize};

/// Central dependency graph tracking all function relationships
#[derive(Debug, Clone)]
pub struct DependencyGraph {
    /// Function nodes indexed by function name
    nodes: HashMap<String, DependencyNode>,
    
    /// Adjacency list for outgoing dependencies (function -> functions it calls)
    outgoing_edges: HashMap<String, Vec<DependencyEdge>>,
    
    /// Adjacency list for incoming dependencies (function -> functions that call it)
    incoming_edges: HashMap<String, Vec<DependencyEdge>>,
    
    /// Module-level dependencies
    module_dependencies: HashMap<String, HashSet<String>>,
    
    /// Graph metadata
    metadata: GraphMetadata,
}

/// Individual function node in the dependency graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DependencyNode {
    /// Function name (fully qualified)
    pub name: String,
    
    /// Module this function belongs to
    pub module: String,
    
    /// Function metadata
    pub metadata: FunctionMetadata,
    
    /// Usage statistics
    pub usage_stats: NodeUsageStats,
    
    /// Whether this function is an entry point (exported)
    pub is_entry_point: bool,
    
    /// Whether this function is marked as dead code
    pub is_dead_code: bool,
    
    /// Complexity metrics
    pub complexity: ComplexityMetrics,
}

/// Edge representing a dependency between two functions
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DependencyEdge {
    /// Source function (caller)
    pub from: String,
    
    /// Target function (callee)
    pub to: String,
    
    /// Type of dependency
    pub edge_type: DependencyType,
    
    /// Call frequency (if available)
    pub call_frequency: Option<u64>,
    
    /// Call context information
    pub context: CallContext,
    
    /// Whether this dependency is critical for correctness
    pub is_critical: bool,
}

/// Types of dependencies between functions
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DependencyType {
    /// Direct function call
    DirectCall,
    
    /// Conditional call (inside if/match)
    ConditionalCall,
    
    /// Higher-order function call (through function pointer)
    HigherOrderCall,
    
    /// Type dependency (shares types)
    TypeDependency,
    
    /// Macro expansion dependency
    MacroDependency,
    
    /// Module import dependency
    ImportDependency,
    
    /// Transitive dependency (indirect through other functions)
    TransitiveDependency,
}

/// Context information for a function call
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CallContext {
    /// Location in source (if available)
    pub location: Option<String>,
    
    /// Call depth from entry points
    pub call_depth: u32,
    
    /// Whether call is in a loop
    pub in_loop: bool,
    
    /// Whether call is in error handling path
    pub in_error_path: bool,
    
    /// Conditional probability (for conditional calls)
    pub conditional_probability: Option<f64>,
}

/// Function metadata for dependency analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionMetadata {
    /// Function signature
    pub signature: String,
    
    /// Function attributes (Listable, Hold, etc.)
    pub attributes: Vec<String>,
    
    /// Documentation
    pub documentation: Option<String>,
    
    /// Whether function is pure (no side effects)
    pub is_pure: bool,
    
    /// Whether function can be inlined
    pub can_inline: bool,
    
    /// Estimated size (in bytes)
    pub estimated_size: usize,
    
    /// Performance characteristics
    pub performance: PerformanceCharacteristics,
}

/// Performance characteristics of a function
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceCharacteristics {
    /// Estimated execution time complexity
    pub time_complexity: String,
    
    /// Estimated space complexity
    pub space_complexity: String,
    
    /// Whether function allocates memory
    pub allocates_memory: bool,
    
    /// Whether function performs I/O
    pub performs_io: bool,
    
    /// Typical execution time (nanoseconds)
    pub typical_execution_time: Option<u64>,
}

/// Usage statistics for a function node
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct NodeUsageStats {
    /// Total number of calls observed
    pub total_calls: u64,
    
    /// Number of unique callers
    pub unique_callers: usize,
    
    /// Average calls per session
    pub avg_calls_per_session: f64,
    
    /// Peak calls per second
    pub peak_calls_per_second: f64,
    
    /// Last time this function was called
    pub last_called: Option<std::time::SystemTime>,
    
    /// Whether function has been called recently
    pub recently_active: bool,
}

/// Complexity metrics for a function
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplexityMetrics {
    /// Cyclomatic complexity
    pub cyclomatic_complexity: u32,
    
    /// Number of parameters
    pub parameter_count: u32,
    
    /// Number of local variables
    pub local_variable_count: u32,
    
    /// Nesting depth
    pub nesting_depth: u32,
    
    /// Number of branches
    pub branch_count: u32,
    
    /// Lines of code
    pub lines_of_code: u32,
}

/// Metadata about the dependency graph
#[derive(Debug, Clone)]
pub struct GraphMetadata {
    /// Total number of nodes
    pub node_count: usize,
    
    /// Total number of edges
    pub edge_count: usize,
    
    /// Number of strongly connected components
    pub scc_count: usize,
    
    /// Maximum dependency depth
    pub max_depth: u32,
    
    /// Number of entry points
    pub entry_point_count: usize,
    
    /// Number of dead code functions
    pub dead_code_count: usize,
    
    /// Graph density (edges / possible_edges)
    pub density: f64,
    
    /// Time when graph was last updated
    pub last_updated: std::time::SystemTime,
}

impl Default for GraphMetadata {
    fn default() -> Self {
        GraphMetadata {
            node_count: 0,
            edge_count: 0,
            scc_count: 0,
            max_depth: 0,
            entry_point_count: 0,
            dead_code_count: 0,
            density: 0.0,
            last_updated: std::time::SystemTime::now(),
        }
    }
}

impl DependencyGraph {
    /// Create a new empty dependency graph
    pub fn new() -> Self {
        DependencyGraph {
            nodes: HashMap::new(),
            outgoing_edges: HashMap::new(),
            incoming_edges: HashMap::new(),
            module_dependencies: HashMap::new(),
            metadata: GraphMetadata::default(),
        }
    }
    
    /// Add a function node to the graph
    pub fn add_node(&mut self, node: DependencyNode) {
        let name = node.name.clone();
        self.nodes.insert(name.clone(), node);
        self.outgoing_edges.entry(name.clone()).or_insert_with(Vec::new);
        self.incoming_edges.entry(name).or_insert_with(Vec::new);
        self.update_metadata();
    }
    
    /// Add a dependency edge between two functions
    pub fn add_edge(&mut self, edge: DependencyEdge) -> Result<(), String> {
        // Verify both nodes exist
        if !self.nodes.contains_key(&edge.from) {
            return Err(format!("Source node '{}' not found", edge.from));
        }
        if !self.nodes.contains_key(&edge.to) {
            return Err(format!("Target node '{}' not found", edge.to));
        }
        
        // Add to outgoing edges
        self.outgoing_edges.entry(edge.from.clone())
            .or_insert_with(Vec::new)
            .push(edge.clone());
        
        // Add to incoming edges
        self.incoming_edges.entry(edge.to.clone())
            .or_insert_with(Vec::new)
            .push(edge);
        
        self.update_metadata();
        Ok(())
    }
    
    /// Get a function node by name
    pub fn get_node(&self, name: &str) -> Option<&DependencyNode> {
        self.nodes.get(name)
    }
    
    /// Get mutable reference to a function node
    pub fn get_node_mut(&mut self, name: &str) -> Option<&mut DependencyNode> {
        self.nodes.get_mut(name)
    }
    
    /// Get all outgoing dependencies for a function
    pub fn get_dependencies(&self, name: &str) -> Vec<&DependencyEdge> {
        self.outgoing_edges.get(name)
            .map(|edges| edges.iter().collect())
            .unwrap_or_default()
    }
    
    /// Get all incoming dependencies for a function (what calls it)
    pub fn get_dependents(&self, name: &str) -> Vec<&DependencyEdge> {
        self.incoming_edges.get(name)
            .map(|edges| edges.iter().collect())
            .unwrap_or_default()
    }
    
    /// Get all function names in the graph
    pub fn all_functions(&self) -> Vec<&String> {
        self.nodes.keys().collect()
    }
    
    /// Get all entry point functions
    pub fn entry_points(&self) -> Vec<&DependencyNode> {
        self.nodes.values()
            .filter(|node| node.is_entry_point)
            .collect()
    }
    
    /// Get all functions marked as dead code
    pub fn dead_code_functions(&self) -> Vec<&DependencyNode> {
        self.nodes.values()
            .filter(|node| node.is_dead_code)
            .collect()
    }
    
    /// Find strongly connected components (cycles)
    pub fn find_strongly_connected_components(&self) -> Vec<Vec<String>> {
        let mut sccs = Vec::new();
        let mut visited = HashSet::new();
        let mut stack = Vec::new();
        let mut in_stack = HashSet::new();
        let mut low_link = HashMap::new();
        let mut index = HashMap::new();
        let mut current_index = 0;
        
        for node_name in self.nodes.keys() {
            if !visited.contains(node_name) {
                self.tarjan_scc(
                    node_name,
                    &mut visited,
                    &mut stack,
                    &mut in_stack,
                    &mut low_link,
                    &mut index,
                    &mut current_index,
                    &mut sccs,
                );
            }
        }
        
        sccs
    }
    
    /// Perform topological sort of the dependency graph
    pub fn topological_sort(&self) -> Result<Vec<String>, String> {
        let mut in_degree = HashMap::new();
        let mut queue = VecDeque::new();
        let mut result = Vec::new();
        
        // Initialize in-degree count
        for node_name in self.nodes.keys() {
            in_degree.insert(node_name.clone(), 0);
        }
        
        // Calculate in-degrees
        for edges in self.outgoing_edges.values() {
            for edge in edges {
                *in_degree.entry(edge.to.clone()).or_insert(0) += 1;
            }
        }
        
        // Find nodes with zero in-degree
        for (node_name, degree) in &in_degree {
            if *degree == 0 {
                queue.push_back(node_name.clone());
            }
        }
        
        // Process nodes
        while let Some(node_name) = queue.pop_front() {
            result.push(node_name.clone());
            
            // Reduce in-degree of dependent nodes
            if let Some(edges) = self.outgoing_edges.get(&node_name) {
                for edge in edges {
                    let degree = in_degree.get_mut(&edge.to).unwrap();
                    *degree -= 1;
                    if *degree == 0 {
                        queue.push_back(edge.to.clone());
                    }
                }
            }
        }
        
        // Check for cycles
        if result.len() != self.nodes.len() {
            Err("Cycle detected in dependency graph".to_string())
        } else {
            Ok(result)
        }
    }
    
    /// Get all functions reachable from entry points
    pub fn reachable_from_entry_points(&self) -> HashSet<String> {
        let mut reachable = HashSet::new();
        let mut stack = Vec::new();
        
        // Start from all entry points
        for entry_point in self.entry_points() {
            stack.push(entry_point.name.clone());
        }
        
        // DFS to find all reachable functions
        while let Some(current) = stack.pop() {
            if reachable.insert(current.clone()) {
                // Add all dependencies to stack
                if let Some(edges) = self.outgoing_edges.get(&current) {
                    for edge in edges {
                        if !reachable.contains(&edge.to) {
                            stack.push(edge.to.clone());
                        }
                    }
                }
            }
        }
        
        reachable
    }
    
    /// Mark functions as dead code if they're not reachable from entry points
    pub fn mark_dead_code(&mut self) {
        let reachable = self.reachable_from_entry_points();
        
        for (name, node) in self.nodes.iter_mut() {
            node.is_dead_code = !reachable.contains(name);
        }
        
        self.update_metadata();
    }
    
    /// Get graph statistics
    pub fn metadata(&self) -> &GraphMetadata {
        &self.metadata
    }
    
    /// Get total number of nodes
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }
    
    /// Get total number of edges
    pub fn edge_count(&self) -> usize {
        self.outgoing_edges.values()
            .map(|edges| edges.len())
            .sum()
    }
    
    /// Add module dependency
    pub fn add_module_dependency(&mut self, from_module: String, to_module: String) {
        self.module_dependencies.entry(from_module)
            .or_insert_with(HashSet::new)
            .insert(to_module);
    }
    
    /// Get module dependencies
    pub fn get_module_dependencies(&self, module: &str) -> Option<&HashSet<String>> {
        self.module_dependencies.get(module)
    }
    
    /// Get all modules with dependencies
    pub fn all_modules(&self) -> HashSet<String> {
        let mut modules = HashSet::new();
        
        // Add modules from nodes
        for node in self.nodes.values() {
            modules.insert(node.module.clone());
        }
        
        // Add modules from module dependencies
        for (from_module, to_modules) in &self.module_dependencies {
            modules.insert(from_module.clone());
            modules.extend(to_modules.iter().cloned());
        }
        
        modules
    }
    
    /// Get module names as a vector
    pub fn get_module_names(&self) -> Vec<String> {
        self.all_modules().into_iter().collect()
    }
    
    /// Get all functions in a specific module
    pub fn get_functions_in_module(&self, module_name: &str) -> Vec<String> {
        self.nodes.values()
            .filter(|node| node.module == module_name)
            .map(|node| node.name.clone())
            .collect()
    }
    
    /// Find circular dependencies in the graph
    pub fn find_circular_dependencies(&self) -> Vec<Vec<String>> {
        // Simple cycle detection implementation
        // In a real implementation, this would use sophisticated algorithms
        let mut cycles = Vec::new();
        let mut visited = std::collections::HashSet::new();
        let mut rec_stack = std::collections::HashSet::new();
        
        for node_name in self.nodes.keys() {
            if !visited.contains(node_name) {
                if let Some(cycle) = self.dfs_find_cycle(node_name, &mut visited, &mut rec_stack) {
                    cycles.push(cycle);
                }
            }
        }
        
        cycles
    }
    
    fn dfs_find_cycle(
        &self,
        node_name: &str,
        visited: &mut std::collections::HashSet<String>,
        rec_stack: &mut std::collections::HashSet<String>,
    ) -> Option<Vec<String>> {
        visited.insert(node_name.to_string());
        rec_stack.insert(node_name.to_string());
        
        // Use outgoing edges instead of node.dependencies
        if let Some(edges) = self.outgoing_edges.get(node_name) {
            for edge in edges {
                let dep = &edge.to;
                if !visited.contains(dep) {
                    if let Some(cycle) = self.dfs_find_cycle(dep, visited, rec_stack) {
                        return Some(cycle);
                    }
                } else if rec_stack.contains(dep) {
                    // Found a cycle
                    return Some(vec![node_name.to_string(), dep.clone()]);
                }
            }
        }
        
        rec_stack.remove(node_name);
        None
    }
    
    // Private helper methods
    
    fn update_metadata(&mut self) {
        self.metadata.node_count = self.nodes.len();
        self.metadata.edge_count = self.edge_count();
        self.metadata.entry_point_count = self.nodes.values()
            .filter(|node| node.is_entry_point)
            .count();
        self.metadata.dead_code_count = self.nodes.values()
            .filter(|node| node.is_dead_code)
            .count();
        
        // Calculate graph density
        let max_edges = self.nodes.len() * (self.nodes.len() - 1);
        self.metadata.density = if max_edges > 0 {
            self.metadata.edge_count as f64 / max_edges as f64
        } else {
            0.0
        };
        
        self.metadata.last_updated = std::time::SystemTime::now();
    }
    
    fn tarjan_scc(
        &self,
        node: &str,
        visited: &mut HashSet<String>,
        stack: &mut Vec<String>,
        in_stack: &mut HashSet<String>,
        low_link: &mut HashMap<String, usize>,
        index: &mut HashMap<String, usize>,
        current_index: &mut usize,
        sccs: &mut Vec<Vec<String>>,
    ) {
        visited.insert(node.to_string());
        index.insert(node.to_string(), *current_index);
        low_link.insert(node.to_string(), *current_index);
        *current_index += 1;
        stack.push(node.to_string());
        in_stack.insert(node.to_string());
        
        // Visit neighbors
        if let Some(edges) = self.outgoing_edges.get(node) {
            for edge in edges {
                let neighbor = &edge.to;
                if !visited.contains(neighbor) {
                    self.tarjan_scc(neighbor, visited, stack, in_stack, low_link, index, current_index, sccs);
                    let neighbor_low = *low_link.get(neighbor).unwrap();
                    let current_low = *low_link.get(node).unwrap();
                    low_link.insert(node.to_string(), current_low.min(neighbor_low));
                } else if in_stack.contains(neighbor) {
                    let neighbor_index = *index.get(neighbor).unwrap();
                    let current_low = *low_link.get(node).unwrap();
                    low_link.insert(node.to_string(), current_low.min(neighbor_index));
                }
            }
        }
        
        // If node is a root of SCC
        if low_link.get(node) == index.get(node) {
            let mut scc = Vec::new();
            loop {
                let w = stack.pop().unwrap();
                in_stack.remove(&w);
                scc.push(w.clone());
                if w == node {
                    break;
                }
            }
            sccs.push(scc);
        }
    }
}

impl Default for DependencyGraph {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_node(name: &str, module: &str) -> DependencyNode {
        DependencyNode {
            name: name.to_string(),
            module: module.to_string(),
            metadata: FunctionMetadata {
                signature: format!("{}() -> Value", name),
                attributes: vec![],
                documentation: None,
                is_pure: true,
                can_inline: true,
                estimated_size: 100,
                performance: PerformanceCharacteristics {
                    time_complexity: "O(1)".to_string(),
                    space_complexity: "O(1)".to_string(),
                    allocates_memory: false,
                    performs_io: false,
                    typical_execution_time: Some(1000),
                },
            },
            usage_stats: NodeUsageStats::default(),
            is_entry_point: false,
            is_dead_code: false,
            complexity: ComplexityMetrics {
                cyclomatic_complexity: 1,
                parameter_count: 0,
                local_variable_count: 0,
                nesting_depth: 1,
                branch_count: 0,
                lines_of_code: 5,
            },
        }
    }

    fn create_test_edge(from: &str, to: &str, edge_type: DependencyType) -> DependencyEdge {
        DependencyEdge {
            from: from.to_string(),
            to: to.to_string(),
            edge_type,
            call_frequency: None,
            context: CallContext {
                location: None,
                call_depth: 1,
                in_loop: false,
                in_error_path: false,
                conditional_probability: None,
            },
            is_critical: true,
        }
    }

    #[test]
    fn test_dependency_graph_creation() {
        let graph = DependencyGraph::new();
        assert_eq!(graph.node_count(), 0);
        assert_eq!(graph.edge_count(), 0);
    }

    #[test]
    fn test_add_nodes() {
        let mut graph = DependencyGraph::new();
        
        let node1 = create_test_node("function1", "std::math");
        let node2 = create_test_node("function2", "std::list");
        
        graph.add_node(node1);
        graph.add_node(node2);
        
        assert_eq!(graph.node_count(), 2);
        assert!(graph.get_node("function1").is_some());
        assert!(graph.get_node("function2").is_some());
    }

    #[test]
    fn test_add_edges() {
        let mut graph = DependencyGraph::new();
        
        // Add nodes first
        graph.add_node(create_test_node("function1", "std::math"));
        graph.add_node(create_test_node("function2", "std::list"));
        
        // Add edge
        let edge = create_test_edge("function1", "function2", DependencyType::DirectCall);
        assert!(graph.add_edge(edge).is_ok());
        
        assert_eq!(graph.edge_count(), 1);
        
        // Test dependencies
        let deps = graph.get_dependencies("function1");
        assert_eq!(deps.len(), 1);
        assert_eq!(deps[0].to, "function2");
        
        // Test dependents
        let dependents = graph.get_dependents("function2");
        assert_eq!(dependents.len(), 1);
        assert_eq!(dependents[0].from, "function1");
    }

    #[test]
    fn test_invalid_edge() {
        let mut graph = DependencyGraph::new();
        
        // Try to add edge without nodes
        let edge = create_test_edge("nonexistent1", "nonexistent2", DependencyType::DirectCall);
        assert!(graph.add_edge(edge).is_err());
    }

    #[test]
    fn test_entry_points() {
        let mut graph = DependencyGraph::new();
        
        let mut node1 = create_test_node("entry1", "std::math");
        node1.is_entry_point = true;
        
        let node2 = create_test_node("internal", "std::math");
        
        graph.add_node(node1);
        graph.add_node(node2);
        
        let entry_points = graph.entry_points();
        assert_eq!(entry_points.len(), 1);
        assert_eq!(entry_points[0].name, "entry1");
    }

    #[test]
    fn test_topological_sort() {
        let mut graph = DependencyGraph::new();
        
        // Create a simple dependency chain: A -> B -> C
        graph.add_node(create_test_node("A", "test"));
        graph.add_node(create_test_node("B", "test"));
        graph.add_node(create_test_node("C", "test"));
        
        graph.add_edge(create_test_edge("A", "B", DependencyType::DirectCall)).unwrap();
        graph.add_edge(create_test_edge("B", "C", DependencyType::DirectCall)).unwrap();
        
        let sorted = graph.topological_sort().unwrap();
        assert_eq!(sorted.len(), 3);
        
        // A should come before B, B should come before C
        let a_pos = sorted.iter().position(|x| x == "A").unwrap();
        let b_pos = sorted.iter().position(|x| x == "B").unwrap();
        let c_pos = sorted.iter().position(|x| x == "C").unwrap();
        
        assert!(a_pos < b_pos);
        assert!(b_pos < c_pos);
    }

    #[test]
    fn test_dead_code_detection() {
        let mut graph = DependencyGraph::new();
        
        // Create entry point and reachable function
        let mut entry = create_test_node("entry", "test");
        entry.is_entry_point = true;
        graph.add_node(entry);
        
        let reachable = create_test_node("reachable", "test");
        graph.add_node(reachable);
        
        let unreachable = create_test_node("unreachable", "test");
        graph.add_node(unreachable);
        
        // Connect entry to reachable
        graph.add_edge(create_test_edge("entry", "reachable", DependencyType::DirectCall)).unwrap();
        
        // Mark dead code
        graph.mark_dead_code();
        
        // Check results
        assert!(!graph.get_node("entry").unwrap().is_dead_code);
        assert!(!graph.get_node("reachable").unwrap().is_dead_code);
        assert!(graph.get_node("unreachable").unwrap().is_dead_code);
    }

    #[test]
    fn test_module_dependencies() {
        let mut graph = DependencyGraph::new();
        
        graph.add_module_dependency("std::math".to_string(), "std::tensor".to_string());
        graph.add_module_dependency("std::list".to_string(), "std::math".to_string());
        
        let math_deps = graph.get_module_dependencies("std::math");
        assert!(math_deps.is_some());
        assert!(math_deps.unwrap().contains("std::tensor"));
        
        let all_modules = graph.all_modules();
        assert!(all_modules.contains("std::math"));
        assert!(all_modules.contains("std::tensor"));
        assert!(all_modules.contains("std::list"));
    }

    #[test]
    fn test_graph_metadata() {
        let mut graph = DependencyGraph::new();
        
        graph.add_node(create_test_node("func1", "test"));
        graph.add_node(create_test_node("func2", "test"));
        
        let metadata = graph.metadata();
        assert_eq!(metadata.node_count, 2);
        assert_eq!(metadata.edge_count, 0);
        assert_eq!(metadata.entry_point_count, 0);
        assert_eq!(metadata.dead_code_count, 0);
    }
}