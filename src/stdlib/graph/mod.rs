//! Graph Theory & Network Analysis Module
//!
//! Comprehensive graph algorithms and network analysis tools for Lyra.
//! This module provides both fundamental graph operations and advanced
//! algorithms for network analysis, community detection, and centrality measures.

pub mod core;
pub mod advanced;

// Re-export main types and functions (avoiding conflicts)
pub use core::Graph;
pub use advanced::{PageRankResult, CommunityResult, CentralityResult};

use crate::vm::{Value, VmResult};
use std::collections::HashMap;

/// Register all graph algorithm functions for the standard library
pub fn register_graph_functions() -> HashMap<String, fn(&[Value]) -> VmResult<Value>> {
    let mut functions = HashMap::new();
    
    // Core graph functions (from graph/core.rs)
    functions.insert("Graph".to_string(), core::graph as fn(&[Value]) -> VmResult<Value>);
    functions.insert("DirectedGraph".to_string(), core::directed_graph as fn(&[Value]) -> VmResult<Value>);
    functions.insert("AdjacencyMatrix".to_string(), core::adjacency_matrix as fn(&[Value]) -> VmResult<Value>);
    functions.insert("GraphProperties".to_string(), core::graph_properties as fn(&[Value]) -> VmResult<Value>);
    
    // Traversal algorithms
    functions.insert("DepthFirstSearch".to_string(), core::depth_first_search as fn(&[Value]) -> VmResult<Value>);
    functions.insert("BreadthFirstSearch".to_string(), core::breadth_first_search as fn(&[Value]) -> VmResult<Value>);
    
    // Shortest path algorithms
    functions.insert("Dijkstra".to_string(), core::dijkstra as fn(&[Value]) -> VmResult<Value>);
    
    // Connectivity algorithms
    functions.insert("ConnectedComponents".to_string(), core::connected_components as fn(&[Value]) -> VmResult<Value>);
    
    // Structural algorithms
    functions.insert("MinimumSpanningTree".to_string(), core::minimum_spanning_tree as fn(&[Value]) -> VmResult<Value>);
    functions.insert("MaximumFlow".to_string(), core::maximum_flow as fn(&[Value]) -> VmResult<Value>);
    functions.insert("ArticulationPoints".to_string(), core::articulation_points as fn(&[Value]) -> VmResult<Value>);
    functions.insert("Bridges".to_string(), core::bridges as fn(&[Value]) -> VmResult<Value>);
    functions.insert("GraphColoring".to_string(), core::graph_coloring as fn(&[Value]) -> VmResult<Value>);
    
    // Path algorithms
    functions.insert("HamiltonianPath".to_string(), core::hamiltonian_path as fn(&[Value]) -> VmResult<Value>);
    functions.insert("EulerianPath".to_string(), core::eulerian_path as fn(&[Value]) -> VmResult<Value>);
    
    // Optimization problems
    functions.insert("VertexCover".to_string(), core::vertex_cover as fn(&[Value]) -> VmResult<Value>);
    functions.insert("IndependentSet".to_string(), core::independent_set as fn(&[Value]) -> VmResult<Value>);
    
    // Core centrality functions (these exist in core.rs)
    functions.insert("CoreBetweennessCentrality".to_string(), core::betweenness_centrality as fn(&[Value]) -> VmResult<Value>);
    functions.insert("CoreClosenessCentrality".to_string(), core::closeness_centrality as fn(&[Value]) -> VmResult<Value>);
    functions.insert("CorePageRank".to_string(), core::pagerank as fn(&[Value]) -> VmResult<Value>);
    functions.insert("HITS".to_string(), core::hits as fn(&[Value]) -> VmResult<Value>);
    functions.insert("CoreCommunityDetection".to_string(), core::community_detection as fn(&[Value]) -> VmResult<Value>);
    
    // Graph isomorphism
    functions.insert("GraphIsomorphism".to_string(), core::graph_isomorphism as fn(&[Value]) -> VmResult<Value>);
    
    // Advanced algorithms (from graph/advanced.rs) - using different names to avoid conflicts
    functions.insert("AdvancedPageRank".to_string(), advanced::page_rank_function as fn(&[Value]) -> VmResult<Value>);
    functions.insert("LouvainCommunityDetection".to_string(), advanced::community_detection_function as fn(&[Value]) -> VmResult<Value>);
    functions.insert("AdvancedBetweennessCentrality".to_string(), advanced::betweenness_centrality_function as fn(&[Value]) -> VmResult<Value>);
    functions.insert("AdvancedClosenessCentrality".to_string(), advanced::closeness_centrality_function as fn(&[Value]) -> VmResult<Value>);
    
    functions
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vm::Value;

    #[test]
    fn test_function_registration() {
        let functions = register_graph_functions();
        
        // Test that core functions are registered
        assert!(functions.contains_key("Graph"));
        assert!(functions.contains_key("AddVertex"));
        assert!(functions.contains_key("AddEdge"));
        assert!(functions.contains_key("DijkstraShortestPath"));
        
        // Test that advanced functions are registered
        assert!(functions.contains_key("PageRank"));
        assert!(functions.contains_key("CommunityDetection"));
        assert!(functions.contains_key("BetweennessCentrality"));
        assert!(functions.contains_key("ClosenessCentrality"));
    }

    #[test]
    fn test_integration_pagerank_with_simple_graph() {
        let functions = register_graph_functions();
        let graph_fn = functions["Graph"];
        let add_vertex_fn = functions["AddVertex"];
        let add_edge_fn = functions["AddEdge"];
        let pagerank_fn = functions["PageRank"];

        // Create a simple triangle graph
        let graph = graph_fn(&[Value::Boolean(true)]).unwrap(); // directed graph
        
        // Add vertices
        let _ = add_vertex_fn(&[graph.clone(), Value::Integer(1)]).unwrap();
        let _ = add_vertex_fn(&[graph.clone(), Value::Integer(2)]).unwrap();
        let _ = add_vertex_fn(&[graph.clone(), Value::Integer(3)]).unwrap();
        
        // Add edges to form a cycle
        let _ = add_edge_fn(&[graph.clone(), Value::Integer(1), Value::Integer(2), Value::Real(1.0)]).unwrap();
        let _ = add_edge_fn(&[graph.clone(), Value::Integer(2), Value::Integer(3), Value::Real(1.0)]).unwrap();
        let _ = add_edge_fn(&[graph.clone(), Value::Integer(3), Value::Integer(1), Value::Real(1.0)]).unwrap();
        
        // Run PageRank
        let result = pagerank_fn(&[
            graph,
            Value::Real(0.85), // damping
            Value::Real(1e-6), // tolerance
            Value::Integer(100) // max iterations
        ]);
        
        assert!(result.is_ok());
        if let Ok(Value::LyObj(obj)) = result {
            assert_eq!(obj.type_name(), "PageRankResult");
        }
    }
}