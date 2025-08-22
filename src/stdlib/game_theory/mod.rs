//! Game Theory Module for Lyra
//!
//! This module provides comprehensive game theory and mechanism design capabilities
//! including equilibrium analysis, auction mechanisms, and strategic algorithms.
//!
//! # Module Structure
//! - `equilibrium`: Nash equilibrium, correlated equilibrium, evolutionary stable strategies
//! - `auctions`: First-price, second-price, Vickrey, combinatorial, English, Dutch auctions
//! - `mechanisms`: VCG mechanisms, optimal auction design, matching algorithms
//!
//! # Design Principles
//! - All game theory concepts implemented as Foreign objects using LyObj wrapper
//! - Comprehensive error handling for game theory edge cases  
//! - Linear programming and optimization algorithms for equilibrium computation
//! - Monte Carlo methods for complex games and mechanism analysis
//! - Integration with VM through Value::LyObj wrapper pattern

pub mod equilibrium;
pub mod auctions;  
pub mod mechanisms;

// Re-export all public functions and types
pub use equilibrium::*;
pub use auctions::*;
pub use mechanisms::*;

/// Game Theory Module Documentation
///
/// This module implements a comprehensive suite of game theory and mechanism design
/// algorithms following academic standards and best practices.
///
/// ## Equilibrium Concepts
/// - **Nash Equilibrium**: Pure and mixed strategy Nash equilibria for n-player games
/// - **Correlated Equilibrium**: Welfare-maximizing correlated equilibria
/// - **Evolutionary Stable Strategy**: ESS computation for symmetric games
/// - **Iterated Dominance**: Elimination of dominated strategies
///
/// ## Auction Mechanisms  
/// - **First-Price Auctions**: Sealed-bid first-price auctions with efficiency analysis
/// - **Second-Price Auctions**: Vickrey auctions with truthfulness guarantees
/// - **Multi-Unit Vickrey**: VCG-based multi-unit auctions  
/// - **Combinatorial Auctions**: Winner determination for package bidding
/// - **English Auctions**: Ascending price auction simulation
/// - **Dutch Auctions**: Descending price auction simulation
///
/// ## Mechanism Design
/// - **VCG Mechanisms**: Vickrey-Clarke-Groves mechanisms with incentive compatibility
/// - **Optimal Auctions**: Revenue-maximizing auction design (Myerson's theorem)
/// - **Revenue Maximization**: Virtual valuations and ironing procedures
/// - **Stable Marriage**: Gale-Shapley deferred acceptance algorithm
/// - **Assignment Problems**: Hungarian algorithm for minimum cost matching
/// - **Stable Assignment**: Two-sided matching with preferences
///
/// ## Foreign Object Architecture
/// All game theory structures are implemented as Foreign objects:
/// - `Game`: Matrix game representation with payoff matrices
/// - `NashEquilibrium`: Nash equilibrium result with strategies and payoffs
/// - `CorrelatedEquilibrium`: Correlated equilibrium with joint distribution
/// - `EvolutionaryStableStrategy`: ESS with fitness and stability analysis
/// - `FirstPriceAuction`: First-price auction result with efficiency metrics
/// - `SecondPriceAuction`: Second-price auction with truthfulness verification
/// - `VickreyAuction`: Multi-unit Vickrey auction with VCG payments
/// - `CombinatorialAuction`: Combinatorial auction with winner determination
/// - `VCGMechanism`: VCG mechanism with incentive compatibility
/// - `OptimalAuction`: Optimal auction design with reserve prices
/// - `StableMarriage`: Stable marriage matching with stability verification
/// - `AssignmentProblem`: Assignment problem solution with optimality
///
/// ## Mathematical Foundations
/// The module implements algorithms based on:
/// - Linear complementarity for Nash equilibrium computation
/// - Replicator dynamics for evolutionary stable strategies  
/// - Linear programming for mechanism design optimization
/// - Hungarian algorithm for assignment problems
/// - Gale-Shapley algorithm for stable matching
/// - Myerson's optimal auction theory
/// - VCG payment computation for incentive compatibility
///
/// ## Error Handling
/// Comprehensive error handling covers:
/// - Invalid game matrices (non-rectangular, mismatched dimensions)
/// - Degenerate auction inputs (no bidders, invalid valuations)
/// - Inconsistent preference structures in matching
/// - Numerical instability in equilibrium computation
/// - Infeasible mechanism design constraints
///
/// ## Performance Characteristics  
/// - Nash equilibrium: O(2^n) worst case, polynomial for special classes
/// - Auction mechanisms: O(n log n) for most auction types
/// - Combinatorial auctions: NP-hard winner determination (greedy approximation)
/// - Stable marriage: O(n²) using Gale-Shapley algorithm
/// - Assignment problems: O(n³) using Hungarian algorithm
/// - VCG mechanisms: O(n * 2^m) for n agents and m outcomes
///
/// ## Integration Examples
/// ```wolfram
/// (* Nash equilibrium computation *)
/// game = {{{3,1}, {0,2}}, {{2,0}, {1,3}}};
/// nash = NashEquilibrium[game];
/// nash.Strategies()  // Mixed strategies for each player
/// nash.Payoffs()     // Expected payoffs
/// 
/// (* Vickrey auction *)
/// valuations = {{20,10}, {15,25}, {30,5}};
/// auction = VickreyAuction[valuations];
/// auction.Allocation()  // Item allocation to bidders
/// auction.Payments()    // VCG payments
/// 
/// (* Stable marriage *)
/// menPrefs = {{1,0,2}, {0,2,1}, {2,1,0}};
/// womenPrefs = {{1,2,0}, {0,2,1}, {2,0,1}};
/// marriage = StableMarriage[menPrefs, womenPrefs];
/// marriage.Matching()   // Stable matching pairs
/// ```
///
/// ## Testing and Validation
/// The module includes comprehensive test suites covering:
/// - Known game theory results for validation  
/// - Property-based tests for equilibrium properties
/// - Integration tests with VM value system
/// - Performance tests for large games and auctions
/// - Edge case handling and error conditions
/// - Mechanism design optimality verification

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vm::Value;
    
    #[test]
    fn test_module_integration() {
        // Test that all main functions can be called without compilation errors
        
        // Create test payoff matrix
        let payoff_matrix = Value::List(vec![
            Value::List(vec![
                Value::List(vec![Value::Real(3.0), Value::Real(1.0)]),
                Value::List(vec![Value::Real(0.0), Value::Real(2.0)]),
            ]),
            Value::List(vec![
                Value::List(vec![Value::Real(2.0), Value::Real(0.0)]),
                Value::List(vec![Value::Real(1.0), Value::Real(3.0)]),
            ]),
        ]);
        
        // Test equilibrium functions
        let nash_result = nash_equilibrium(&[payoff_matrix.clone()]);
        assert!(nash_result.is_ok(), "Nash equilibrium should succeed");
        
        let corr_result = correlated_equilibrium(&[payoff_matrix.clone()]);
        assert!(corr_result.is_ok(), "Correlated equilibrium should succeed");
        
        // Test auction functions
        let valuations = Value::List(vec![Value::Real(100.0), Value::Real(80.0)]);
        let bids = Value::List(vec![Value::Real(75.0), Value::Real(60.0)]);
        
        let fp_result = first_price_auction(&[valuations.clone(), bids.clone()]);
        assert!(fp_result.is_ok(), "First-price auction should succeed");
        
        let sp_result = second_price_auction(&[valuations, bids]);
        assert!(sp_result.is_ok(), "Second-price auction should succeed");
        
        // Test mechanism functions
        let agent_valuations = Value::List(vec![
            Value::List(vec![Value::Real(20.0), Value::Real(10.0)]),
            Value::List(vec![Value::Real(15.0), Value::Real(25.0)]),
        ]);
        
        let vcg_result = vcg_mechanism(&[agent_valuations]);
        assert!(vcg_result.is_ok(), "VCG mechanism should succeed");
        
        // Test matching functions
        let men_prefs = Value::List(vec![
            Value::List(vec![Value::Integer(0), Value::Integer(1)]),
            Value::List(vec![Value::Integer(1), Value::Integer(0)]),
        ]);
        let women_prefs = Value::List(vec![
            Value::List(vec![Value::Integer(1), Value::Integer(0)]),
            Value::List(vec![Value::Integer(0), Value::Integer(1)]),
        ]);
        
        let marriage_result = stable_marriage(&[men_prefs, women_prefs]);
        assert!(marriage_result.is_ok(), "Stable marriage should succeed");
    }
    
    #[test]
    fn test_foreign_object_types() {
        // Test that Foreign objects have correct type names
        use crate::foreign::Foreign;
        
        let game = Game::new(vec![
            vec![vec![1.0, 0.0], vec![0.0, 1.0]],
            vec![vec![1.0, 0.0], vec![0.0, 1.0]],
        ]).unwrap();
        
        let nash = NashEquilibrium::compute_2player_nash(game).unwrap();
        assert_eq!(nash.type_name(), "NashEquilibrium");
        
        let fp_auction = FirstPriceAuction::conduct(vec![100.0, 80.0], vec![75.0, 60.0]).unwrap();
        assert_eq!(fp_auction.type_name(), "FirstPriceAuction");
        
        let vcg = VCGMechanism::implement(vec![
            vec![20.0, 10.0],
            vec![15.0, 25.0],
        ]).unwrap();
        assert_eq!(vcg.type_name(), "VCGMechanism");
        
        let marriage = StableMarriage::solve(
            vec![vec![0, 1], vec![1, 0]],
            vec![vec![1, 0], vec![0, 1]],
        ).unwrap();
        assert_eq!(marriage.type_name(), "StableMarriage");
    }
    
    #[test]
    fn test_error_handling() {
        // Test that functions properly handle invalid inputs
        
        // Empty payoff matrix should fail
        let empty_matrix = Value::List(vec![]);
        let nash_result = nash_equilibrium(&[empty_matrix]);
        assert!(nash_result.is_err(), "Should reject empty matrix");
        
        // Mismatched valuations and bids should fail
        let valuations = Value::List(vec![Value::Real(100.0), Value::Real(80.0)]);
        let bids = Value::List(vec![Value::Real(75.0)]); // Wrong length
        let fp_result = first_price_auction(&[valuations, bids]);
        assert!(fp_result.is_err(), "Should reject mismatched lengths");
        
        // Invalid preference lists should fail
        let invalid_prefs = Value::List(vec![
            Value::String("invalid".to_string())  // Should be list of integers
        ]);
        let women_prefs = Value::List(vec![
            Value::List(vec![Value::Integer(0)])
        ]);
        let marriage_result = stable_marriage(&[invalid_prefs, women_prefs]);
        assert!(marriage_result.is_err(), "Should reject invalid preferences");
    }
}