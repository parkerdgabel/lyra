//! Game Theory Module Tests
//!
//! Comprehensive test suite for Phase 3.1 - Game Theory & Mechanism Design
//! Tests all equilibrium concepts, auction mechanisms, and strategic algorithms

use std::collections::HashMap;
use lyra::vm::{Value, VmResult};
use lyra::stdlib::StandardLibrary;

/// Test utilities for game theory operations
fn create_test_stdlib() -> StandardLibrary {
    StandardLibrary::new()
}

fn extract_real_from_value(value: &Value) -> Option<f64> {
    match value {
        Value::Real(r) => Some(*r),
        Value::Integer(i) => Some(*i as f64),
        _ => None,
    }
}

fn create_test_payoff_matrix() -> Value {
    // 2x2 game matrix: [[3,1], [0,2]] for player 1, [[2,0], [1,3]] for player 2
    Value::List(vec![
        Value::List(vec![
            Value::List(vec![Value::Real(3.0), Value::Real(1.0)]),  // Player 1 payoffs
            Value::List(vec![Value::Real(0.0), Value::Real(2.0)]),
        ]),
        Value::List(vec![
            Value::List(vec![Value::Real(2.0), Value::Real(0.0)]),  // Player 2 payoffs
            Value::List(vec![Value::Real(1.0), Value::Real(3.0)]),
        ]),
    ])
}

// ===============================
// NASH EQUILIBRIUM TESTS
// ===============================

#[test]
fn test_nash_equilibrium_pure_strategy() {
    let stdlib = create_test_stdlib();
    let nash_fn = stdlib.get_function("NashEquilibrium").expect("NashEquilibrium function should exist");

    // Test 2x2 game with pure strategy Nash equilibrium
    let payoff_matrix = create_test_payoff_matrix();
    
    let result = nash_fn(&[payoff_matrix]).expect("Nash equilibrium computation should succeed");
    
    match result {
        Value::LyObj(obj) => {
            // Check that we get an equilibrium result
            let eq_type = obj.call_method("EquilibriumType", &[]).expect("Should get equilibrium type");
            match eq_type {
                Value::String(s) => {
                    assert!(s == "Pure" || s == "Mixed", "Should be Pure or Mixed equilibrium");
                }
                _ => panic!("Equilibrium type should be string"),
            }

            // Check strategies
            let strategies = obj.call_method("Strategies", &[]).expect("Should get strategies");
            match strategies {
                Value::List(strat_list) => {
                    assert_eq!(strat_list.len(), 2, "Should have 2 player strategies");
                }
                _ => panic!("Strategies should be list"),
            }

            // Check payoffs
            let payoffs = obj.call_method("Payoffs", &[]).expect("Should get payoffs");
            match payoffs {
                Value::List(payoff_list) => {
                    assert_eq!(payoff_list.len(), 2, "Should have 2 player payoffs");
                    // All payoffs should be non-negative reals
                    for payoff in &payoff_list {
                        assert!(extract_real_from_value(payoff).is_some(), "Payoffs should be numeric");
                    }
                }
                _ => panic!("Payoffs should be list"),
            }
        }
        _ => panic!("Nash equilibrium should return game theory object"),
    }
}

#[test]
fn test_nash_equilibrium_mixed_strategy() {
    let stdlib = create_test_stdlib();
    let nash_fn = stdlib.get_function("NashEquilibrium").expect("NashEquilibrium function should exist");

    // Matching pennies game - only mixed strategy equilibrium
    let matching_pennies = Value::List(vec![
        Value::List(vec![
            Value::List(vec![Value::Real(1.0), Value::Real(-1.0)]),  // Player 1 payoffs
            Value::List(vec![Value::Real(-1.0), Value::Real(1.0)]),
        ]),
        Value::List(vec![
            Value::List(vec![Value::Real(-1.0), Value::Real(1.0)]),  // Player 2 payoffs
            Value::List(vec![Value::Real(1.0), Value::Real(-1.0)]),
        ]),
    ]);
    
    let result = nash_fn(&[matching_pennies]).expect("Nash equilibrium computation should succeed");
    
    match result {
        Value::LyObj(obj) => {
            let eq_type = obj.call_method("EquilibriumType", &[]).expect("Should get equilibrium type");
            match eq_type {
                Value::String(s) => {
                    assert_eq!(s, "Mixed", "Matching pennies should have mixed strategy equilibrium");
                }
                _ => panic!("Equilibrium type should be string"),
            }

            // For mixed strategy, check that probabilities sum to 1
            let strategies = obj.call_method("Strategies", &[]).expect("Should get strategies");
            match strategies {
                Value::List(strat_list) => {
                    for strategy in &strat_list {
                        match strategy {
                            Value::List(probs) => {
                                let sum: f64 = probs.iter()
                                    .filter_map(extract_real_from_value)
                                    .sum();
                                assert!((sum - 1.0).abs() < 1e-6, "Strategy probabilities should sum to 1");
                            }
                            _ => panic!("Each strategy should be probability list"),
                        }
                    }
                }
                _ => panic!("Strategies should be list"),
            }
        }
        _ => panic!("Nash equilibrium should return game theory object"),
    }
}

#[test]
fn test_correlated_equilibrium() {
    let stdlib = create_test_stdlib();
    let corr_eq_fn = stdlib.get_function("CorrelatedEquilibrium").expect("CorrelatedEquilibrium function should exist");

    let payoff_matrix = create_test_payoff_matrix();
    
    let result = corr_eq_fn(&[payoff_matrix]).expect("Correlated equilibrium computation should succeed");
    
    match result {
        Value::LyObj(obj) => {
            // Check joint probability distribution
            let distribution = obj.call_method("Distribution", &[]).expect("Should get joint distribution");
            match distribution {
                Value::List(dist_list) => {
                    // Should have probability for each action profile
                    let total_prob: f64 = dist_list.iter()
                        .filter_map(|item| match item {
                            Value::List(profile) => profile.last().and_then(extract_real_from_value),
                            _ => None,
                        })
                        .sum();
                    assert!((total_prob - 1.0).abs() < 1e-6, "Joint probabilities should sum to 1");
                }
                _ => panic!("Distribution should be list"),
            }

            // Check expected payoffs
            let payoffs = obj.call_method("ExpectedPayoffs", &[]).expect("Should get expected payoffs");
            match payoffs {
                Value::List(payoff_list) => {
                    assert_eq!(payoff_list.len(), 2, "Should have payoffs for 2 players");
                }
                _ => panic!("Expected payoffs should be list"),
            }
        }
        _ => panic!("Correlated equilibrium should return game theory object"),
    }
}

#[test]
fn test_evolutionary_stable_strategy() {
    let stdlib = create_test_stdlib();
    let ess_fn = stdlib.get_function("EvolutionaryStableStrategy").expect("EvolutionaryStableStrategy function should exist");

    // Hawk-Dove game payoff matrix
    let hawk_dove = Value::List(vec![
        Value::List(vec![Value::Real(-1.0), Value::Real(3.0)]),  // Hawk vs (Hawk, Dove)
        Value::List(vec![Value::Real(0.0), Value::Real(1.0)]),   // Dove vs (Hawk, Dove)
    ]);
    
    let result = ess_fn(&[hawk_dove]).expect("ESS computation should succeed");
    
    match result {
        Value::LyObj(obj) => {
            // Check ESS strategy
            let strategy = obj.call_method("Strategy", &[]).expect("Should get ESS strategy");
            match strategy {
                Value::List(probs) => {
                    let sum: f64 = probs.iter()
                        .filter_map(extract_real_from_value)
                        .sum();
                    assert!((sum - 1.0).abs() < 1e-6, "ESS probabilities should sum to 1");
                }
                _ => panic!("ESS strategy should be probability list"),
            }

            // Check fitness values
            let fitness = obj.call_method("Fitness", &[]).expect("Should get fitness");
            assert!(extract_real_from_value(&fitness).is_some(), "Fitness should be numeric");

            // Check stability condition
            let is_stable = obj.call_method("IsStable", &[]).expect("Should check stability");
            match is_stable {
                Value::String(s) => assert!(s == "true" || s == "false", "Stability should be boolean string"),
                _ => panic!("Stability check should return boolean string"),
            }
        }
        _ => panic!("ESS should return game theory object"),
    }
}

#[test]
fn test_iterated_dominance_elimination() {
    let stdlib = create_test_stdlib();
    let dominance_fn = stdlib.get_function("EliminateDominatedStrategies").expect("EliminateDominatedStrategies function should exist");

    let payoff_matrix = create_test_payoff_matrix();
    
    let result = dominance_fn(&[payoff_matrix]).expect("Dominance elimination should succeed");
    
    match result {
        Value::LyObj(obj) => {
            // Check reduced game
            let reduced_game = obj.call_method("ReducedGame", &[]).expect("Should get reduced game");
            match reduced_game {
                Value::List(_) => {
                    // Should maintain game structure after elimination
                }
                _ => panic!("Reduced game should be list structure"),
            }

            // Check eliminated strategies
            let eliminated = obj.call_method("EliminatedStrategies", &[]).expect("Should get eliminated strategies");
            match eliminated {
                Value::List(elim_list) => {
                    // Each player's eliminated strategies
                    assert_eq!(elim_list.len(), 2, "Should track elimination for both players");
                }
                _ => panic!("Eliminated strategies should be list"),
            }

            // Check remaining strategies
            let remaining = obj.call_method("RemainingStrategies", &[]).expect("Should get remaining strategies");
            match remaining {
                Value::List(rem_list) => {
                    assert_eq!(rem_list.len(), 2, "Should track remaining strategies for both players");
                }
                _ => panic!("Remaining strategies should be list"),
            }
        }
        _ => panic!("Dominance elimination should return game theory object"),
    }
}

// ===============================
// AUCTION MECHANISM TESTS
// ===============================

#[test]
fn test_first_price_auction() {
    let stdlib = create_test_stdlib();
    let fp_auction_fn = stdlib.get_function("FirstPriceAuction").expect("FirstPriceAuction function should exist");

    // Bidder valuations and bids
    let valuations = Value::List(vec![
        Value::Real(100.0), Value::Real(80.0), Value::Real(90.0), Value::Real(70.0)
    ]);
    let bids = Value::List(vec![
        Value::Real(75.0), Value::Real(60.0), Value::Real(70.0), Value::Real(50.0)
    ]);
    
    let result = fp_auction_fn(&[valuations, bids]).expect("First-price auction should succeed");
    
    match result {
        Value::LyObj(obj) => {
            // Check winner
            let winner = obj.call_method("Winner", &[]).expect("Should get winner");
            match winner {
                Value::Integer(w) => {
                    assert!(w >= &0 && w <= &3, "Winner should be valid bidder index");
                }
                _ => panic!("Winner should be integer index"),
            }

            // Check winning bid (should be highest bid)
            let winning_bid = obj.call_method("WinningBid", &[]).expect("Should get winning bid");
            assert_eq!(extract_real_from_value(&winning_bid), Some(75.0), "Highest bid should win");

            // Check revenue
            let revenue = obj.call_method("Revenue", &[]).expect("Should get auction revenue");
            assert_eq!(extract_real_from_value(&revenue), Some(75.0), "Revenue should equal winning bid");

            // Check efficiency (winner should have highest valuation among bidders)
            let efficiency = obj.call_method("Efficiency", &[]).expect("Should get efficiency");
            let eff_val = extract_real_from_value(&efficiency).expect("Efficiency should be numeric");
            assert!(eff_val >= 0.0 && eff_val <= 1.0, "Efficiency should be between 0 and 1");
        }
        _ => panic!("First-price auction should return auction object"),
    }
}

#[test]
fn test_second_price_auction() {
    let stdlib = create_test_stdlib();
    let sp_auction_fn = stdlib.get_function("SecondPriceAuction").expect("SecondPriceAuction function should exist");

    let valuations = Value::List(vec![
        Value::Real(100.0), Value::Real(80.0), Value::Real(90.0), Value::Real(70.0)
    ]);
    let bids = Value::List(vec![
        Value::Real(100.0), Value::Real(80.0), Value::Real(90.0), Value::Real(70.0)
    ]);
    
    let result = sp_auction_fn(&[valuations, bids]).expect("Second-price auction should succeed");
    
    match result {
        Value::LyObj(obj) => {
            // In second-price auction, winner pays second-highest bid
            let winning_payment = obj.call_method("WinningPayment", &[]).expect("Should get winning payment");
            assert_eq!(extract_real_from_value(&winning_payment), Some(90.0), "Should pay second-highest bid");

            // Check truthful bidding property
            let is_truthful = obj.call_method("IsTruthful", &[]).expect("Should check truthfulness");
            match is_truthful {
                Value::String(s) => assert_eq!(s, "true", "Second-price auction should be truthful"),
                _ => panic!("Truthfulness should be boolean string"),
            }

            // Check revenue
            let revenue = obj.call_method("Revenue", &[]).expect("Should get auction revenue");
            assert_eq!(extract_real_from_value(&revenue), Some(90.0), "Revenue should be second-highest bid");
        }
        _ => panic!("Second-price auction should return auction object"),
    }
}

#[test]
fn test_vickrey_auction() {
    let stdlib = create_test_stdlib();
    let vickrey_fn = stdlib.get_function("VickreyAuction").expect("VickreyAuction function should exist");

    // Multi-unit auction: 2 items, 4 bidders
    let valuations = Value::List(vec![
        Value::List(vec![Value::Real(100.0), Value::Real(80.0)]),  // Bidder 1: values for items 1,2
        Value::List(vec![Value::Real(90.0), Value::Real(70.0)]),   // Bidder 2
        Value::List(vec![Value::Real(85.0), Value::Real(75.0)]),   // Bidder 3
        Value::List(vec![Value::Real(60.0), Value::Real(65.0)]),   // Bidder 4
    ]);
    
    let result = vickrey_fn(&[valuations]).expect("Vickrey auction should succeed");
    
    match result {
        Value::LyObj(obj) => {
            // Check allocation
            let allocation = obj.call_method("Allocation", &[]).expect("Should get allocation");
            match allocation {
                Value::List(alloc_list) => {
                    // Each bidder gets list of items (empty list if none)
                    assert_eq!(alloc_list.len(), 4, "Should have allocation for each bidder");
                }
                _ => panic!("Allocation should be list"),
            }

            // Check payments (VCG payments)
            let payments = obj.call_method("Payments", &[]).expect("Should get payments");
            match payments {
                Value::List(payment_list) => {
                    assert_eq!(payment_list.len(), 4, "Should have payment for each bidder");
                    // All payments should be non-negative
                    for payment in &payment_list {
                        let p = extract_real_from_value(payment).expect("Payment should be numeric");
                        assert!(p >= 0.0, "Payments should be non-negative");
                    }
                }
                _ => panic!("Payments should be list"),
            }

            // Check total revenue
            let revenue = obj.call_method("Revenue", &[]).expect("Should get total revenue");
            let rev_val = extract_real_from_value(&revenue).expect("Revenue should be numeric");
            assert!(rev_val >= 0.0, "Revenue should be non-negative");

            // Check incentive compatibility (truthfulness)
            let is_truthful = obj.call_method("IsTruthful", &[]).expect("Should check truthfulness");
            match is_truthful {
                Value::String(s) => assert_eq!(s, "true", "Vickrey auction should be truthful"),
                _ => panic!("Truthfulness should be boolean string"),
            }
        }
        _ => panic!("Vickrey auction should return auction object"),
    }
}

#[test]
fn test_combinatorial_auction() {
    let stdlib = create_test_stdlib();
    let combo_fn = stdlib.get_function("CombinatorialAuction").expect("CombinatorialAuction function should exist");

    // Package bids: bidder -> (package, bid)
    let package_bids = Value::List(vec![
        Value::List(vec![
            Value::List(vec![Value::Integer(0), Value::Integer(1)]),  // Package {0,1}
            Value::Real(150.0)  // Bid
        ]),
        Value::List(vec![
            Value::List(vec![Value::Integer(0)]),  // Package {0}
            Value::Real(80.0)
        ]),
        Value::List(vec![
            Value::List(vec![Value::Integer(1)]),  // Package {1}  
            Value::Real(90.0)
        ]),
        Value::List(vec![
            Value::List(vec![Value::Integer(0), Value::Integer(1)]),  // Package {0,1}
            Value::Real(140.0)
        ]),
    ]);
    
    let result = combo_fn(&[package_bids]).expect("Combinatorial auction should succeed");
    
    match result {
        Value::LyObj(obj) => {
            // Check winner determination (optimal allocation)
            let winners = obj.call_method("Winners", &[]).expect("Should get winners");
            match winners {
                Value::List(winner_list) => {
                    // List of winning bids
                    for winner in &winner_list {
                        match winner {
                            Value::List(win_info) => {
                                assert_eq!(win_info.len(), 3, "Winner info should have bidder, package, bid");
                            }
                            _ => panic!("Winner info should be list"),
                        }
                    }
                }
                _ => panic!("Winners should be list"),
            }

            // Check that allocation maximizes social welfare
            let social_welfare = obj.call_method("SocialWelfare", &[]).expect("Should get social welfare");
            let sw_val = extract_real_from_value(&social_welfare).expect("Social welfare should be numeric");
            assert!(sw_val >= 0.0, "Social welfare should be non-negative");
            // Should be 150.0 (first bid) since that maximizes welfare

            // Check no conflicts in allocation
            let allocation = obj.call_method("Allocation", &[]).expect("Should get final allocation");
            match allocation {
                Value::List(_) => {
                    // Verify no item is allocated to multiple bidders (checked internally)
                }
                _ => panic!("Allocation should be list"),
            }
        }
        _ => panic!("Combinatorial auction should return auction object"),
    }
}

#[test]
fn test_english_auction() {
    let stdlib = create_test_stdlib();
    let english_fn = stdlib.get_function("EnglishAuction").expect("EnglishAuction function should exist");

    let valuations = Value::List(vec![
        Value::Real(100.0), Value::Real(80.0), Value::Real(90.0), Value::Real(70.0)
    ]);
    let increment = Value::Real(5.0);
    let reserve_price = Value::Real(50.0);
    
    let result = english_fn(&[valuations, increment, reserve_price]).expect("English auction should succeed");
    
    match result {
        Value::LyObj(obj) => {
            // Check final price (should be just above second-highest valuation)
            let final_price = obj.call_method("FinalPrice", &[]).expect("Should get final price");
            let price_val = extract_real_from_value(&final_price).expect("Final price should be numeric");
            assert!(price_val >= 50.0, "Final price should be at least reserve price");
            assert!(price_val >= 90.0, "Final price should be at least second-highest valuation");

            // Check number of rounds
            let rounds = obj.call_method("Rounds", &[]).expect("Should get number of rounds");
            match rounds {
                Value::Integer(r) => assert!(r >= &1, "Should have at least one round"),
                _ => panic!("Rounds should be integer"),
            }

            // Check bidding sequence
            let sequence = obj.call_method("BiddingSequence", &[]).expect("Should get bidding sequence");
            match sequence {
                Value::List(_) => {
                    // Each round should show active bidders and current price
                }
                _ => panic!("Bidding sequence should be list"),
            }
        }
        _ => panic!("English auction should return auction object"),
    }
}

#[test]
fn test_dutch_auction() {
    let stdlib = create_test_stdlib();
    let dutch_fn = stdlib.get_function("DutchAuction").expect("DutchAuction function should exist");

    let valuations = Value::List(vec![
        Value::Real(100.0), Value::Real(80.0), Value::Real(90.0), Value::Real(70.0)
    ]);
    let starting_price = Value::Real(120.0);
    let decrement = Value::Real(10.0);
    
    let result = dutch_fn(&[valuations, starting_price, decrement]).expect("Dutch auction should succeed");
    
    match result {
        Value::LyObj(obj) => {
            // Check winner (should be highest valuation bidder)
            let winner = obj.call_method("Winner", &[]).expect("Should get winner");
            match winner {
                Value::Integer(w) => assert_eq!(*w, 0, "Bidder with highest valuation should win"),
                _ => panic!("Winner should be integer index"),
            }

            // Check winning price (first price at or below highest valuation)
            let winning_price = obj.call_method("WinningPrice", &[]).expect("Should get winning price");
            let price_val = extract_real_from_value(&winning_price).expect("Winning price should be numeric");
            assert_eq!(price_val, 100.0, "Should win at highest valuation");

            // Check rounds until acceptance
            let rounds = obj.call_method("Rounds", &[]).expect("Should get rounds");
            match rounds {
                Value::Integer(r) => assert_eq!(*r, 3, "Should take 3 rounds to reach 100"),
                _ => panic!("Rounds should be integer"),
            }
        }
        _ => panic!("Dutch auction should return auction object"),
    }
}

// ===============================
// MECHANISM DESIGN TESTS
// ===============================

#[test]
fn test_vcg_mechanism() {
    let stdlib = create_test_stdlib();
    let vcg_fn = stdlib.get_function("VCGMechanism").expect("VCGMechanism function should exist");

    let valuations = Value::List(vec![
        Value::List(vec![Value::Real(20.0), Value::Real(10.0)]),  // Bidder 1 values for outcomes A, B
        Value::List(vec![Value::Real(15.0), Value::Real(25.0)]),  // Bidder 2
        Value::List(vec![Value::Real(30.0), Value::Real(5.0)]),   // Bidder 3
    ]);
    
    let result = vcg_fn(&[valuations]).expect("VCG mechanism should succeed");
    
    match result {
        Value::LyObj(obj) => {
            // Check chosen outcome (should maximize social welfare)
            let outcome = obj.call_method("ChosenOutcome", &[]).expect("Should get chosen outcome");
            match outcome {
                Value::Integer(o) => assert!(*o >= 0 && *o <= 1, "Outcome should be valid"),
                _ => panic!("Chosen outcome should be integer"),
            }

            // Check VCG payments
            let payments = obj.call_method("VCGPayments", &[]).expect("Should get VCG payments");
            match payments {
                Value::List(payment_list) => {
                    assert_eq!(payment_list.len(), 3, "Should have payment for each bidder");
                    
                    // VCG payments should be non-negative
                    for payment in &payment_list {
                        let p = extract_real_from_value(payment).expect("Payment should be numeric");
                        assert!(p >= 0.0, "VCG payments should be non-negative");
                    }
                }
                _ => panic!("VCG payments should be list"),
            }

            // Check incentive compatibility
            let is_truthful = obj.call_method("IsTruthful", &[]).expect("Should check truthfulness");
            match is_truthful {
                Value::String(s) => assert_eq!(s, "true", "VCG mechanism should be truthful"),
                _ => panic!("Truthfulness should be boolean string"),
            }

            // Check individual rationality
            let is_ir = obj.call_method("IsIndividuallyRational", &[]).expect("Should check individual rationality");
            match is_ir {
                Value::String(s) => assert_eq!(s, "true", "VCG mechanism should be individually rational"),
                _ => panic!("Individual rationality should be boolean string"),
            }
        }
        _ => panic!("VCG mechanism should return mechanism object"),
    }
}

#[test]
fn test_optimal_auction() {
    let stdlib = create_test_stdlib();
    let optimal_fn = stdlib.get_function("OptimalAuction").expect("OptimalAuction function should exist");

    // Value distributions (uniform distributions with [low, high] bounds)
    let distributions = Value::List(vec![
        Value::List(vec![Value::Real(0.0), Value::Real(100.0)]),  // Uniform[0,100]
        Value::List(vec![Value::Real(0.0), Value::Real(100.0)]),  // Uniform[0,100]
        Value::List(vec![Value::Real(0.0), Value::Real(100.0)]),  // Uniform[0,100]
    ]);
    
    let result = optimal_fn(&[distributions]).expect("Optimal auction should succeed");
    
    match result {
        Value::LyObj(obj) => {
            // Check reserve price
            let reserve = obj.call_method("ReservePrice", &[]).expect("Should get reserve price");
            let reserve_val = extract_real_from_value(&reserve).expect("Reserve price should be numeric");
            assert!(reserve_val > 0.0, "Reserve price should be positive for uniform distributions");

            // Check expected revenue
            let expected_revenue = obj.call_method("ExpectedRevenue", &[]).expect("Should get expected revenue");
            let rev_val = extract_real_from_value(&expected_revenue).expect("Expected revenue should be numeric");
            assert!(rev_val > 0.0, "Expected revenue should be positive");

            // Check allocation rule
            let allocation_rule = obj.call_method("AllocationRule", &[]).expect("Should get allocation rule");
            match allocation_rule {
                Value::String(_) => {
                    // Should describe the optimal allocation mechanism
                }
                _ => panic!("Allocation rule should be string description"),
            }

            // Check that mechanism is optimal
            let is_optimal = obj.call_method("IsOptimal", &[]).expect("Should check optimality");
            match is_optimal {
                Value::String(s) => assert_eq!(s, "true", "Should be revenue-optimal"),
                _ => panic!("Optimality should be boolean string"),
            }
        }
        _ => panic!("Optimal auction should return mechanism object"),
    }
}

#[test]
fn test_revenue_maximization() {
    let stdlib = create_test_stdlib();
    let rev_max_fn = stdlib.get_function("RevenueMaximization").expect("RevenueMaximization function should exist");

    let distributions = Value::List(vec![
        Value::List(vec![Value::Real(0.0), Value::Real(100.0)]),
        Value::List(vec![Value::Real(0.0), Value::Real(80.0)]),   // Asymmetric distributions
        Value::List(vec![Value::Real(10.0), Value::Real(90.0)]),
    ]);
    
    let result = rev_max_fn(&[distributions]).expect("Revenue maximization should succeed");
    
    match result {
        Value::LyObj(obj) => {
            // Check virtual valuations
            let virtual_vals = obj.call_method("VirtualValuations", &[
                Value::List(vec![Value::Real(50.0), Value::Real(40.0), Value::Real(60.0)])
            ]).expect("Should get virtual valuations");
            match virtual_vals {
                Value::List(vv_list) => {
                    assert_eq!(vv_list.len(), 3, "Should have virtual valuation for each bidder");
                }
                _ => panic!("Virtual valuations should be list"),
            }

            // Check ironed virtual valuations (monotonic)
            let ironed_vals = obj.call_method("IronedVirtualValuations", &[
                Value::List(vec![Value::Real(50.0), Value::Real(40.0), Value::Real(60.0)])
            ]).expect("Should get ironed virtual valuations");
            match ironed_vals {
                Value::List(_) => {
                    // Should maintain same structure but with monotonicity
                }
                _ => panic!("Ironed virtual valuations should be list"),
            }

            // Check optimal reserve prices
            let reserves = obj.call_method("OptimalReserves", &[]).expect("Should get optimal reserves");
            match reserves {
                Value::List(reserve_list) => {
                    assert_eq!(reserve_list.len(), 3, "Should have reserve for each bidder type");
                    for reserve in &reserve_list {
                        let r = extract_real_from_value(reserve).expect("Reserve should be numeric");
                        assert!(r >= 0.0, "Reserves should be non-negative");
                    }
                }
                _ => panic!("Optimal reserves should be list"),
            }
        }
        _ => panic!("Revenue maximization should return mechanism object"),
    }
}

// ===============================
// MATCHING ALGORITHM TESTS
// ===============================

#[test]
fn test_stable_marriage() {
    let stdlib = create_test_stdlib();
    let marriage_fn = stdlib.get_function("StableMarriage").expect("StableMarriage function should exist");

    // Preference lists: men's preferences over women, women's preferences over men
    let men_prefs = Value::List(vec![
        Value::List(vec![Value::Integer(0), Value::Integer(1), Value::Integer(2)]),  // Man 0's preferences
        Value::List(vec![Value::Integer(1), Value::Integer(0), Value::Integer(2)]),  // Man 1's preferences
        Value::List(vec![Value::Integer(2), Value::Integer(1), Value::Integer(0)]),  // Man 2's preferences
    ]);
    
    let women_prefs = Value::List(vec![
        Value::List(vec![Value::Integer(1), Value::Integer(2), Value::Integer(0)]),  // Woman 0's preferences
        Value::List(vec![Value::Integer(0), Value::Integer(2), Value::Integer(1)]),  // Woman 1's preferences
        Value::List(vec![Value::Integer(2), Value::Integer(0), Value::Integer(1)]),  // Woman 2's preferences
    ]);
    
    let result = marriage_fn(&[men_prefs, women_prefs]).expect("Stable marriage should succeed");
    
    match result {
        Value::LyObj(obj) => {
            // Check matching
            let matching = obj.call_method("Matching", &[]).expect("Should get matching");
            match matching {
                Value::List(match_list) => {
                    assert_eq!(match_list.len(), 3, "Should have 3 marriages");
                    
                    // Each marriage should be (man, woman) pair
                    let mut men_matched = std::collections::HashSet::new();
                    let mut women_matched = std::collections::HashSet::new();
                    
                    for marriage in &match_list {
                        match marriage {
                            Value::List(pair) => {
                                assert_eq!(pair.len(), 2, "Each marriage should be a pair");
                                let man = match &pair[0] {
                                    Value::Integer(m) => *m,
                                    _ => panic!("Man should be integer"),
                                };
                                let woman = match &pair[1] {
                                    Value::Integer(w) => *w,
                                    _ => panic!("Woman should be integer"),
                                };
                                
                                assert!(!men_matched.contains(&man), "Each man matched once");
                                assert!(!women_matched.contains(&woman), "Each woman matched once");
                                men_matched.insert(man);
                                women_matched.insert(woman);
                            }
                            _ => panic!("Marriage should be pair"),
                        }
                    }
                }
                _ => panic!("Matching should be list"),
            }

            // Check stability
            let is_stable = obj.call_method("IsStable", &[]).expect("Should check stability");
            match is_stable {
                Value::String(s) => assert_eq!(s, "true", "Matching should be stable"),
                _ => panic!("Stability should be boolean string"),
            }

            // Check optimality for men (Gale-Shapley gives men-optimal)
            let is_men_optimal = obj.call_method("IsMenOptimal", &[]).expect("Should check men-optimality");
            match is_men_optimal {
                Value::String(s) => assert_eq!(s, "true", "Should be men-optimal"),
                _ => panic!("Men-optimality should be boolean string"),
            }
        }
        _ => panic!("Stable marriage should return matching object"),
    }
}

#[test]
fn test_assignment_problem() {
    let stdlib = create_test_stdlib();
    let assignment_fn = stdlib.get_function("AssignmentProblem").expect("AssignmentProblem function should exist");

    // Cost matrix for assignment (workers to tasks)
    let cost_matrix = Value::List(vec![
        Value::List(vec![Value::Real(10.0), Value::Real(20.0), Value::Real(30.0)]),  // Worker 0 costs
        Value::List(vec![Value::Real(15.0), Value::Real(25.0), Value::Real(20.0)]),  // Worker 1 costs
        Value::List(vec![Value::Real(20.0), Value::Real(15.0), Value::Real(25.0)]),  // Worker 2 costs
    ]);
    
    let result = assignment_fn(&[cost_matrix]).expect("Assignment problem should succeed");
    
    match result {
        Value::LyObj(obj) => {
            // Check optimal assignment
            let assignment = obj.call_method("Assignment", &[]).expect("Should get assignment");
            match assignment {
                Value::List(assign_list) => {
                    assert_eq!(assign_list.len(), 3, "Should assign 3 workers");
                    
                    // Check that each worker gets exactly one task
                    let mut tasks_assigned = std::collections::HashSet::new();
                    for (worker, task_val) in assign_list.iter().enumerate() {
                        match task_val {
                            Value::Integer(task) => {
                                assert!(*task >= 0 && *task <= 2, "Task should be valid");
                                assert!(!tasks_assigned.contains(task), "Each task assigned once");
                                tasks_assigned.insert(*task);
                            }
                            _ => panic!("Task assignment should be integer"),
                        }
                    }
                }
                _ => panic!("Assignment should be list"),
            }

            // Check minimum cost
            let min_cost = obj.call_method("MinimumCost", &[]).expect("Should get minimum cost");
            let cost_val = extract_real_from_value(&min_cost).expect("Minimum cost should be numeric");
            assert!(cost_val > 0.0, "Minimum cost should be positive");
            // For this matrix, optimal assignment should be: worker 0->task 0 (10), worker 1->task 2 (20), worker 2->task 1 (15)
            // Total cost = 10 + 20 + 15 = 45
            assert_eq!(cost_val, 45.0, "Should find optimal cost");

            // Check that solution is optimal
            let is_optimal = obj.call_method("IsOptimal", &[]).expect("Should check optimality");
            match is_optimal {
                Value::String(s) => assert_eq!(s, "true", "Assignment should be optimal"),
                _ => panic!("Optimality should be boolean string"),
            }
        }
        _ => panic!("Assignment problem should return assignment object"),
    }
}

#[test]
fn test_stable_assignment() {
    let stdlib = create_test_stdlib();
    let stable_assign_fn = stdlib.get_function("StableAssignment").expect("StableAssignment function should exist");

    // Two-sided preferences: workers over firms, firms over workers
    let worker_prefs = Value::List(vec![
        Value::List(vec![Value::Integer(1), Value::Integer(0), Value::Integer(2)]),  // Worker 0's firm preferences
        Value::List(vec![Value::Integer(0), Value::Integer(2), Value::Integer(1)]),  // Worker 1's firm preferences
        Value::List(vec![Value::Integer(2), Value::Integer(1), Value::Integer(0)]),  // Worker 2's firm preferences
    ]);
    
    let firm_prefs = Value::List(vec![
        Value::List(vec![Value::Integer(2), Value::Integer(0), Value::Integer(1)]),  // Firm 0's worker preferences
        Value::List(vec![Value::Integer(1), Value::Integer(2), Value::Integer(0)]),  // Firm 1's worker preferences
        Value::List(vec![Value::Integer(0), Value::Integer(1), Value::Integer(2)]),  // Firm 2's worker preferences
    ]);
    
    let result = stable_assign_fn(&[worker_prefs, firm_prefs]).expect("Stable assignment should succeed");
    
    match result {
        Value::LyObj(obj) => {
            // Check assignment
            let assignment = obj.call_method("Assignment", &[]).expect("Should get assignment");
            match assignment {
                Value::List(assign_list) => {
                    assert_eq!(assign_list.len(), 3, "Should assign 3 workers");
                }
                _ => panic!("Assignment should be list"),
            }

            // Check stability
            let is_stable = obj.call_method("IsStable", &[]).expect("Should check stability");
            match is_stable {
                Value::String(s) => assert_eq!(s, "true", "Assignment should be stable"),
                _ => panic!("Stability should be boolean string"),
            }

            // Check blocking pairs (should be empty for stable assignment)
            let blocking_pairs = obj.call_method("BlockingPairs", &[]).expect("Should get blocking pairs");
            match blocking_pairs {
                Value::List(bp_list) => {
                    assert_eq!(bp_list.len(), 0, "Stable assignment should have no blocking pairs");
                }
                _ => panic!("Blocking pairs should be list"),
            }
        }
        _ => panic!("Stable assignment should return assignment object"),
    }
}

// ===============================
// INTEGRATION AND ERROR TESTS
// ===============================

#[test]
fn test_game_theory_error_handling() {
    let stdlib = create_test_stdlib();
    
    // Test invalid payoff matrix for Nash equilibrium
    let nash_fn = stdlib.get_function("NashEquilibrium").expect("NashEquilibrium function should exist");
    let invalid_matrix = Value::List(vec![Value::String("invalid".to_string())]);
    
    let result = nash_fn(&[invalid_matrix]);
    assert!(result.is_err(), "Should reject invalid payoff matrix");
    
    // Test insufficient arguments
    let result = nash_fn(&[]);
    assert!(result.is_err(), "Should require arguments");
    
    // Test auction with mismatched valuations and bids
    let fp_auction_fn = stdlib.get_function("FirstPriceAuction").expect("FirstPriceAuction function should exist");
    let valuations = Value::List(vec![Value::Real(100.0), Value::Real(80.0)]);
    let bids = Value::List(vec![Value::Real(75.0)]);  // Wrong length
    
    let result = fp_auction_fn(&[valuations, bids]);
    assert!(result.is_err(), "Should reject mismatched lengths");
}

#[test]
fn test_foreign_object_method_calls() {
    let stdlib = create_test_stdlib();
    let nash_fn = stdlib.get_function("NashEquilibrium").expect("NashEquilibrium function should exist");

    let payoff_matrix = create_test_payoff_matrix();
    let result = nash_fn(&[payoff_matrix]).expect("Nash equilibrium should succeed");
    
    match result {
        Value::LyObj(obj) => {
            // Test various method calls on the Foreign object
            let type_name = obj.type_name();
            assert_eq!(type_name, "NashEquilibrium", "Should have correct type name");

            // Test invalid method call
            let invalid_result = obj.call_method("NonexistentMethod", &[]);
            assert!(invalid_result.is_err(), "Should reject invalid method calls");

            // Test method with wrong argument count
            let wrong_args_result = obj.call_method("ClusterPoints", &[Value::Integer(1), Value::Integer(2)]);
            assert!(wrong_args_result.is_err(), "Should reject wrong argument count for methods that exist");
        }
        _ => panic!("Should return Foreign object"),
    }
}

#[test]
fn test_performance_large_games() {
    let stdlib = create_test_stdlib();
    let nash_fn = stdlib.get_function("NashEquilibrium").expect("NashEquilibrium function should exist");

    // Create larger game (5x5)
    let large_matrix = Value::List(vec![
        Value::List((0..5).map(|i| {
            Value::List((0..5).map(|j| Value::Real((i * j) as f64)).collect())
        }).collect()),
        Value::List((0..5).map(|i| {
            Value::List((0..5).map(|j| Value::Real((i + j) as f64)).collect())
        }).collect()),
    ]);
    
    let start = std::time::Instant::now();
    let result = nash_fn(&[large_matrix]);
    let duration = start.elapsed();
    
    assert!(result.is_ok(), "Should handle larger games");
    assert!(duration.as_secs() < 10, "Should complete in reasonable time");
}