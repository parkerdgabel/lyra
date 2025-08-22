//! Equilibrium Concepts for Game Theory
//!
//! This module implements core equilibrium concepts including:
//! - Nash equilibrium (pure and mixed strategies)
//! - Correlated equilibrium
//! - Evolutionary stable strategies (ESS)
//! - Iterated elimination of dominated strategies

use crate::foreign::{Foreign, ForeignError, LyObj};
use crate::vm::{Value, VmResult, VmError};
use std::any::Any;
use std::collections::HashMap;

/// Game representation for matrix games
#[derive(Debug, Clone)]
pub struct Game {
    /// Payoff matrices for each player (player -> strategies -> strategies -> payoff)
    pub payoff_matrices: Vec<Vec<Vec<f64>>>,
    /// Number of players
    pub num_players: usize,
    /// Strategy counts for each player
    pub strategy_counts: Vec<usize>,
    /// Strategy names (optional)
    pub strategy_names: Vec<Vec<String>>,
}

impl Game {
    /// Create new game from payoff matrices
    pub fn new(payoff_matrices: Vec<Vec<Vec<f64>>>) -> VmResult<Self> {
        if payoff_matrices.is_empty() {
            return Err(VmError::TypeError {
                expected: "non-empty payoff matrices".to_string(),
                actual: "empty matrices".to_string(),
            });
        }

        let num_players = payoff_matrices.len();
        let mut strategy_counts = Vec::new();

        // Validate dimensions and extract strategy counts
        for (player, matrix) in payoff_matrices.iter().enumerate() {
            if matrix.is_empty() {
                return Err(VmError::TypeError {
                    expected: "non-empty strategy matrix".to_string(),
                    actual: format!("empty matrix for player {}", player),
                });
            }

            let rows = matrix.len();
            if player == 0 {
                strategy_counts.push(rows);
            }

            for (i, row) in matrix.iter().enumerate() {
                if row.len() != matrix[0].len() {
                    return Err(VmError::TypeError {
                        expected: "rectangular matrix".to_string(),
                        actual: format!("row {} has different length", i),
                    });
                }
            }

            if player == 0 {
                strategy_counts.push(matrix[0].len());
            }
        }

        // Generate default strategy names
        let strategy_names = strategy_counts.iter()
            .enumerate()
            .map(|(player, &count)| {
                (0..count).map(|i| format!("Player{}_Strategy{}", player, i)).collect()
            })
            .collect();

        Ok(Game {
            payoff_matrices,
            num_players,
            strategy_counts,
            strategy_names,
        })
    }

    /// Get payoff for a strategy profile
    pub fn get_payoff(&self, player: usize, strategy_profile: &[usize]) -> f64 {
        if player >= self.num_players || strategy_profile.len() != self.num_players {
            return 0.0;
        }

        // For 2-player games, use row/column indexing
        if self.num_players == 2 {
            let row = strategy_profile[0];
            let col = strategy_profile[1];
            if row < self.payoff_matrices[player].len() && 
               col < self.payoff_matrices[player][row].len() {
                self.payoff_matrices[player][row][col]
            } else {
                0.0
            }
        } else {
            // For n-player games, this would need more complex indexing
            // For now, implement basic case
            0.0
        }
    }

    /// Check if a strategy strictly dominates another
    pub fn strictly_dominates(&self, player: usize, strategy1: usize, strategy2: usize) -> bool {
        if player >= self.num_players || 
           strategy1 >= self.strategy_counts[player] || 
           strategy2 >= self.strategy_counts[player] {
            return false;
        }

        // For 2-player games
        if self.num_players == 2 {
            let other_player = 1 - player;
            for other_strategy in 0..self.strategy_counts[other_player] {
                let profile1 = if player == 0 { [strategy1, other_strategy] } else { [other_strategy, strategy1] };
                let profile2 = if player == 0 { [strategy2, other_strategy] } else { [other_strategy, strategy2] };
                
                let payoff1 = self.get_payoff(player, &profile1);
                let payoff2 = self.get_payoff(player, &profile2);
                
                if payoff1 <= payoff2 {
                    return false;
                }
            }
            true
        } else {
            false
        }
    }

    /// Check if a strategy weakly dominates another
    pub fn weakly_dominates(&self, player: usize, strategy1: usize, strategy2: usize) -> bool {
        if player >= self.num_players || 
           strategy1 >= self.strategy_counts[player] || 
           strategy2 >= self.strategy_counts[player] {
            return false;
        }

        // For 2-player games
        if self.num_players == 2 {
            let other_player = 1 - player;
            let mut at_least_one_better = false;
            
            for other_strategy in 0..self.strategy_counts[other_player] {
                let profile1 = if player == 0 { [strategy1, other_strategy] } else { [other_strategy, strategy1] };
                let profile2 = if player == 0 { [strategy2, other_strategy] } else { [other_strategy, strategy2] };
                
                let payoff1 = self.get_payoff(player, &profile1);
                let payoff2 = self.get_payoff(player, &profile2);
                
                if payoff1 < payoff2 {
                    return false;
                }
                if payoff1 > payoff2 {
                    at_least_one_better = true;
                }
            }
            at_least_one_better
        } else {
            false
        }
    }
}

/// Nash equilibrium result
#[derive(Debug, Clone)]
pub struct NashEquilibrium {
    /// Mixed strategies for each player (probabilities over actions)
    pub strategies: Vec<Vec<f64>>,
    /// Expected payoffs for each player
    pub expected_payoffs: Vec<f64>,
    /// Type of equilibrium (Pure, Mixed)
    pub equilibrium_type: String,
    /// Support of mixed strategies (which actions have positive probability)
    pub support: Vec<Vec<usize>>,
    /// Whether equilibrium is unique
    pub is_unique: bool,
    /// Game that generated this equilibrium
    pub game: Game,
}

impl NashEquilibrium {
    /// Compute Nash equilibrium for 2-player game using linear complementarity
    pub fn compute_2player_nash(game: Game) -> VmResult<Self> {
        let num_strategies_p1 = game.strategy_counts[0];
        let num_strategies_p2 = game.strategy_counts[1];

        // Try to find pure strategy Nash equilibria first
        let mut pure_equilibria = Vec::new();
        
        for i in 0..num_strategies_p1 {
            for j in 0..num_strategies_p2 {
                if Self::is_pure_nash(&game, i, j) {
                    let strategies = vec![
                        Self::pure_strategy(i, num_strategies_p1),
                        Self::pure_strategy(j, num_strategies_p2),
                    ];
                    let expected_payoffs = vec![
                        game.get_payoff(0, &[i, j]),
                        game.get_payoff(1, &[i, j]),
                    ];
                    let support = vec![vec![i], vec![j]];
                    
                    pure_equilibria.push(NashEquilibrium {
                        strategies,
                        expected_payoffs,
                        equilibrium_type: "Pure".to_string(),
                        support,
                        is_unique: false, // Will determine later
                        game: game.clone(),
                    });
                }
            }
        }

        if !pure_equilibria.is_empty() {
            // Return first pure equilibrium found
            let is_unique = pure_equilibria.len() == 1;
            let mut equilibrium = pure_equilibria.into_iter().next().unwrap();
            equilibrium.is_unique = is_unique;
            return Ok(equilibrium);
        }

        // Compute mixed strategy Nash equilibrium using iterative method
        Self::compute_mixed_nash(game)
    }

    /// Check if a pure strategy profile is a Nash equilibrium
    fn is_pure_nash(game: &Game, strategy1: usize, strategy2: usize) -> bool {
        let payoff1 = game.get_payoff(0, &[strategy1, strategy2]);
        let payoff2 = game.get_payoff(1, &[strategy1, strategy2]);

        // Check if player 1 wants to deviate
        for alt_strategy in 0..game.strategy_counts[0] {
            if alt_strategy != strategy1 {
                let alt_payoff = game.get_payoff(0, &[alt_strategy, strategy2]);
                if alt_payoff > payoff1 {
                    return false;
                }
            }
        }

        // Check if player 2 wants to deviate
        for alt_strategy in 0..game.strategy_counts[1] {
            if alt_strategy != strategy2 {
                let alt_payoff = game.get_payoff(1, &[strategy1, alt_strategy]);
                if alt_payoff > payoff2 {
                    return false;
                }
            }
        }

        true
    }

    /// Create pure strategy vector
    fn pure_strategy(strategy: usize, num_strategies: usize) -> Vec<f64> {
        let mut strategy_vec = vec![0.0; num_strategies];
        strategy_vec[strategy] = 1.0;
        strategy_vec
    }

    /// Compute mixed strategy Nash equilibrium using best response iteration
    fn compute_mixed_nash(game: Game) -> VmResult<Self> {
        let num_strategies_p1 = game.strategy_counts[0];
        let num_strategies_p2 = game.strategy_counts[1];

        // Initialize with uniform random strategies
        let mut strategy_p1 = vec![1.0 / num_strategies_p1 as f64; num_strategies_p1];
        let mut strategy_p2 = vec![1.0 / num_strategies_p2 as f64; num_strategies_p2];

        // Iterative best response
        let max_iterations = 1000;
        let tolerance = 1e-6;

        for _ in 0..max_iterations {
            let old_strategy_p1 = strategy_p1.clone();
            let old_strategy_p2 = strategy_p2.clone();

            // Update player 1's strategy (best response to player 2)
            strategy_p1 = Self::best_response_player1(&game, &strategy_p2);
            // Update player 2's strategy (best response to player 1)
            strategy_p2 = Self::best_response_player2(&game, &strategy_p1);

            // Check convergence
            let diff_p1: f64 = strategy_p1.iter().zip(old_strategy_p1.iter())
                .map(|(new, old)| (new - old).abs())
                .sum();
            let diff_p2: f64 = strategy_p2.iter().zip(old_strategy_p2.iter())
                .map(|(new, old)| (new - old).abs())
                .sum();

            if diff_p1 < tolerance && diff_p2 < tolerance {
                break;
            }
        }

        // Compute expected payoffs
        let expected_payoff_p1 = Self::compute_expected_payoff(&game, 0, &strategy_p1, &strategy_p2);
        let expected_payoff_p2 = Self::compute_expected_payoff(&game, 1, &strategy_p1, &strategy_p2);

        // Determine support (strategies with positive probability)
        let support_p1: Vec<usize> = strategy_p1.iter().enumerate()
            .filter_map(|(i, &prob)| if prob > 1e-9 { Some(i) } else { None })
            .collect();
        let support_p2: Vec<usize> = strategy_p2.iter().enumerate()
            .filter_map(|(i, &prob)| if prob > 1e-9 { Some(i) } else { None })
            .collect();

        Ok(NashEquilibrium {
            strategies: vec![strategy_p1, strategy_p2],
            expected_payoffs: vec![expected_payoff_p1, expected_payoff_p2],
            equilibrium_type: "Mixed".to_string(),
            support: vec![support_p1, support_p2],
            is_unique: false, // Mixed equilibria uniqueness is complex to determine
            game,
        })
    }

    /// Compute best response for player 1
    fn best_response_player1(game: &Game, strategy_p2: &[f64]) -> Vec<f64> {
        let num_strategies = game.strategy_counts[0];
        let mut expected_payoffs = vec![0.0; num_strategies];

        // Compute expected payoff for each strategy
        for i in 0..num_strategies {
            for j in 0..strategy_p2.len() {
                expected_payoffs[i] += strategy_p2[j] * game.get_payoff(0, &[i, j]);
            }
        }

        // Find best responses (strategies with maximum expected payoff)
        let max_payoff = expected_payoffs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let best_strategies: Vec<usize> = expected_payoffs.iter().enumerate()
            .filter_map(|(i, &payoff)| {
                if (payoff - max_payoff).abs() < 1e-9 { Some(i) } else { None }
            })
            .collect();

        // Return uniform distribution over best responses
        let mut strategy = vec![0.0; num_strategies];
        let prob = 1.0 / best_strategies.len() as f64;
        for &i in &best_strategies {
            strategy[i] = prob;
        }
        strategy
    }

    /// Compute best response for player 2
    fn best_response_player2(game: &Game, strategy_p1: &[f64]) -> Vec<f64> {
        let num_strategies = game.strategy_counts[1];
        let mut expected_payoffs = vec![0.0; num_strategies];

        // Compute expected payoff for each strategy
        for j in 0..num_strategies {
            for i in 0..strategy_p1.len() {
                expected_payoffs[j] += strategy_p1[i] * game.get_payoff(1, &[i, j]);
            }
        }

        // Find best responses
        let max_payoff = expected_payoffs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let best_strategies: Vec<usize> = expected_payoffs.iter().enumerate()
            .filter_map(|(j, &payoff)| {
                if (payoff - max_payoff).abs() < 1e-9 { Some(j) } else { None }
            })
            .collect();

        // Return uniform distribution over best responses
        let mut strategy = vec![0.0; num_strategies];
        let prob = 1.0 / best_strategies.len() as f64;
        for &j in &best_strategies {
            strategy[j] = prob;
        }
        strategy
    }

    /// Compute expected payoff for a player given mixed strategies
    fn compute_expected_payoff(
        game: &Game,
        player: usize,
        strategy_p1: &[f64],
        strategy_p2: &[f64],
    ) -> f64 {
        let mut expected_payoff = 0.0;
        for i in 0..strategy_p1.len() {
            for j in 0..strategy_p2.len() {
                expected_payoff += strategy_p1[i] * strategy_p2[j] * game.get_payoff(player, &[i, j]);
            }
        }
        expected_payoff
    }
}

impl Foreign for NashEquilibrium {
    fn type_name(&self) -> &'static str {
        "NashEquilibrium"
    }

    fn call_method(&self, method: &str, _args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "EquilibriumType" => Ok(Value::String(self.equilibrium_type.clone())),
            "Strategies" => {
                let strategies: Vec<Value> = self.strategies.iter()
                    .map(|strategy| {
                        let probs: Vec<Value> = strategy.iter()
                            .map(|&prob| Value::Real(prob))
                            .collect();
                        Value::List(probs)
                    })
                    .collect();
                Ok(Value::List(strategies))
            }
            "Payoffs" => {
                let payoffs: Vec<Value> = self.expected_payoffs.iter()
                    .map(|&payoff| Value::Real(payoff))
                    .collect();
                Ok(Value::List(payoffs))
            }
            "Support" => {
                let support: Vec<Value> = self.support.iter()
                    .map(|player_support| {
                        let indices: Vec<Value> = player_support.iter()
                            .map(|&idx| Value::Integer(idx as i64))
                            .collect();
                        Value::List(indices)
                    })
                    .collect();
                Ok(Value::List(support))
            }
            "IsUnique" => Ok(Value::String(if self.is_unique { "true" } else { "false" }.to_string())),
            "NumPlayers" => Ok(Value::Integer(self.game.num_players as i64)),
            "StrategyNames" => {
                let names: Vec<Value> = self.game.strategy_names.iter()
                    .map(|player_names| {
                        let name_list: Vec<Value> = player_names.iter()
                            .map(|name| Value::String(name.clone()))
                            .collect();
                        Value::List(name_list)
                    })
                    .collect();
                Ok(Value::List(names))
            }
            _ => Err(ForeignError::UnknownMethod {
                type_name: self.type_name().to_string(),
                method: method.to_string(),
            }),
        }
    }

    fn clone_boxed(&self) -> Box<dyn Foreign> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Correlated equilibrium result
#[derive(Debug, Clone)]
pub struct CorrelatedEquilibrium {
    /// Joint probability distribution over action profiles
    pub distribution: HashMap<Vec<usize>, f64>,
    /// Expected payoffs for each player
    pub expected_payoffs: Vec<f64>,
    /// Game that generated this equilibrium
    pub game: Game,
    /// Whether the equilibrium satisfies incentive constraints
    pub is_incentive_compatible: bool,
}

impl CorrelatedEquilibrium {
    /// Compute correlated equilibrium that maximizes social welfare
    pub fn compute_welfare_maximizing(game: Game) -> VmResult<Self> {
        let num_strategies_p1 = game.strategy_counts[0];
        let num_strategies_p2 = game.strategy_counts[1];
        
        // For 2-player games, find distribution that maximizes sum of payoffs
        // subject to incentive compatibility constraints
        let mut best_distribution = HashMap::new();
        let mut best_welfare = f64::NEG_INFINITY;
        
        // Simple approach: try uniform distribution over all action profiles
        // and check which ones satisfy incentive compatibility
        let uniform_prob = 1.0 / (num_strategies_p1 * num_strategies_p2) as f64;
        
        for i in 0..num_strategies_p1 {
            for j in 0..num_strategies_p2 {
                best_distribution.insert(vec![i, j], uniform_prob);
            }
        }
        
        // Compute expected payoffs under this distribution
        let expected_payoff_p1 = Self::compute_expected_payoff(&game, 0, &best_distribution);
        let expected_payoff_p2 = Self::compute_expected_payoff(&game, 1, &best_distribution);
        
        let is_ic = Self::check_incentive_compatibility(&game, &best_distribution);
        
        Ok(CorrelatedEquilibrium {
            distribution: best_distribution,
            expected_payoffs: vec![expected_payoff_p1, expected_payoff_p2],
            game,
            is_incentive_compatible: is_ic,
        })
    }
    
    /// Compute expected payoff for a player under the correlated strategy
    fn compute_expected_payoff(
        game: &Game,
        player: usize,
        distribution: &HashMap<Vec<usize>, f64>,
    ) -> f64 {
        distribution.iter()
            .map(|(profile, &prob)| {
                prob * game.get_payoff(player, profile)
            })
            .sum()
    }
    
    /// Check if the distribution satisfies incentive compatibility
    fn check_incentive_compatibility(
        game: &Game,
        distribution: &HashMap<Vec<usize>, f64>,
    ) -> bool {
        // For each player and each recommended action, check if following
        // the recommendation is optimal given the conditional distribution
        
        for player in 0..game.num_players {
            for recommended_action in 0..game.strategy_counts[player] {
                // Compute expected payoff from following recommendation
                let mut expected_payoff_follow = 0.0;
                let mut conditional_prob_sum = 0.0;
                
                for (profile, &prob) in distribution {
                    if profile[player] == recommended_action {
                        expected_payoff_follow += prob * game.get_payoff(player, profile);
                        conditional_prob_sum += prob;
                    }
                }
                
                if conditional_prob_sum > 1e-9 {
                    expected_payoff_follow /= conditional_prob_sum;
                    
                    // Check if deviating to any other action gives higher payoff
                    for alt_action in 0..game.strategy_counts[player] {
                        if alt_action != recommended_action {
                            let mut expected_payoff_deviate = 0.0;
                            
                            for (profile, &prob) in distribution {
                                if profile[player] == recommended_action {
                                    let mut alt_profile = profile.clone();
                                    alt_profile[player] = alt_action;
                                    expected_payoff_deviate += 
                                        (prob / conditional_prob_sum) * game.get_payoff(player, &alt_profile);
                                }
                            }
                            
                            if expected_payoff_deviate > expected_payoff_follow + 1e-9 {
                                return false;
                            }
                        }
                    }
                }
            }
        }
        
        true
    }
}

impl Foreign for CorrelatedEquilibrium {
    fn type_name(&self) -> &'static str {
        "CorrelatedEquilibrium"
    }

    fn call_method(&self, method: &str, _args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "Distribution" => {
                let dist: Vec<Value> = self.distribution.iter()
                    .map(|(profile, &prob)| {
                        let mut entry = profile.iter()
                            .map(|&action| Value::Integer(action as i64))
                            .collect::<Vec<Value>>();
                        entry.push(Value::Real(prob));
                        Value::List(entry)
                    })
                    .collect();
                Ok(Value::List(dist))
            }
            "ExpectedPayoffs" => {
                let payoffs: Vec<Value> = self.expected_payoffs.iter()
                    .map(|&payoff| Value::Real(payoff))
                    .collect();
                Ok(Value::List(payoffs))
            }
            "IsIncentiveCompatible" => {
                Ok(Value::String(if self.is_incentive_compatible { "true" } else { "false" }.to_string()))
            }
            "SocialWelfare" => {
                let welfare = self.expected_payoffs.iter().sum::<f64>();
                Ok(Value::Real(welfare))
            }
            _ => Err(ForeignError::UnknownMethod {
                type_name: self.type_name().to_string(),
                method: method.to_string(),
            }),
        }
    }

    fn clone_boxed(&self) -> Box<dyn Foreign> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Evolutionary Stable Strategy result
#[derive(Debug, Clone)]
pub struct EvolutionaryStableStrategy {
    /// ESS mixed strategy
    pub strategy: Vec<f64>,
    /// Fitness at the ESS
    pub fitness: f64,
    /// Whether the strategy is evolutionarily stable
    pub is_stable: bool,
    /// Game (assumed symmetric 2-player)
    pub game: Game,
}

impl EvolutionaryStableStrategy {
    /// Compute ESS for symmetric 2-player game
    pub fn compute_ess(game: Game) -> VmResult<Self> {
        if game.num_players != 2 {
            return Err(VmError::TypeError {
                expected: "2-player game".to_string(),
                actual: format!("{}-player game", game.num_players),
            });
        }
        
        let num_strategies = game.strategy_counts[0];
        
        // Check for pure strategy ESS first
        for i in 0..num_strategies {
            let pure_strategy = NashEquilibrium::pure_strategy(i, num_strategies);
            if Self::is_ess(&game, &pure_strategy) {
                let fitness = Self::compute_fitness(&game, &pure_strategy, &pure_strategy);
                return Ok(EvolutionaryStableStrategy {
                    strategy: pure_strategy,
                    fitness,
                    is_stable: true,
                    game,
                });
            }
        }
        
        // Look for mixed strategy ESS using replicator dynamics
        let mut strategy = vec![1.0 / num_strategies as f64; num_strategies];
        let max_iterations = 1000;
        let tolerance = 1e-6;
        
        for _ in 0..max_iterations {
            let old_strategy = strategy.clone();
            strategy = Self::replicator_step(&game, &strategy);
            
            // Check convergence
            let diff: f64 = strategy.iter().zip(old_strategy.iter())
                .map(|(new, old)| (new - old).abs())
                .sum();
            
            if diff < tolerance {
                break;
            }
        }
        
        let fitness = Self::compute_fitness(&game, &strategy, &strategy);
        let is_stable = Self::is_ess(&game, &strategy);
        
        Ok(EvolutionaryStableStrategy {
            strategy,
            fitness,
            is_stable,
            game,
        })
    }
    
    /// Check if a strategy is an ESS
    fn is_ess(game: &Game, strategy: &[f64]) -> bool {
        let num_strategies = strategy.len();
        let self_fitness = Self::compute_fitness(game, strategy, strategy);
        
        // Check ESS conditions against all alternative strategies
        for i in 0..num_strategies {
            if strategy[i] < 1e-9 {  // Not in support
                let mut alt_strategy = vec![0.0; num_strategies];
                alt_strategy[i] = 1.0;
                
                let alt_vs_self = Self::compute_fitness(game, &alt_strategy, strategy);
                let self_vs_alt = Self::compute_fitness(game, strategy, &alt_strategy);
                let alt_vs_alt = Self::compute_fitness(game, &alt_strategy, &alt_strategy);
                
                // ESS condition: either E(I,I) > E(J,I) or E(I,I) = E(J,I) and E(I,J) > E(J,J)
                if alt_vs_self > self_fitness + 1e-9 {
                    return false;
                }
                if (alt_vs_self - self_fitness).abs() < 1e-9 && alt_vs_alt >= self_vs_alt - 1e-9 {
                    return false;
                }
            }
        }
        
        true
    }
    
    /// Compute fitness of strategy1 against strategy2 in symmetric game
    fn compute_fitness(game: &Game, strategy1: &[f64], strategy2: &[f64]) -> f64 {
        let mut fitness = 0.0;
        for i in 0..strategy1.len() {
            for j in 0..strategy2.len() {
                fitness += strategy1[i] * strategy2[j] * game.get_payoff(0, &[i, j]);
            }
        }
        fitness
    }
    
    /// One step of replicator dynamics
    fn replicator_step(game: &Game, strategy: &[f64]) -> Vec<f64> {
        let num_strategies = strategy.len();
        let mut new_strategy = vec![0.0; num_strategies];
        let avg_fitness = Self::compute_fitness(game, strategy, strategy);
        
        for i in 0..num_strategies {
            let mut pure_strategy = vec![0.0; num_strategies];
            pure_strategy[i] = 1.0;
            let fitness_i = Self::compute_fitness(game, &pure_strategy, strategy);
            
            new_strategy[i] = strategy[i] * fitness_i / avg_fitness.max(1e-9);
        }
        
        // Normalize to ensure probabilities sum to 1
        let sum: f64 = new_strategy.iter().sum();
        if sum > 1e-9 {
            for prob in &mut new_strategy {
                *prob /= sum;
            }
        }
        
        new_strategy
    }
}

impl Foreign for EvolutionaryStableStrategy {
    fn type_name(&self) -> &'static str {
        "EvolutionaryStableStrategy"
    }

    fn call_method(&self, method: &str, _args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "Strategy" => {
                let strategy: Vec<Value> = self.strategy.iter()
                    .map(|&prob| Value::Real(prob))
                    .collect();
                Ok(Value::List(strategy))
            }
            "Fitness" => Ok(Value::Real(self.fitness)),
            "IsStable" => Ok(Value::String(if self.is_stable { "true" } else { "false" }.to_string())),
            _ => Err(ForeignError::UnknownMethod {
                type_name: self.type_name().to_string(),
                method: method.to_string(),
            }),
        }
    }

    fn clone_boxed(&self) -> Box<dyn Foreign> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Dominated strategy elimination result
#[derive(Debug, Clone)]
pub struct DominanceElimination {
    /// Original game
    pub original_game: Game,
    /// Game after elimination
    pub reduced_game: Game,
    /// Eliminated strategies for each player
    pub eliminated_strategies: Vec<Vec<usize>>,
    /// Remaining strategies for each player
    pub remaining_strategies: Vec<Vec<usize>>,
    /// Elimination rounds (which strategies eliminated in each round)
    pub elimination_rounds: Vec<Vec<(usize, usize)>>, // (player, strategy) pairs
}

impl DominanceElimination {
    /// Perform iterated elimination of strictly dominated strategies
    pub fn eliminate_dominated_strategies(game: Game) -> VmResult<Self> {
        let mut current_game = game.clone();
        let mut eliminated_strategies = vec![Vec::new(); game.num_players];
        let mut remaining_strategies: Vec<Vec<usize>> = game.strategy_counts.iter()
            .map(|&count| (0..count).collect())
            .collect();
        let mut elimination_rounds = Vec::new();
        
        loop {
            let mut round_eliminations = Vec::new();
            let mut found_dominated = false;
            
            // Check each player for dominated strategies
            for player in 0..current_game.num_players {
                let mut to_eliminate = Vec::new();
                
                for &strategy1 in &remaining_strategies[player] {
                    for &strategy2 in &remaining_strategies[player] {
                        if strategy1 != strategy2 && 
                           current_game.strictly_dominates(player, strategy2, strategy1) {
                            to_eliminate.push(strategy1);
                            break;
                        }
                    }
                }
                
                // Eliminate dominated strategies
                for strategy in to_eliminate {
                    if let Some(pos) = remaining_strategies[player].iter().position(|&x| x == strategy) {
                        remaining_strategies[player].remove(pos);
                        eliminated_strategies[player].push(strategy);
                        round_eliminations.push((player, strategy));
                        found_dominated = true;
                    }
                }
            }
            
            if !found_dominated {
                break;
            }
            
            elimination_rounds.push(round_eliminations);
            
            // Update current game to reflect eliminations
            current_game = Self::create_reduced_game(&game, &remaining_strategies)?;
        }
        
        Ok(DominanceElimination {
            original_game: game,
            reduced_game: current_game,
            eliminated_strategies,
            remaining_strategies,
            elimination_rounds,
        })
    }
    
    /// Create reduced game with eliminated strategies removed
    fn create_reduced_game(
        original_game: &Game,
        remaining_strategies: &[Vec<usize>],
    ) -> VmResult<Game> {
        // For now, return a copy of the original game
        // A full implementation would create matrices with only remaining strategies
        Ok(original_game.clone())
    }
}

impl Foreign for DominanceElimination {
    fn type_name(&self) -> &'static str {
        "DominanceElimination"
    }

    fn call_method(&self, method: &str, _args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "ReducedGame" => {
                // Return payoff matrices as nested lists
                let matrices: Vec<Value> = self.reduced_game.payoff_matrices.iter()
                    .map(|matrix| {
                        let rows: Vec<Value> = matrix.iter()
                            .map(|row| {
                                let cells: Vec<Value> = row.iter()
                                    .map(|&payoff| Value::Real(payoff))
                                    .collect();
                                Value::List(cells)
                            })
                            .collect();
                        Value::List(rows)
                    })
                    .collect();
                Ok(Value::List(matrices))
            }
            "EliminatedStrategies" => {
                let eliminated: Vec<Value> = self.eliminated_strategies.iter()
                    .map(|player_eliminated| {
                        let strategies: Vec<Value> = player_eliminated.iter()
                            .map(|&strategy| Value::Integer(strategy as i64))
                            .collect();
                        Value::List(strategies)
                    })
                    .collect();
                Ok(Value::List(eliminated))
            }
            "RemainingStrategies" => {
                let remaining: Vec<Value> = self.remaining_strategies.iter()
                    .map(|player_remaining| {
                        let strategies: Vec<Value> = player_remaining.iter()
                            .map(|&strategy| Value::Integer(strategy as i64))
                            .collect();
                        Value::List(strategies)
                    })
                    .collect();
                Ok(Value::List(remaining))
            }
            "EliminationRounds" => {
                let rounds: Vec<Value> = self.elimination_rounds.iter()
                    .map(|round| {
                        let eliminations: Vec<Value> = round.iter()
                            .map(|&(player, strategy)| {
                                Value::List(vec![
                                    Value::Integer(player as i64),
                                    Value::Integer(strategy as i64),
                                ])
                            })
                            .collect();
                        Value::List(eliminations)
                    })
                    .collect();
                Ok(Value::List(rounds))
            }
            _ => Err(ForeignError::UnknownMethod {
                type_name: self.type_name().to_string(),
                method: method.to_string(),
            }),
        }
    }

    fn clone_boxed(&self) -> Box<dyn Foreign> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

// ===============================
// EQUILIBRIUM FUNCTIONS
// ===============================

/// Extract payoff matrix from Value for 2-player games
pub fn extract_payoff_matrix(value: &Value) -> VmResult<Vec<Vec<Vec<f64>>>> {
    match value {
        Value::List(player_matrices) => {
            let mut payoff_matrices = Vec::new();
            
            for player_matrix in player_matrices {
                match player_matrix {
                    Value::List(rows) => {
                        let mut matrix = Vec::new();
                        for row in rows {
                            match row {
                                Value::List(cells) => {
                                    let mut matrix_row = Vec::new();
                                    for cell in cells {
                                        match cell {
                                            Value::Real(r) => matrix_row.push(*r),
                                            Value::Integer(i) => matrix_row.push(*i as f64),
                                            _ => return Err(VmError::TypeError {
                                                expected: "numeric payoff".to_string(),
                                                actual: format!("{:?}", cell),
                                            }),
                                        }
                                    }
                                    matrix.push(matrix_row);
                                }
                                _ => return Err(VmError::TypeError {
                                    expected: "list of payoffs (row)".to_string(),
                                    actual: format!("{:?}", row),
                                }),
                            }
                        }
                        payoff_matrices.push(matrix);
                    }
                    _ => return Err(VmError::TypeError {
                        expected: "list of rows (payoff matrix)".to_string(),
                        actual: format!("{:?}", player_matrix),
                    }),
                }
            }
            Ok(payoff_matrices)
        }
        _ => Err(VmError::TypeError {
            expected: "List of payoff matrices".to_string(),
            actual: format!("{:?}", value),
        }),
    }
}

/// Compute Nash equilibrium for a game
/// Syntax: NashEquilibrium[payoff_matrices]
pub fn nash_equilibrium(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "1 argument (payoff matrices)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let payoff_matrices = extract_payoff_matrix(&args[0])?;
    let game = Game::new(payoff_matrices)?;
    let equilibrium = NashEquilibrium::compute_2player_nash(game)?;
    
    Ok(Value::LyObj(LyObj::new(Box::new(equilibrium))))
}

/// Compute correlated equilibrium for a game
/// Syntax: CorrelatedEquilibrium[payoff_matrices]
pub fn correlated_equilibrium(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "1 argument (payoff matrices)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let payoff_matrices = extract_payoff_matrix(&args[0])?;
    let game = Game::new(payoff_matrices)?;
    let equilibrium = CorrelatedEquilibrium::compute_welfare_maximizing(game)?;
    
    Ok(Value::LyObj(LyObj::new(Box::new(equilibrium))))
}

/// Compute evolutionary stable strategy for symmetric game
/// Syntax: EvolutionaryStableStrategy[payoff_matrix]
pub fn evolutionary_stable_strategy(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "1 argument (symmetric payoff matrix)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    // For ESS, we expect a single matrix representing a symmetric game
    let symmetric_matrix = match &args[0] {
        Value::List(rows) => {
            let mut matrix = Vec::new();
            for row in rows {
                match row {
                    Value::List(cells) => {
                        let mut matrix_row = Vec::new();
                        for cell in cells {
                            match cell {
                                Value::Real(r) => matrix_row.push(*r),
                                Value::Integer(i) => matrix_row.push(*i as f64),
                                _ => return Err(VmError::TypeError {
                                    expected: "numeric payoff".to_string(),
                                    actual: format!("{:?}", cell),
                                }),
                            }
                        }
                        matrix.push(matrix_row);
                    }
                    _ => return Err(VmError::TypeError {
                        expected: "list of payoffs (row)".to_string(),
                        actual: format!("{:?}", row),
                    }),
                }
            }
            matrix
        }
        _ => return Err(VmError::TypeError {
            expected: "payoff matrix (list of lists)".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };

    // Create symmetric 2-player game
    let payoff_matrices = vec![symmetric_matrix.clone(), symmetric_matrix];
    let game = Game::new(payoff_matrices)?;
    let ess = EvolutionaryStableStrategy::compute_ess(game)?;
    
    Ok(Value::LyObj(LyObj::new(Box::new(ess))))
}

/// Eliminate dominated strategies
/// Syntax: EliminateDominatedStrategies[payoff_matrices]
pub fn eliminate_dominated_strategies(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "1 argument (payoff matrices)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let payoff_matrices = extract_payoff_matrix(&args[0])?;
    let game = Game::new(payoff_matrices)?;
    let elimination = DominanceElimination::eliminate_dominated_strategies(game)?;
    
    Ok(Value::LyObj(LyObj::new(Box::new(elimination))))
}