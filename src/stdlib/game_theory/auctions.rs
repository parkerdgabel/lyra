//! Auction Mechanisms for Game Theory
//!
//! This module implements various auction mechanisms including:
//! - First-price sealed-bid auctions
//! - Second-price (Vickrey) auctions
//! - English and Dutch auctions
//! - Combinatorial auctions with winner determination

use crate::foreign::{Foreign, ForeignError, LyObj};
use crate::vm::{Value, VmResult, VmError};
use std::any::Any;
use std::collections::{HashMap, HashSet};

/// Bidder information
#[derive(Debug, Clone)]
pub struct Bidder {
    /// Bidder ID
    pub id: usize,
    /// True valuation (private information)
    pub valuation: f64,
    /// Submitted bid
    pub bid: f64,
}

/// First-price sealed-bid auction result
#[derive(Debug, Clone)]
pub struct FirstPriceAuction {
    /// Bidders participating in auction
    pub bidders: Vec<Bidder>,
    /// Winner's bidder ID
    pub winner: usize,
    /// Winning bid (payment)
    pub winning_bid: f64,
    /// Auction revenue
    pub revenue: f64,
    /// Efficiency (value to winner / max possible value)
    pub efficiency: f64,
    /// All bids in descending order
    pub bid_ranking: Vec<(usize, f64)>,
}

impl FirstPriceAuction {
    /// Conduct first-price auction
    pub fn conduct(valuations: Vec<f64>, bids: Vec<f64>) -> VmResult<Self> {
        if valuations.len() != bids.len() {
            return Err(VmError::TypeError {
                expected: "equal number of valuations and bids".to_string(),
                actual: format!("valuations: {}, bids: {}", valuations.len(), bids.len()),
            });
        }

        if valuations.is_empty() {
            return Err(VmError::TypeError {
                expected: "at least one bidder".to_string(),
                actual: "no bidders".to_string(),
            });
        }

        let bidders: Vec<Bidder> = valuations.into_iter().zip(bids.into_iter())
            .enumerate()
            .map(|(id, (valuation, bid))| Bidder { id, valuation, bid })
            .collect();

        // Find winner (highest bidder)
        let winner_bidder = bidders.iter()
            .max_by(|a, b| a.bid.partial_cmp(&b.bid).unwrap())
            .unwrap();

        let winner = winner_bidder.id;
        let winning_bid = winner_bidder.bid;
        let revenue = winning_bid;

        // Calculate efficiency
        let max_valuation = bidders.iter().map(|b| b.valuation).fold(f64::NEG_INFINITY, f64::max);
        let efficiency = if max_valuation > 0.0 {
            winner_bidder.valuation / max_valuation
        } else {
            0.0
        };

        // Create bid ranking
        let mut bid_ranking: Vec<(usize, f64)> = bidders.iter()
            .map(|b| (b.id, b.bid))
            .collect();
        bid_ranking.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        Ok(FirstPriceAuction {
            bidders,
            winner,
            winning_bid,
            revenue,
            efficiency,
            bid_ranking,
        })
    }
}

impl Foreign for FirstPriceAuction {
    fn type_name(&self) -> &'static str {
        "FirstPriceAuction"
    }

    fn call_method(&self, method: &str, _args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "Winner" => Ok(Value::Integer(self.winner as i64)),
            "WinningBid" => Ok(Value::Real(self.winning_bid)),
            "Revenue" => Ok(Value::Real(self.revenue)),
            "Efficiency" => Ok(Value::Real(self.efficiency)),
            "BidRanking" => {
                let ranking: Vec<Value> = self.bid_ranking.iter()
                    .map(|&(bidder_id, bid)| {
                        Value::List(vec![
                            Value::Integer(bidder_id as i64),
                            Value::Real(bid),
                        ])
                    })
                    .collect();
                Ok(Value::List(ranking))
            }
            "WinnerValuation" => {
                let winner_valuation = self.bidders[self.winner].valuation;
                Ok(Value::Real(winner_valuation))
            }
            "WinnerSurplus" => {
                let winner_bidder = &self.bidders[self.winner];
                let surplus = winner_bidder.valuation - winner_bidder.bid;
                Ok(Value::Real(surplus))
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

/// Second-price (Vickrey) auction result
#[derive(Debug, Clone)]
pub struct SecondPriceAuction {
    /// Bidders participating in auction
    pub bidders: Vec<Bidder>,
    /// Winner's bidder ID
    pub winner: usize,
    /// Winning payment (second-highest bid)
    pub winning_payment: f64,
    /// Auction revenue
    pub revenue: f64,
    /// Whether auction is truthful (dominant strategy is truthful bidding)
    pub is_truthful: bool,
    /// All bids in descending order
    pub bid_ranking: Vec<(usize, f64)>,
}

impl SecondPriceAuction {
    /// Conduct second-price auction
    pub fn conduct(valuations: Vec<f64>, bids: Vec<f64>) -> VmResult<Self> {
        if valuations.len() != bids.len() {
            return Err(VmError::TypeError {
                expected: "equal number of valuations and bids".to_string(),
                actual: format!("valuations: {}, bids: {}", valuations.len(), bids.len()),
            });
        }

        if valuations.len() < 2 {
            return Err(VmError::TypeError {
                expected: "at least two bidders".to_string(),
                actual: format!("{} bidders", valuations.len()),
            });
        }

        let bidders: Vec<Bidder> = valuations.into_iter().zip(bids.into_iter())
            .enumerate()
            .map(|(id, (valuation, bid))| Bidder { id, valuation, bid })
            .collect();

        // Sort by bids to find winner and second-highest bid
        let mut sorted_bidders = bidders.clone();
        sorted_bidders.sort_by(|a, b| b.bid.partial_cmp(&a.bid).unwrap());

        let winner = sorted_bidders[0].id;
        let winning_payment = if sorted_bidders.len() > 1 {
            sorted_bidders[1].bid
        } else {
            0.0
        };
        let revenue = winning_payment;

        // Create bid ranking
        let bid_ranking: Vec<(usize, f64)> = sorted_bidders.iter()
            .map(|b| (b.id, b.bid))
            .collect();

        Ok(SecondPriceAuction {
            bidders,
            winner,
            winning_payment,
            revenue,
            is_truthful: true, // Second-price auctions are always truthful
            bid_ranking,
        })
    }
}

impl Foreign for SecondPriceAuction {
    fn type_name(&self) -> &'static str {
        "SecondPriceAuction"
    }

    fn call_method(&self, method: &str, _args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "Winner" => Ok(Value::Integer(self.winner as i64)),
            "WinningPayment" => Ok(Value::Real(self.winning_payment)),
            "Revenue" => Ok(Value::Real(self.revenue)),
            "IsTruthful" => Ok(Value::String(if self.is_truthful { "true" } else { "false" }.to_string())),
            "BidRanking" => {
                let ranking: Vec<Value> = self.bid_ranking.iter()
                    .map(|&(bidder_id, bid)| {
                        Value::List(vec![
                            Value::Integer(bidder_id as i64),
                            Value::Real(bid),
                        ])
                    })
                    .collect();
                Ok(Value::List(ranking))
            }
            "WinnerSurplus" => {
                let winner_bidder = &self.bidders[self.winner];
                let surplus = winner_bidder.valuation - self.winning_payment;
                Ok(Value::Real(surplus))
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

/// Multi-unit Vickrey auction result
#[derive(Debug, Clone)]
pub struct VickreyAuction {
    /// Bidder valuations for each item
    pub bidder_valuations: Vec<Vec<f64>>,
    /// Final allocation (bidder -> list of items)
    pub allocation: HashMap<usize, Vec<usize>>,
    /// VCG payments for each bidder
    pub payments: Vec<f64>,
    /// Total auction revenue
    pub revenue: f64,
    /// Whether auction is truthful
    pub is_truthful: bool,
    /// Social welfare achieved
    pub social_welfare: f64,
}

impl VickreyAuction {
    /// Conduct Vickrey auction for multiple items
    pub fn conduct(bidder_valuations: Vec<Vec<f64>>) -> VmResult<Self> {
        if bidder_valuations.is_empty() {
            return Err(VmError::TypeError {
                expected: "at least one bidder".to_string(),
                actual: "no bidders".to_string(),
            });
        }

        let num_bidders = bidder_valuations.len();
        let num_items = bidder_valuations[0].len();

        // Verify all bidders have valuations for all items
        for (i, valuations) in bidder_valuations.iter().enumerate() {
            if valuations.len() != num_items {
                return Err(VmError::TypeError {
                    expected: format!("{} item valuations", num_items),
                    actual: format!("bidder {} has {} valuations", i, valuations.len()),
                });
            }
        }

        // Find welfare-maximizing allocation using greedy assignment
        let mut allocation = HashMap::new();
        let mut allocated_items = HashSet::new();
        
        // Create (bidder, item, value) tuples and sort by value
        let mut value_tuples: Vec<(usize, usize, f64)> = Vec::new();
        for (bidder, valuations) in bidder_valuations.iter().enumerate() {
            for (item, &value) in valuations.iter().enumerate() {
                value_tuples.push((bidder, item, value));
            }
        }
        value_tuples.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());

        // Greedy allocation
        for (bidder, item, _value) in value_tuples {
            if !allocated_items.contains(&item) {
                allocation.entry(bidder).or_insert_with(Vec::new).push(item);
                allocated_items.insert(item);
            }
        }

        // Compute VCG payments
        let mut payments = vec![0.0; num_bidders];
        let social_welfare = Self::compute_social_welfare(&bidder_valuations, &allocation);
        
        for bidder in 0..num_bidders {
            // Compute welfare without this bidder
            let mut reduced_valuations = bidder_valuations.clone();
            reduced_valuations.remove(bidder);
            
            let reduced_allocation = Self::compute_optimal_allocation(&reduced_valuations);
            let welfare_without_bidder = Self::compute_social_welfare(&reduced_valuations, &reduced_allocation);
            
            // Compute welfare of others in current allocation
            let welfare_of_others = social_welfare - 
                allocation.get(&bidder).unwrap_or(&Vec::new()).iter()
                    .map(|&item| bidder_valuations[bidder][item])
                    .sum::<f64>();
            
            // VCG payment = welfare loss to others
            payments[bidder] = welfare_without_bidder - welfare_of_others;
            payments[bidder] = payments[bidder].max(0.0); // Non-negative
        }

        let revenue = payments.iter().sum();

        Ok(VickreyAuction {
            bidder_valuations,
            allocation,
            payments,
            revenue,
            is_truthful: true, // Vickrey auctions are truthful
            social_welfare,
        })
    }

    /// Compute optimal allocation for given valuations
    fn compute_optimal_allocation(valuations: &[Vec<f64>]) -> HashMap<usize, Vec<usize>> {
        let mut allocation = HashMap::new();
        let mut allocated_items = HashSet::new();
        
        let num_items = if valuations.is_empty() { 0 } else { valuations[0].len() };
        
        // Create and sort value tuples
        let mut value_tuples: Vec<(usize, usize, f64)> = Vec::new();
        for (bidder, bidder_valuations) in valuations.iter().enumerate() {
            for (item, &value) in bidder_valuations.iter().enumerate() {
                value_tuples.push((bidder, item, value));
            }
        }
        value_tuples.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());

        // Greedy allocation
        for (bidder, item, _value) in value_tuples {
            if !allocated_items.contains(&item) {
                allocation.entry(bidder).or_insert_with(Vec::new).push(item);
                allocated_items.insert(item);
            }
        }

        allocation
    }

    /// Compute social welfare for an allocation
    fn compute_social_welfare(
        valuations: &[Vec<f64>], 
        allocation: &HashMap<usize, Vec<usize>>
    ) -> f64 {
        allocation.iter()
            .map(|(&bidder, items)| {
                items.iter()
                    .map(|&item| valuations[bidder][item])
                    .sum::<f64>()
            })
            .sum()
    }
}

impl Foreign for VickreyAuction {
    fn type_name(&self) -> &'static str {
        "VickreyAuction"
    }

    fn call_method(&self, method: &str, _args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "Allocation" => {
                let num_bidders = self.bidder_valuations.len();
                let allocation: Vec<Value> = (0..num_bidders)
                    .map(|bidder| {
                        let empty_vec = Vec::new();
                        let items = self.allocation.get(&bidder).unwrap_or(&empty_vec);
                        let item_list: Vec<Value> = items.iter()
                            .map(|&item| Value::Integer(item as i64))
                            .collect();
                        Value::List(item_list)
                    })
                    .collect();
                Ok(Value::List(allocation))
            }
            "Payments" => {
                let payments: Vec<Value> = self.payments.iter()
                    .map(|&payment| Value::Real(payment))
                    .collect();
                Ok(Value::List(payments))
            }
            "Revenue" => Ok(Value::Real(self.revenue)),
            "IsTruthful" => Ok(Value::String(if self.is_truthful { "true" } else { "false" }.to_string())),
            "SocialWelfare" => Ok(Value::Real(self.social_welfare)),
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

/// Package bid for combinatorial auction
#[derive(Debug, Clone)]
pub struct PackageBid {
    /// Bidder ID
    pub bidder: usize,
    /// Set of items in package
    pub package: HashSet<usize>,
    /// Bid amount
    pub bid: f64,
}

/// Combinatorial auction result
#[derive(Debug, Clone)]
pub struct CombinatorialAuction {
    /// All package bids
    pub package_bids: Vec<PackageBid>,
    /// Winning bids
    pub winners: Vec<PackageBid>,
    /// Total social welfare achieved
    pub social_welfare: f64,
    /// Total revenue
    pub revenue: f64,
    /// Final allocation
    pub allocation: HashMap<usize, HashSet<usize>>,
}

impl CombinatorialAuction {
    /// Conduct combinatorial auction with winner determination
    pub fn conduct(package_bids: Vec<(HashSet<usize>, f64)>) -> VmResult<Self> {
        if package_bids.is_empty() {
            return Err(VmError::TypeError {
                expected: "at least one package bid".to_string(),
                actual: "no bids".to_string(),
            });
        }

        let bids: Vec<PackageBid> = package_bids.into_iter()
            .enumerate()
            .map(|(bidder, (package, bid))| PackageBid { bidder, package, bid })
            .collect();

        // Solve winner determination problem (maximize social welfare)
        let winners = Self::solve_winner_determination(&bids);
        
        let social_welfare = winners.iter().map(|bid| bid.bid).sum();
        let revenue = social_welfare; // In first-price combinatorial auction

        // Build allocation map
        let mut allocation = HashMap::new();
        for winner in &winners {
            allocation.insert(winner.bidder, winner.package.clone());
        }

        Ok(CombinatorialAuction {
            package_bids: bids,
            winners,
            social_welfare,
            revenue,
            allocation,
        })
    }

    /// Solve winner determination problem using greedy approximation
    fn solve_winner_determination(bids: &[PackageBid]) -> Vec<PackageBid> {
        let mut winners = Vec::new();
        let mut allocated_items = HashSet::new();
        
        // Sort bids by value density (bid / package size) in descending order
        let mut sorted_bids = bids.to_vec();
        sorted_bids.sort_by(|a, b| {
            let density_a = a.bid / a.package.len() as f64;
            let density_b = b.bid / b.package.len() as f64;
            density_b.partial_cmp(&density_a).unwrap()
        });

        // Greedy selection
        for bid in sorted_bids {
            // Check if package conflicts with already allocated items
            if bid.package.is_disjoint(&allocated_items) {
                allocated_items.extend(&bid.package);
                winners.push(bid);
            }
        }

        winners
    }
}

impl Foreign for CombinatorialAuction {
    fn type_name(&self) -> &'static str {
        "CombinatorialAuction"
    }

    fn call_method(&self, method: &str, _args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "Winners" => {
                let winners: Vec<Value> = self.winners.iter()
                    .map(|winner| {
                        let package: Vec<Value> = winner.package.iter()
                            .map(|&item| Value::Integer(item as i64))
                            .collect();
                        Value::List(vec![
                            Value::Integer(winner.bidder as i64),
                            Value::List(package),
                            Value::Real(winner.bid),
                        ])
                    })
                    .collect();
                Ok(Value::List(winners))
            }
            "SocialWelfare" => Ok(Value::Real(self.social_welfare)),
            "Revenue" => Ok(Value::Real(self.revenue)),
            "Allocation" => {
                let allocation: Vec<Value> = self.allocation.iter()
                    .map(|(&bidder, package)| {
                        let items: Vec<Value> = package.iter()
                            .map(|&item| Value::Integer(item as i64))
                            .collect();
                        Value::List(vec![
                            Value::Integer(bidder as i64),
                            Value::List(items),
                        ])
                    })
                    .collect();
                Ok(Value::List(allocation))
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

/// English (ascending) auction result
#[derive(Debug, Clone)]
pub struct EnglishAuction {
    /// Bidder valuations
    pub valuations: Vec<f64>,
    /// Final price
    pub final_price: f64,
    /// Winner
    pub winner: usize,
    /// Number of bidding rounds
    pub rounds: usize,
    /// Bidding sequence
    pub bidding_sequence: Vec<(f64, Vec<usize>)>, // (price, active_bidders)
    /// Price increment
    pub increment: f64,
    /// Reserve price
    pub reserve_price: f64,
}

impl EnglishAuction {
    /// Simulate English auction
    pub fn simulate(
        valuations: Vec<f64>, 
        increment: f64, 
        reserve_price: f64
    ) -> VmResult<Self> {
        if valuations.is_empty() {
            return Err(VmError::TypeError {
                expected: "at least one bidder".to_string(),
                actual: "no bidders".to_string(),
            });
        }

        let mut current_price = reserve_price;
        let mut active_bidders: Vec<usize> = (0..valuations.len()).collect();
        let mut bidding_sequence = Vec::new();
        let mut rounds = 0;

        // Remove bidders with valuations below reserve price
        active_bidders.retain(|&bidder| valuations[bidder] >= reserve_price);
        
        if active_bidders.is_empty() {
            return Err(VmError::TypeError {
                expected: "at least one bidder above reserve price".to_string(),
                actual: "no bidders above reserve".to_string(),
            });
        }

        bidding_sequence.push((current_price, active_bidders.clone()));

        // Continue until only one bidder remains
        while active_bidders.len() > 1 {
            current_price += increment;
            rounds += 1;

            // Remove bidders whose valuation is below current price
            active_bidders.retain(|&bidder| valuations[bidder] >= current_price);
            
            if active_bidders.is_empty() {
                // Should not happen if we have at least one bidder above reserve
                current_price -= increment;
                active_bidders.push(
                    (0..valuations.len())
                        .max_by(|&a, &b| valuations[a].partial_cmp(&valuations[b]).unwrap())
                        .unwrap()
                );
                break;
            }

            bidding_sequence.push((current_price, active_bidders.clone()));
        }

        let winner = active_bidders[0];
        let final_price = current_price;

        Ok(EnglishAuction {
            valuations,
            final_price,
            winner,
            rounds,
            bidding_sequence,
            increment,
            reserve_price,
        })
    }
}

impl Foreign for EnglishAuction {
    fn type_name(&self) -> &'static str {
        "EnglishAuction"
    }

    fn call_method(&self, method: &str, _args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "Winner" => Ok(Value::Integer(self.winner as i64)),
            "FinalPrice" => Ok(Value::Real(self.final_price)),
            "Rounds" => Ok(Value::Integer(self.rounds as i64)),
            "BiddingSequence" => {
                let sequence: Vec<Value> = self.bidding_sequence.iter()
                    .map(|(price, active)| {
                        let active_list: Vec<Value> = active.iter()
                            .map(|&bidder| Value::Integer(bidder as i64))
                            .collect();
                        Value::List(vec![
                            Value::Real(*price),
                            Value::List(active_list),
                        ])
                    })
                    .collect();
                Ok(Value::List(sequence))
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

/// Dutch (descending) auction result
#[derive(Debug, Clone)]
pub struct DutchAuction {
    /// Bidder valuations
    pub valuations: Vec<f64>,
    /// Starting price
    pub starting_price: f64,
    /// Price decrement per round
    pub decrement: f64,
    /// Winner
    pub winner: usize,
    /// Winning price (price at which winner accepted)
    pub winning_price: f64,
    /// Number of rounds until acceptance
    pub rounds: usize,
}

impl DutchAuction {
    /// Simulate Dutch auction
    pub fn simulate(
        valuations: Vec<f64>,
        starting_price: f64,
        decrement: f64,
    ) -> VmResult<Self> {
        if valuations.is_empty() {
            return Err(VmError::TypeError {
                expected: "at least one bidder".to_string(),
                actual: "no bidders".to_string(),
            });
        }

        if decrement <= 0.0 {
            return Err(VmError::TypeError {
                expected: "positive price decrement".to_string(),
                actual: format!("decrement: {}", decrement),
            });
        }

        let mut current_price = starting_price;
        let mut rounds = 0;

        // Find the bidder with highest valuation (they will accept first)
        let winner = valuations.iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap().0;

        let winner_valuation = valuations[winner];

        // Decrease price until winner's valuation is reached
        while current_price > winner_valuation {
            current_price -= decrement;
            rounds += 1;
        }

        // Winner accepts at first price <= their valuation
        let winning_price = current_price;

        Ok(DutchAuction {
            valuations,
            starting_price,
            decrement,
            winner,
            winning_price,
            rounds,
        })
    }
}

impl Foreign for DutchAuction {
    fn type_name(&self) -> &'static str {
        "DutchAuction"
    }

    fn call_method(&self, method: &str, _args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "Winner" => Ok(Value::Integer(self.winner as i64)),
            "WinningPrice" => Ok(Value::Real(self.winning_price)),
            "Rounds" => Ok(Value::Integer(self.rounds as i64)),
            "StartingPrice" => Ok(Value::Real(self.starting_price)),
            "Decrement" => Ok(Value::Real(self.decrement)),
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
// AUCTION FUNCTIONS
// ===============================

/// Extract list of reals from Value
pub fn extract_real_list(value: &Value) -> VmResult<Vec<f64>> {
    match value {
        Value::List(list) => {
            let mut reals = Vec::new();
            for item in list {
                match item {
                    Value::Real(r) => reals.push(*r),
                    Value::Integer(i) => reals.push(*i as f64),
                    _ => return Err(VmError::TypeError {
                        expected: "numeric value".to_string(),
                        actual: format!("{:?}", item),
                    }),
                }
            }
            Ok(reals)
        }
        _ => Err(VmError::TypeError {
            expected: "List of numbers".to_string(),
            actual: format!("{:?}", value),
        }),
    }
}

/// Conduct first-price auction
/// Syntax: FirstPriceAuction[valuations, bids]
pub fn first_price_auction(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::TypeError {
            expected: "2 arguments (valuations, bids)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let valuations = extract_real_list(&args[0])?;
    let bids = extract_real_list(&args[1])?;

    let auction = FirstPriceAuction::conduct(valuations, bids)?;
    Ok(Value::LyObj(LyObj::new(Box::new(auction))))
}

/// Conduct second-price auction
/// Syntax: SecondPriceAuction[valuations, bids]
pub fn second_price_auction(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::TypeError {
            expected: "2 arguments (valuations, bids)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let valuations = extract_real_list(&args[0])?;
    let bids = extract_real_list(&args[1])?;

    let auction = SecondPriceAuction::conduct(valuations, bids)?;
    Ok(Value::LyObj(LyObj::new(Box::new(auction))))
}

/// Conduct Vickrey auction
/// Syntax: VickreyAuction[bidder_valuations]
pub fn vickrey_auction(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "1 argument (bidder valuations matrix)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    // Extract matrix of valuations
    let bidder_valuations = match &args[0] {
        Value::List(bidders) => {
            let mut valuations = Vec::new();
            for bidder in bidders {
                let bidder_vals = extract_real_list(bidder)?;
                valuations.push(bidder_vals);
            }
            valuations
        }
        _ => return Err(VmError::TypeError {
            expected: "List of bidder valuations".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };

    let auction = VickreyAuction::conduct(bidder_valuations)?;
    Ok(Value::LyObj(LyObj::new(Box::new(auction))))
}

/// Conduct combinatorial auction
/// Syntax: CombinatorialAuction[package_bids]
pub fn combinatorial_auction(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "1 argument (package bids)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    // Extract package bids
    let package_bids = match &args[0] {
        Value::List(bids) => {
            let mut packages = Vec::new();
            for bid in bids {
                match bid {
                    Value::List(bid_info) => {
                        if bid_info.len() != 2 {
                            return Err(VmError::TypeError {
                                expected: "bid with [package, amount]".to_string(),
                                actual: format!("bid with {} elements", bid_info.len()),
                            });
                        }

                        // Extract package (set of item indices)
                        let package = match &bid_info[0] {
                            Value::List(items) => {
                                let mut item_set = HashSet::new();
                                for item in items {
                                    match item {
                                        Value::Integer(i) => {
                                            item_set.insert(*i as usize);
                                        }
                                        _ => return Err(VmError::TypeError {
                                            expected: "integer item index".to_string(),
                                            actual: format!("{:?}", item),
                                        }),
                                    }
                                }
                                item_set
                            }
                            _ => return Err(VmError::TypeError {
                                expected: "list of item indices".to_string(),
                                actual: format!("{:?}", bid_info[0]),
                            }),
                        };

                        // Extract bid amount
                        let amount = match &bid_info[1] {
                            Value::Real(r) => *r,
                            Value::Integer(i) => *i as f64,
                            _ => return Err(VmError::TypeError {
                                expected: "numeric bid amount".to_string(),
                                actual: format!("{:?}", bid_info[1]),
                            }),
                        };

                        packages.push((package, amount));
                    }
                    _ => return Err(VmError::TypeError {
                        expected: "bid as list [package, amount]".to_string(),
                        actual: format!("{:?}", bid),
                    }),
                }
            }
            packages
        }
        _ => return Err(VmError::TypeError {
            expected: "List of package bids".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };

    let auction = CombinatorialAuction::conduct(package_bids)?;
    Ok(Value::LyObj(LyObj::new(Box::new(auction))))
}

/// Simulate English auction
/// Syntax: EnglishAuction[valuations, increment, reserve_price]
pub fn english_auction(args: &[Value]) -> VmResult<Value> {
    if args.len() != 3 {
        return Err(VmError::TypeError {
            expected: "3 arguments (valuations, increment, reserve_price)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let valuations = extract_real_list(&args[0])?;
    
    let increment = match &args[1] {
        Value::Real(r) => *r,
        Value::Integer(i) => *i as f64,
        _ => return Err(VmError::TypeError {
            expected: "numeric increment".to_string(),
            actual: format!("{:?}", args[1]),
        }),
    };

    let reserve_price = match &args[2] {
        Value::Real(r) => *r,
        Value::Integer(i) => *i as f64,
        _ => return Err(VmError::TypeError {
            expected: "numeric reserve price".to_string(),
            actual: format!("{:?}", args[2]),
        }),
    };

    let auction = EnglishAuction::simulate(valuations, increment, reserve_price)?;
    Ok(Value::LyObj(LyObj::new(Box::new(auction))))
}

/// Simulate Dutch auction
/// Syntax: DutchAuction[valuations, starting_price, decrement]
pub fn dutch_auction(args: &[Value]) -> VmResult<Value> {
    if args.len() != 3 {
        return Err(VmError::TypeError {
            expected: "3 arguments (valuations, starting_price, decrement)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let valuations = extract_real_list(&args[0])?;
    
    let starting_price = match &args[1] {
        Value::Real(r) => *r,
        Value::Integer(i) => *i as f64,
        _ => return Err(VmError::TypeError {
            expected: "numeric starting price".to_string(),
            actual: format!("{:?}", args[1]),
        }),
    };

    let decrement = match &args[2] {
        Value::Real(r) => *r,
        Value::Integer(i) => *i as f64,
        _ => return Err(VmError::TypeError {
            expected: "numeric decrement".to_string(),
            actual: format!("{:?}", args[2]),
        }),
    };

    let auction = DutchAuction::simulate(valuations, starting_price, decrement)?;
    Ok(Value::LyObj(LyObj::new(Box::new(auction))))
}