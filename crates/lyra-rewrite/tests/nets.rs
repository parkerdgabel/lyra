use lyra_rewrite::PatternNet;

#[test]
#[ignore]
fn pattern_net_indexes_and_filters_candidates() {
    let net = PatternNet::new();
    assert_eq!(net.len(), 0);
    // TODO: insert patterns and assert plausible candidate filtering.
}
