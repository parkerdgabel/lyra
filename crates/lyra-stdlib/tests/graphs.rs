#![cfg(feature = "graphs")]

use lyra_core::value::Value;
use lyra_runtime::Evaluator;
use lyra_stdlib as stdlib;

#[test]
fn graphs_basic_create_add_list() {
    let mut ev = Evaluator::new();
    stdlib::register_all(&mut ev);
    // Create directed simple graph
    let g = ev.eval(Value::expr(
        Value::Symbol("GraphCreate".into()),
        vec![Value::Assoc(
            [
                ("Directed".to_string(), Value::Boolean(true)),
                ("Multi".to_string(), Value::Boolean(false)),
            ]
            .into_iter()
            .collect(),
        )],
    ));
    // Add nodes and edges
    let nodes = Value::List(vec![
        Value::String("A".into()),
        Value::String("B".into()),
        Value::String("C".into()),
    ]);
    let _n_added = ev.eval(Value::expr(Value::Symbol("AddNodes".into()), vec![g.clone(), nodes]));
    let edges = Value::List(vec![
        Value::Assoc(
            [
                ("Src".to_string(), Value::String("A".into())),
                ("Dst".to_string(), Value::String("B".into())),
            ]
            .into_iter()
            .collect(),
        ),
        Value::Assoc(
            [
                ("Src".to_string(), Value::String("B".into())),
                ("Dst".to_string(), Value::String("C".into())),
            ]
            .into_iter()
            .collect(),
        ),
        Value::Assoc(
            [
                ("Src".to_string(), Value::String("A".into())),
                ("Dst".to_string(), Value::String("C".into())),
            ]
            .into_iter()
            .collect(),
        ),
    ]);
    let _e_added = ev.eval(Value::expr(Value::Symbol("AddEdges".into()), vec![g.clone(), edges]));
    // Sanity: HasNode/HasEdge
    let has_a = ev.eval(Value::expr(
        Value::Symbol("HasNode".into()),
        vec![g.clone(), Value::String("A".into())],
    ));
    assert!(matches!(has_a, lyra_core::value::Value::Boolean(true)));
    let has_edge = ev.eval(Value::expr(
        Value::Symbol("HasEdge".into()),
        vec![g.clone(), Value::String("A".into()), Value::String("C".into())],
    ));
    assert!(matches!(has_edge, lyra_core::value::Value::Boolean(true)));
    // Neighbors
    let neigh = ev.eval(Value::expr(
        Value::Symbol("Neighbors".into()),
        vec![g.clone(), Value::String("A".into()), Value::String("out".into())],
    ));
    if let lyra_core::value::Value::List(xs) = neigh {
        assert_eq!(xs.len(), 2);
    } else {
        panic!("Neighbors not list");
    }
    // ListNodes/ListEdges return Dataset handles currently; ensure they evaluate to DatasetFromRows expr
    let _nodes = ev.eval(Value::expr(Value::Symbol("ListNodes".into()), vec![g.clone()]));
    let _edges = ev.eval(Value::expr(Value::Symbol("ListEdges".into()), vec![g.clone()]));
    // GraphInfo
    let info = ev.eval(Value::expr(Value::Symbol("GraphInfo".into()), vec![g.clone()]));
    if let lyra_core::value::Value::Assoc(m) = info {
        assert!(matches!(m.get("nodes"), Some(lyra_core::value::Value::Integer(n)) if *n==3));
        assert!(matches!(m.get("edges"), Some(lyra_core::value::Value::Integer(n)) if *n==3));
    } else {
        panic!("GraphInfo invalid");
    }
}

#[test]
fn graphs_upsert_remove_incident_subgraph_sample() {
    let mut ev = Evaluator::new();
    stdlib::register_all(&mut ev);
    // Build graph
    let g = ev.eval(Value::expr(
        Value::Symbol("GraphCreate".into()),
        vec![Value::Assoc(
            [
                ("Directed".to_string(), Value::Boolean(true)),
                ("Multi".to_string(), Value::Boolean(false)),
            ]
            .into_iter()
            .collect(),
        )],
    ));
    let nodes = Value::List(vec![
        Value::String("N1".into()),
        Value::String("N2".into()),
        Value::String("N3".into()),
    ]);
    let _ = ev.eval(Value::expr(Value::Symbol("AddNodes".into()), vec![g.clone(), nodes]));
    let edges = Value::List(vec![
        Value::Assoc(
            [
                ("Src".into(), Value::String("N1".into())),
                ("Dst".into(), Value::String("N2".into())),
            ]
            .into_iter()
            .collect(),
        ),
        Value::Assoc(
            [
                ("Src".into(), Value::String("N2".into())),
                ("Dst".into(), Value::String("N3".into())),
            ]
            .into_iter()
            .collect(),
        ),
    ]);
    let _ = ev.eval(Value::expr(Value::Symbol("AddEdges".into()), vec![g.clone(), edges]));
    // Upsert node with attrs
    let _ = ev.eval(Value::expr(
        Value::Symbol("UpsertNodes".into()),
        vec![
            g.clone(),
            Value::Assoc(
                [
                    ("id".into(), Value::String("N2".into())),
                    (
                        "attrs".into(),
                        Value::Assoc(
                            [("role".into(), Value::String("mid".into()))].into_iter().collect(),
                        ),
                    ),
                ]
                .into_iter()
                .collect(),
            ),
        ],
    ));
    // Upsert edge weight
    let _ = ev.eval(Value::expr(
        Value::Symbol("UpsertEdges".into()),
        vec![
            g.clone(),
            Value::Assoc(
                [
                    ("Src".into(), Value::String("N1".into())),
                    ("Dst".into(), Value::String("N2".into())),
                    ("Weight".into(), Value::Real(2.5)),
                ]
                .into_iter()
                .collect(),
            ),
        ],
    ));
    // Incident edges on N2
    let inc = ev.eval(Value::expr(
        Value::Symbol("IncidentEdges".into()),
        vec![g.clone(), Value::String("N2".into()), Value::String("all".into())],
    ));
    let inc_txt = lyra_core::pretty::format_value(&inc);
    assert!(inc_txt.contains("\"src\" -> \"N1\"") || inc_txt.contains("\"dst\" -> \"N2\""));
    // Subgraph on {N1,N2}
    let sg = ev.eval(Value::expr(
        Value::Symbol("Subgraph".into()),
        vec![g.clone(), Value::List(vec![Value::String("N1".into()), Value::String("N2".into())])],
    ));
    let info = ev.eval(Value::expr(Value::Symbol("GraphInfo".into()), vec![sg.clone()]));
    let info_txt = lyra_core::pretty::format_value(&info);
    assert!(info_txt.contains("\"nodes\" -> 2"));
    // Sampling
    let _sn = ev.eval(Value::expr(
        Value::Symbol("SampleNodes".into()),
        vec![
            g.clone(),
            Value::Integer(2),
            Value::Assoc([("Seed".into(), Value::Integer(42))].into_iter().collect()),
        ],
    ));
    let _se = ev.eval(Value::expr(
        Value::Symbol("SampleEdges".into()),
        vec![
            g.clone(),
            Value::Integer(1),
            Value::Assoc([("Seed".into(), Value::Integer(7))].into_iter().collect()),
        ],
    ));
    // Remove edge and node
    let _ = ev.eval(Value::expr(
        Value::Symbol("RemoveEdges".into()),
        vec![
            g.clone(),
            Value::Assoc(
                [
                    ("Src".into(), Value::String("N1".into())),
                    ("Dst".into(), Value::String("N2".into())),
                ]
                .into_iter()
                .collect(),
            ),
        ],
    ));
    let _ = ev.eval(Value::expr(
        Value::Symbol("RemoveNodes".into()),
        vec![g.clone(), Value::String("N3".into())],
    ));
    let info2 = ev.eval(Value::expr(Value::Symbol("GraphInfo".into()), vec![g.clone()]));
    if let Value::Assoc(m) = info2 {
        assert!(matches!(m.get("nodes"), Some(Value::Integer(n)) if *n==2));
    } else {
        panic!("GraphInfo invalid");
    }
}

#[test]
fn graphs_algorithms_basic() {
    let mut ev = Evaluator::new();
    stdlib::register_all(&mut ev);
    // Build a small DAG A->B->C, A->C, C->D
    let g = ev.eval(Value::expr(
        Value::Symbol("GraphCreate".into()),
        vec![Value::Assoc(
            [
                ("Directed".to_string(), Value::Boolean(true)),
                ("Multi".to_string(), Value::Boolean(false)),
            ]
            .into_iter()
            .collect(),
        )],
    ));
    let _ = ev.eval(Value::expr(
        Value::Symbol("AddNodes".into()),
        vec![
            g.clone(),
            Value::List(
                vec!["A", "B", "C", "D"].into_iter().map(|s| Value::String(s.into())).collect(),
            ),
        ],
    ));
    let edges = vec![("A", "B"), ("B", "C"), ("A", "C"), ("C", "D")];
    let edge_rows: Vec<Value> = edges
        .into_iter()
        .map(|(s, d)| {
            Value::Assoc(
                [("Src".into(), Value::String(s.into())), ("Dst".into(), Value::String(d.into()))]
                    .into_iter()
                    .collect(),
            )
        })
        .collect();
    let _ = ev.eval(Value::expr(
        Value::Symbol("AddEdges".into()),
        vec![g.clone(), Value::List(edge_rows)],
    ));
    // BFS
    let bfs_res = ev
        .eval(Value::expr(Value::Symbol("BFS".into()), vec![g.clone(), Value::String("A".into())]));
    let bfs_txt = lyra_core::pretty::format_value(&bfs_res);
    assert!(bfs_txt.contains("\"order\""));
    // ShortestPaths unweighted
    let sp = ev.eval(Value::expr(
        Value::Symbol("ShortestPaths".into()),
        vec![g.clone(), Value::String("A".into())],
    ));
    let sp_txt = lyra_core::pretty::format_value(&sp);
    assert!(sp_txt.contains("\"C\" -> 1") || sp_txt.contains("\"C\" -> 2"));
    // Topological sort
    let topo = ev.eval(Value::expr(Value::Symbol("TopologicalSort".into()), vec![g.clone()]));
    assert!(!matches!(topo, Value::Symbol(ref s) if s=="Null"));
    // WCC on a disconnected graph: add isolated E
    let _ = ev.eval(Value::expr(
        Value::Symbol("AddNodes".into()),
        vec![g.clone(), Value::List(vec![Value::String("E".into())])],
    ));
    let _wcc = ev.eval(Value::expr(Value::Symbol("ConnectedComponents".into()), vec![g.clone()]));
    // SCC on a cycle: add edge D->A to form a cycle
    let _ = ev.eval(Value::expr(
        Value::Symbol("AddEdges".into()),
        vec![
            g.clone(),
            Value::Assoc(
                [
                    ("Src".into(), Value::String("D".into())),
                    ("Dst".into(), Value::String("A".into())),
                ]
                .into_iter()
                .collect(),
            ),
        ],
    ));
    let scc =
        ev.eval(Value::expr(Value::Symbol("StronglyConnectedComponents".into()), vec![g.clone()]));
    let scc_txt = lyra_core::pretty::format_value(&scc);
    assert!(scc_txt.contains("A") && scc_txt.contains("D"));
    // PageRank sum approx 1
    let pr = ev.eval(Value::expr(Value::Symbol("PageRank".into()), vec![g.clone()]));
    if let Value::Assoc(m) = pr {
        let sum: f64 = m.values().map(|v| if let Value::Real(f) = v { *f } else { 0.0 }).sum();
        assert!((sum - 1.0).abs() < 1e-3);
    } else {
        panic!("pagerank assoc");
    }
    // Degree centrality returns all nodes
    let dc = ev.eval(Value::expr(Value::Symbol("DegreeCentrality".into()), vec![g.clone()]));
    if let Value::Assoc(m) = dc {
        assert_eq!(m.len(), 5);
    } else {
        panic!("deg cent");
    }
}

#[test]
fn graphs_mst_and_maxflow() {
    let mut ev = Evaluator::new();
    stdlib::register_all(&mut ev);
    // Undirected triangle with weights
    let g = ev.eval(Value::expr(
        Value::Symbol("GraphCreate".into()),
        vec![Value::Assoc(
            [
                ("Directed".to_string(), Value::Boolean(false)),
                ("Multi".to_string(), Value::Boolean(false)),
            ]
            .into_iter()
            .collect(),
        )],
    ));
    let _ = ev.eval(Value::expr(
        Value::Symbol("AddNodes".into()),
        vec![
            g.clone(),
            Value::List(vec!["A", "B", "C"].into_iter().map(|s| Value::String(s.into())).collect()),
        ],
    ));
    let e = |s: &str, d: &str, w: f64| {
        Value::Assoc(
            [
                ("Src".into(), Value::String(s.into())),
                ("Dst".into(), Value::String(d.into())),
                ("Weight".into(), Value::Real(w)),
            ]
            .into_iter()
            .collect(),
        )
    };
    let _ = ev.eval(Value::expr(
        Value::Symbol("AddEdges".into()),
        vec![g.clone(), Value::List(vec![e("A", "B", 1.0), e("B", "C", 2.0), e("A", "C", 3.0)])],
    ));
    let mst = ev.eval(Value::expr(Value::Symbol("MinimumSpanningTree".into()), vec![g.clone()]));
    let txt = lyra_core::pretty::format_value(&mst);
    assert!(txt.contains("\"id\""));

    // Simple directed flow: s->a (3), s->b(2), a->t(2), b->t(4)
    let g2 = ev.eval(Value::expr(
        Value::Symbol("GraphCreate".into()),
        vec![Value::Assoc([("Directed".into(), Value::Boolean(true))].into_iter().collect())],
    ));
    let _ = ev.eval(Value::expr(
        Value::Symbol("AddNodes".into()),
        vec![
            g2.clone(),
            Value::List(
                vec!["s", "a", "b", "t"].into_iter().map(|s| Value::String(s.into())).collect(),
            ),
        ],
    ));
    let e2 = |s: &str, d: &str, w: f64| {
        Value::Assoc(
            [
                ("Src".into(), Value::String(s.into())),
                ("Dst".into(), Value::String(d.into())),
                ("Weight".into(), Value::Real(w)),
            ]
            .into_iter()
            .collect(),
        )
    };
    let _ = ev.eval(Value::expr(
        Value::Symbol("AddEdges".into()),
        vec![
            g2.clone(),
            Value::List(vec![
                e2("s", "a", 3.0),
                e2("s", "b", 2.0),
                e2("a", "t", 2.0),
                e2("b", "t", 4.0),
            ]),
        ],
    ));
    let mf = ev.eval(Value::expr(
        Value::Symbol("MaxFlow".into()),
        vec![g2.clone(), Value::String("s".into()), Value::String("t".into())],
    ));
    assert!(matches!(mf, Value::Real(f) if (f-4.0).abs() < 1e-6));
}
