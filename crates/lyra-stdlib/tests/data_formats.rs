use lyra_core::value::Value;
use lyra_runtime::Evaluator;
use lyra_stdlib as stdlib;

fn ev_all() -> Evaluator {
    let mut e = Evaluator::new();
    stdlib::register_all(&mut e);
    e
}

#[test]
fn json_yaml_toml_roundtrips() {
    let mut ev = ev_all();
    // JSON aliases
    let v = ev.eval(Value::Expr {
        head: Box::new(Value::Symbol("JsonParse".into())),
        args: vec![Value::String("{\"a\":1,\"b\":[2,3]}".into())],
    });
    let s = ev.eval(Value::Expr {
        head: Box::new(Value::Symbol("JsonStringify".into())),
        args: vec![v.clone()],
    });
    assert!(matches!(s, Value::String(_)));

    // YAML
    let y = "a: 1\nb:\n  - 2\n  - 3\n";
    let v2 = ev.eval(Value::Expr {
        head: Box::new(Value::Symbol("YamlParse".into())),
        args: vec![Value::String(y.into())],
    });
    let y2 = ev.eval(Value::Expr {
        head: Box::new(Value::Symbol("YamlStringify".into())),
        args: vec![v2],
    });
    assert!(matches!(y2, Value::String(_)));

    // TOML
    let t = "a = 1\n[[b]]\nx = 2\n[[b]]\nx = 3\n";
    let vt = ev.eval(Value::Expr {
        head: Box::new(Value::Symbol("TomlParse".into())),
        args: vec![Value::String(t.into())],
    });
    let t2 = ev.eval(Value::Expr {
        head: Box::new(Value::Symbol("TomlStringify".into())),
        args: vec![vt],
    });
    assert!(matches!(t2, Value::String(_)));
}

#[test]
fn bytes_and_encoding() {
    let mut ev = ev_all();
    // TextEncode/Decode
    let b = ev.eval(Value::Expr {
        head: Box::new(Value::Symbol("TextEncode".into())),
        args: vec![Value::String("hello".into())],
    });
    match &b {
        Value::List(items) => assert_eq!(items.len(), 5),
        _ => panic!("expected bytes list"),
    }
    let s =
        ev.eval(Value::Expr { head: Box::new(Value::Symbol("TextDecode".into())), args: vec![b] });
    assert_eq!(s, Value::String("hello".into()));

    // Base64
    let enc = ev.eval(Value::Expr {
        head: Box::new(Value::Symbol("Base64Encode".into())),
        args: vec![Value::List(
            vec![104, 101, 108, 108, 111].into_iter().map(|x| Value::Integer(x)).collect(),
        )],
    });
    let dec = ev.eval(Value::Expr {
        head: Box::new(Value::Symbol("Base64Decode".into())),
        args: vec![enc],
    });
    assert_eq!(
        dec,
        Value::List(vec![104, 101, 108, 108, 111].into_iter().map(|x| Value::Integer(x)).collect())
    );

    // Hex
    let hex = ev.eval(Value::Expr {
        head: Box::new(Value::Symbol("HexEncode".into())),
        args: vec![Value::List(
            vec![0xde, 0xad, 0xbe, 0xef].into_iter().map(|x| Value::Integer(x)).collect(),
        )],
    });
    assert_eq!(hex, Value::String("deadbeef".into()));
    let back =
        ev.eval(Value::Expr { head: Box::new(Value::Symbol("HexDecode".into())), args: vec![hex] });
    assert_eq!(
        back,
        Value::List(vec![222, 173, 190, 239].into_iter().map(|x| Value::Integer(x)).collect())
    );

    // Concat/Slice/Length
    let c = ev.eval(Value::Expr {
        head: Box::new(Value::Symbol("BytesConcat".into())),
        args: vec![Value::List(vec![
            Value::List(vec![1, 2, 3].into_iter().map(|x| Value::Integer(x)).collect()),
            Value::List(vec![4, 5].into_iter().map(|x| Value::Integer(x)).collect()),
        ])],
    });
    assert_eq!(
        c,
        Value::List(vec![1, 2, 3, 4, 5].into_iter().map(|x| Value::Integer(x)).collect())
    );
    let sl = ev.eval(Value::Expr {
        head: Box::new(Value::Symbol("BytesSlice".into())),
        args: vec![c.clone(), Value::Integer(1), Value::Integer(4)],
    });
    assert_eq!(sl, Value::List(vec![2, 3, 4].into_iter().map(|x| Value::Integer(x)).collect()));
    let ln =
        ev.eval(Value::Expr { head: Box::new(Value::Symbol("BytesLength".into())), args: vec![c] });
    assert_eq!(ln, Value::Integer(5));
}

#[test]
fn uuids() {
    let mut ev = ev_all();
    let u4 = ev.eval(Value::Expr { head: Box::new(Value::Symbol("UuidV4".into())), args: vec![] });
    let u7 = ev.eval(Value::Expr { head: Box::new(Value::Symbol("UuidV7".into())), args: vec![] });
    match (u4, u7) {
        (Value::String(a), Value::String(b)) => {
            assert!(a.len() >= 30);
            assert!(b.len() >= 30);
        }
        _ => panic!("expected strings"),
    }
}
