use lyra_notebook_core as nbcore;

#[test]
fn create_and_serialize_roundtrip() {
    let nb = nbcore::notebook_create(nbcore::ops::NotebookCreateOpts {
        title: Some("Test Notebook".into()),
        ..Default::default()
    });
    let cell1 = nbcore::cell_create(nbcore::schema::CellType::Code, "1+1", Default::default());
    let nb = nbcore::cell_insert(&nb, cell1, nbcore::ops::InsertPos::Index(0));
    let rep = nbcore::validate_notebook(&nb);
    assert!(rep.valid, "validation errors: {:?}", rep.errors);

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.lynb");
    nbcore::write_notebook(&path, &nb, Default::default()).unwrap();
    let nb2 = nbcore::read_notebook(&path).unwrap();
    assert_eq!(nb.version, nb2.version);
    assert_eq!(nb.cells.len(), nb2.cells.len());
}

#[test]
fn insert_move_delete_cells() {
    let nb = nbcore::notebook_create(Default::default());
    let c1 = nbcore::cell_create(nbcore::schema::CellType::Code, "a=1", Default::default());
    let c2 = nbcore::cell_create(nbcore::schema::CellType::Code, "a+1", Default::default());
    let nb = nbcore::cell_insert(&nb, c1.clone(), nbcore::ops::InsertPos::Index(0));
    let nb = nbcore::cell_insert(&nb, c2.clone(), nbcore::ops::InsertPos::After(c1.id));
    assert_eq!(nb.cells.len(), 2);
    assert_eq!(nb.cells[0].id, c1.id);
    assert_eq!(nb.cells[1].id, c2.id);

    let nb = nbcore::cell_move(&nb, c2.id, 0);
    assert_eq!(nb.cells[0].id, c2.id);

    let nb = nbcore::cell_delete(&nb, c2.id);
    assert_eq!(nb.cells.len(), 1);
}
