use uuid::Uuid;

pub fn new_id_v7() -> Uuid {
    Uuid::now_v7()
}
