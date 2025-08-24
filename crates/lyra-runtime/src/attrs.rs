bitflags::bitflags! {
    #[derive(Clone, Copy)]
    pub struct Attributes: u32 {
        const LISTABLE = 0b0001;
        const FLAT     = 0b0010;
        const ORDERLESS= 0b0100;
        const HOLD_ALL = 0b1000;
        // Future: HOLD_FIRST, HOLD_REST, FLAT, ORDERLESS, ONE_IDENTITY, NUMERIC_FUNCTION
    }
}

impl Default for Attributes {
    fn default() -> Self { Attributes::empty() }
}
