bitflags::bitflags! {
    #[derive(Clone, Copy)]
    pub struct Attributes: u32 {
        const LISTABLE = 0b0001;
        const FLAT     = 0b0010;
        const ORDERLESS= 0b0100;
        const HOLD_ALL = 0b1000;
        const HOLD_FIRST = 0b1_0000;
        const HOLD_REST  = 0b10_0000;
        const ONE_IDENTITY = 0b100_0000;
        // Future: NUMERIC_FUNCTION, etc.
    }
}

impl Default for Attributes {
    fn default() -> Self { Attributes::empty() }
}
