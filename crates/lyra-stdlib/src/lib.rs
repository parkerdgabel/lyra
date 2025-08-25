//! Lyra Standard Library registration helpers.

use lyra_runtime::Evaluator;

#[cfg(feature = "math")] pub mod math;
#[cfg(feature = "algebra")] pub mod algebra;
#[cfg(feature = "logic")] pub mod logic;
#[cfg(feature = "tools")] #[macro_use] pub mod tools;
#[cfg(feature = "list")] pub mod list;
#[cfg(feature = "string")] pub mod string;
#[cfg(feature = "assoc")] pub mod assoc;
#[cfg(feature = "concurrency")] pub mod concurrency;
#[cfg(feature = "schema")] pub mod schema;
#[cfg(feature = "explain")] pub mod explain;
#[cfg(feature = "testing")] pub mod testing;
#[cfg(feature = "io")] pub mod io;
#[cfg(feature = "net")] pub mod net;
#[cfg(feature = "dataset")] pub mod dataset;
#[cfg(feature = "db")] pub mod db;
#[cfg(feature = "containers")] pub mod containers;
#[cfg(feature = "graphs")] pub mod graphs;
#[cfg(feature = "crypto")] pub mod crypto;
#[cfg(feature = "image")] pub mod image;
#[cfg(feature = "audio")] pub mod audio;
#[cfg(feature = "media")] pub mod media;
#[cfg(feature = "text")] pub mod text;
#[cfg(feature = "text_fuzzy")] pub mod text_fuzzy;
#[cfg(feature = "text_index")] pub mod text_index;
#[cfg(feature = "collections")] pub mod collections;
#[cfg(feature = "ndarray")] pub mod ndarray;
mod dispatch;

pub fn register_all(ev: &mut Evaluator) {
    // Core forms from the runtime (assignment, replacement, threading)
    #[cfg(feature = "core")] lyra_runtime::eval::register_core(ev);
    // Introspection helpers for tool discovery
    lyra_runtime::eval::register_introspection(ev);
    #[cfg(feature = "string")] string::register_string(ev);
    #[cfg(feature = "math")] math::register_math(ev);
    #[cfg(feature = "algebra")] algebra::register_algebra(ev);
    #[cfg(feature = "list")] list::register_list(ev);
    #[cfg(feature = "tools")] tools::register_tools(ev);
    #[cfg(feature = "assoc")] assoc::register_assoc(ev);
    #[cfg(feature = "logic")] logic::register_logic(ev);
    #[cfg(feature = "concurrency")] concurrency::register_concurrency(ev);
    #[cfg(feature = "schema")] schema::register_schema(ev);
    #[cfg(feature = "explain")] explain::register_explain(ev);
    #[cfg(feature = "io")] io::register_io(ev);
    #[cfg(feature = "net")] net::register_net(ev);
    #[cfg(feature = "dataset")] dataset::register_dataset(ev);
    #[cfg(feature = "db")] db::register_db(ev);
    #[cfg(feature = "containers")] containers::register_containers(ev);
    #[cfg(feature = "graphs")] graphs::register_graphs(ev);
    #[cfg(feature = "crypto")] crypto::register_crypto(ev);
    #[cfg(feature = "image")] image::register_image(ev);
    #[cfg(feature = "audio")] audio::register_audio(ev);
    #[cfg(feature = "media")] media::register_media(ev);
    #[cfg(feature = "text")] text::register_text(ev);
    #[cfg(feature = "text_fuzzy")] text_fuzzy::register_text_fuzzy(ev);
    #[cfg(feature = "text_index")] text_index::register_text_index(ev);
    #[cfg(feature = "collections")] collections::register_collections(ev);
    #[cfg(feature = "ndarray")] ndarray::register_ndarray(ev);
    #[cfg(feature = "testing")] testing::register_testing(ev);
    // Register dispatchers last to resolve name conflicts (Join, etc.)
    dispatch::register_dispatch(ev);
}

pub fn register_with(ev: &mut Evaluator, groups: &[&str]) {
    for g in groups {
        match *g {
            "string" => { #[cfg(feature = "string")] string::register_string(ev) }
            "math" => { #[cfg(feature = "math")] math::register_math(ev) }
            "algebra" => { #[cfg(feature = "algebra")] algebra::register_algebra(ev) }
            "list" => { #[cfg(feature = "list")] list::register_list(ev) }
            "tools" => { #[cfg(feature = "tools")] tools::register_tools(ev) }
            "assoc" => { #[cfg(feature = "assoc")] assoc::register_assoc(ev) }
            "logic" => { #[cfg(feature = "logic")] logic::register_logic(ev) }
            "concurrency" => { #[cfg(feature = "concurrency")] concurrency::register_concurrency(ev) }
            "schema" => { #[cfg(feature = "schema")] schema::register_schema(ev) }
            "explain" => { #[cfg(feature = "explain")] explain::register_explain(ev) }
            "io" => { #[cfg(feature = "io")] io::register_io(ev) }
            "net" => { #[cfg(feature = "net")] net::register_net(ev) }
            "dataset" => { #[cfg(feature = "dataset")] dataset::register_dataset(ev) }
            "db" => { #[cfg(feature = "db")] db::register_db(ev) }
            "containers" => { #[cfg(feature = "containers")] containers::register_containers(ev) }
            "graphs" => { #[cfg(feature = "graphs")] graphs::register_graphs(ev) }
            "crypto" => { #[cfg(feature = "crypto")] crypto::register_crypto(ev) }
            "image" => { #[cfg(feature = "image")] image::register_image(ev) }
            "audio" => { #[cfg(feature = "audio")] audio::register_audio(ev) }
            "media" => { #[cfg(feature = "media")] media::register_media(ev) }
            "text" => { #[cfg(feature = "text")] text::register_text(ev) }
            "text_fuzzy" => { #[cfg(feature = "text_fuzzy")] text_fuzzy::register_text_fuzzy(ev) }
            "text_index" => { #[cfg(feature = "text_index")] text_index::register_text_index(ev) }
            "collections" => { #[cfg(feature = "collections")] collections::register_collections(ev) }
            _ => {}
        }
    }
}
