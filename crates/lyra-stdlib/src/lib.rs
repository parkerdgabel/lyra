//! Lyra Standard Library registration helpers.

use lyra_runtime::Evaluator;
use lyra_runtime::attrs::Attributes;

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
#[cfg(feature = "ml")] pub mod ml;
#[cfg(feature = "nn")] pub mod nn;
#[cfg(feature = "functional")] pub mod functional;
mod dispatch;

// Conditional registration helper used by filtered registrars
pub fn register_if(
    ev: &mut Evaluator,
    filter: &dyn Fn(&str) -> bool,
    name: &str,
    f: fn(&mut Evaluator, Vec<lyra_core::value::Value>) -> lyra_core::value::Value,
    attrs: Attributes,
) {
    if filter(name) { ev.register(name, f, attrs); }
}

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
    #[cfg(feature = "ml")] ml::register_ml(ev);
    #[cfg(feature = "nn")] nn::register_nn(ev);
    #[cfg(feature = "functional")] functional::register_functional(ev);
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
            "ml" => { #[cfg(feature = "ml")] ml::register_ml(ev) }
            "nn" => { #[cfg(feature = "nn")] nn::register_nn(ev) }
            "functional" => { #[cfg(feature = "functional")] functional::register_functional(ev) }
            _ => {}
        }
    }
}

// Register only selected symbol names across modules. Always includes core + introspection.
pub fn register_selected(ev: &mut Evaluator, names: &std::collections::HashSet<&str>) {
    #[cfg(feature = "core")] lyra_runtime::eval::register_core(ev);
    lyra_runtime::eval::register_introspection(ev);
    let predicate = |n: &str| names.contains(n);

    #[cfg(feature = "string")] crate::string::register_string_filtered(ev, &predicate);
    #[cfg(feature = "math")] crate::math::register_math_filtered(ev, &predicate);
    #[cfg(feature = "algebra")] crate::algebra::register_algebra_filtered(ev, &predicate);
    #[cfg(feature = "list")] crate::list::register_list_filtered(ev, &predicate);
    #[cfg(feature = "tools")] crate::tools::register_tools_filtered(ev, &predicate);
    #[cfg(feature = "assoc")] crate::assoc::register_assoc_filtered(ev, &predicate);
    #[cfg(feature = "logic")] crate::logic::register_logic_filtered(ev, &predicate);
    #[cfg(feature = "schema")] crate::schema::register_schema_filtered(ev, &predicate);
    #[cfg(feature = "explain")] crate::explain::register_explain_filtered(ev, &predicate);
    #[cfg(feature = "io")] crate::io::register_io_filtered(ev, &predicate);
    #[cfg(feature = "net")] crate::net::register_net_filtered(ev, &predicate);
    #[cfg(feature = "dataset")] crate::dataset::register_dataset_filtered(ev, &predicate);
    #[cfg(feature = "db")] crate::db::register_db_filtered(ev, &predicate);
    #[cfg(feature = "containers")] crate::containers::register_containers_filtered(ev, &predicate);
    #[cfg(feature = "graphs")] crate::graphs::register_graphs_filtered(ev, &predicate);
    #[cfg(feature = "crypto")] crate::crypto::register_crypto_filtered(ev, &predicate);
    #[cfg(feature = "image")] crate::image::register_image_filtered(ev, &predicate);
    #[cfg(feature = "audio")] crate::audio::register_audio_filtered(ev, &predicate);
    #[cfg(feature = "media")] crate::media::register_media_filtered(ev, &predicate);
    #[cfg(feature = "text")] crate::text::register_text_filtered(ev, &predicate);
    #[cfg(feature = "text_fuzzy")] crate::text_fuzzy::register_text_fuzzy_filtered(ev, &predicate);
    #[cfg(feature = "text_index")] crate::text_index::register_text_index_filtered(ev, &predicate);
    #[cfg(feature = "collections")] crate::collections::register_collections_filtered(ev, &predicate);
    #[cfg(feature = "ndarray")] crate::ndarray::register_ndarray_filtered(ev, &predicate);
    #[cfg(feature = "ml")] crate::ml::register_ml_filtered(ev, &predicate);
    #[cfg(feature = "nn")] crate::nn::register_nn_filtered(ev, &predicate);
    #[cfg(feature = "functional")] crate::functional::register_functional_filtered(ev, &predicate);
    #[cfg(feature = "testing")] crate::testing::register_testing_filtered(ev, &predicate);

    #[cfg(feature = "math")] crate::math::register_math_filtered(ev, &predicate);
    #[cfg(feature = "list")] crate::list::register_list_filtered(ev, &predicate);
    #[cfg(feature = "assoc")] crate::assoc::register_assoc_filtered(ev, &predicate);
    #[cfg(feature = "functional")] crate::functional::register_functional_filtered(ev, &predicate);
    // Dispatchers last
    crate::dispatch::register_dispatch(ev);
}
