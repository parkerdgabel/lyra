//! Lyra Standard Library registration helpers.

use lyra_runtime::attrs::Attributes;
use lyra_runtime::Evaluator;

#[cfg(feature = "algebra")]
pub mod algebra;
#[cfg(feature = "logic")]
pub mod logic;
#[cfg(feature = "math")]
pub mod math;
#[cfg(feature = "tools")]
#[macro_use]
pub mod tools;
#[cfg(feature = "assoc")]
pub mod assoc;
#[cfg(feature = "audio")]
pub mod audio;
#[cfg(feature = "collections")]
pub mod collections;
#[cfg(feature = "concurrency")]
pub mod concurrency;
#[cfg(feature = "containers")]
pub mod containers;
#[cfg(feature = "crypto")]
pub mod crypto;
#[cfg(feature = "dataset")]
pub mod dataset;
#[cfg(feature = "frame")]
pub mod frame;
#[cfg(feature = "db")]
pub mod db;
#[cfg(feature = "dev")]
pub mod dev;
mod dispatch;
mod docs;
#[cfg(feature = "explain")]
pub mod explain;
#[cfg(feature = "fs")]
pub mod fs;
#[cfg(feature = "functional")]
pub mod functional;
#[cfg(feature = "git")]
pub mod git;
#[cfg(feature = "graphs")]
pub mod graphs;
#[cfg(feature = "image")]
pub mod image;
#[cfg(feature = "io")]
pub mod io;
#[cfg(feature = "list")]
pub mod list;
#[cfg(feature = "logging")]
pub mod logging;
#[cfg(feature = "media")]
pub mod media;
#[cfg(feature = "memory")]
pub mod memory;
#[cfg(feature = "metrics")]
pub mod metrics;
#[cfg(feature = "ml")]
pub mod ml;
#[cfg(feature = "model")]
pub mod model;
#[cfg(feature = "module")]
pub mod module;
#[cfg(feature = "ndarray")]
pub mod ndarray;
#[cfg(feature = "net")]
pub mod net;
#[cfg(feature = "nn")]
pub mod nn;
#[cfg(feature = "package")]
pub mod package;
#[cfg(feature = "policy")]
pub mod policy;
#[cfg(feature = "process")]
pub mod process;
#[cfg(feature = "project")]
pub mod project;
#[cfg(feature = "rag")]
pub mod rag;
#[cfg(feature = "schema")]
pub mod schema;
#[cfg(feature = "string")]
pub mod string;
#[cfg(feature = "testing")]
pub mod testing;
#[cfg(feature = "text")]
pub mod text;
#[cfg(feature = "text_fuzzy")]
pub mod text_fuzzy;
#[cfg(feature = "text_index")]
pub mod text_index;
#[cfg(feature = "time")]
pub mod time;
#[cfg(feature = "trace")]
pub mod trace;
#[cfg(feature = "vector")]
pub mod vector;
#[cfg(feature = "visual")]
pub mod visual;
#[cfg(feature = "workflow")]
pub mod workflow;

// Conditional registration helper used by filtered registrars
pub fn register_if(
    ev: &mut Evaluator,
    filter: &dyn Fn(&str) -> bool,
    name: &str,
    f: fn(&mut Evaluator, Vec<lyra_core::value::Value>) -> lyra_core::value::Value,
    attrs: Attributes,
) {
    if filter(name) {
        ev.register(name, f, attrs);
    }
}

pub fn register_all(ev: &mut Evaluator) {
    // Core forms from the runtime (assignment, replacement, threading)
    #[cfg(feature = "core")]
    lyra_runtime::eval::register_core(ev);
    // Introspection helpers for tool discovery
    lyra_runtime::eval::register_introspection(ev);
    #[cfg(feature = "string")]
    string::register_string(ev);
    #[cfg(feature = "math")]
    math::register_math(ev);
    #[cfg(feature = "algebra")]
    algebra::register_algebra(ev);
    #[cfg(feature = "list")]
    list::register_list(ev);
    #[cfg(feature = "tools")]
    tools::register_tools(ev);
    #[cfg(feature = "assoc")]
    assoc::register_assoc(ev);
    #[cfg(feature = "logic")]
    logic::register_logic(ev);
    #[cfg(feature = "concurrency")]
    concurrency::register_concurrency(ev);
    #[cfg(feature = "schema")]
    schema::register_schema(ev);
    #[cfg(feature = "explain")]
    explain::register_explain(ev);
    #[cfg(feature = "io")]
    io::register_io(ev);
    #[cfg(feature = "model")]
    model::register_model(ev);
    #[cfg(feature = "trace")]
    trace::register_trace(ev);
    #[cfg(feature = "metrics")]
    metrics::register_metrics(ev);
    #[cfg(feature = "memory")]
    memory::register_memory(ev);
    #[cfg(feature = "policy")]
    policy::register_policy(ev);
    #[cfg(feature = "dev")]
    dev::register_dev(ev);
    #[cfg(feature = "workflow")]
    workflow::register_workflow(ev);
    #[cfg(feature = "vector")]
    vector::register_vector(ev);
    #[cfg(feature = "rag")]
    rag::register_rag(ev);
    #[cfg(feature = "net")]
    net::register_net(ev);
    #[cfg(feature = "time")]
    time::register_time(ev);
    #[cfg(feature = "logging")]
    logging::register_logging(ev);
    #[cfg(feature = "process")]
    process::register_process(ev);
    #[cfg(feature = "git")]
    git::register_git(ev);
    #[cfg(feature = "fs")]
    fs::register_fs(ev);
    #[cfg(feature = "dataset")]
    dataset::register_dataset(ev);
    #[cfg(feature = "frame")]
    frame::register_frame(ev);
    #[cfg(feature = "db")]
    db::register_db(ev);
    #[cfg(feature = "containers")]
    containers::register_containers(ev);
    #[cfg(feature = "graphs")]
    graphs::register_graphs(ev);
    #[cfg(feature = "crypto")]
    crypto::register_crypto(ev);
    #[cfg(feature = "image")]
    image::register_image(ev);
    #[cfg(feature = "visual")]
    visual::register_visual(ev);
    #[cfg(feature = "audio")]
    audio::register_audio(ev);
    #[cfg(feature = "media")]
    media::register_media(ev);
    #[cfg(feature = "text")]
    text::register_text(ev);
    #[cfg(feature = "text_fuzzy")]
    text_fuzzy::register_text_fuzzy(ev);
    #[cfg(feature = "text_index")]
    text_index::register_text_index(ev);
    #[cfg(feature = "collections")]
    collections::register_collections(ev);
    #[cfg(feature = "ndarray")]
    ndarray::register_ndarray(ev);
    #[cfg(feature = "ml")]
    ml::register_ml(ev);
    #[cfg(feature = "nn")]
    nn::register_nn(ev);
    #[cfg(feature = "functional")]
    functional::register_functional(ev);
    #[cfg(feature = "package")]
    package::register_package(ev);
    #[cfg(feature = "module")]
    module::register_module(ev);
    #[cfg(feature = "project")]
    project::register_project(ev);
    #[cfg(feature = "testing")]
    testing::register_testing(ev);
    // Register dispatchers last to resolve name conflicts (Join, etc.)
    dispatch::register_dispatch(ev);
    // Seed human-facing docs last so DescribeBuiltins can expose summaries/params
    docs::register_docs(ev);
    docs::register_docs_extra(ev);
    // Auto-populate from ToolsDescribe specs provided by modules
    docs::autoseed_from_tools(ev);
}

pub fn register_with(ev: &mut Evaluator, groups: &[&str]) {
    for g in groups {
        match *g {
            "string" =>
            {
                #[cfg(feature = "string")]
                string::register_string(ev)
            }
            "math" =>
            {
                #[cfg(feature = "math")]
                math::register_math(ev)
            }
            "algebra" =>
            {
                #[cfg(feature = "algebra")]
                algebra::register_algebra(ev)
            }
            "list" =>
            {
                #[cfg(feature = "list")]
                list::register_list(ev)
            }
            "tools" =>
            {
                #[cfg(feature = "tools")]
                tools::register_tools(ev)
            }
            "assoc" =>
            {
                #[cfg(feature = "assoc")]
                assoc::register_assoc(ev)
            }
            "logic" =>
            {
                #[cfg(feature = "logic")]
                logic::register_logic(ev)
            }
            "concurrency" =>
            {
                #[cfg(feature = "concurrency")]
                concurrency::register_concurrency(ev)
            }
            "schema" =>
            {
                #[cfg(feature = "schema")]
                schema::register_schema(ev)
            }
            "explain" =>
            {
                #[cfg(feature = "explain")]
                explain::register_explain(ev)
            }
            "io" =>
            {
                #[cfg(feature = "io")]
                io::register_io(ev)
            }
            "net" =>
            {
                #[cfg(feature = "net")]
                net::register_net(ev)
            }
            "time" =>
            {
                #[cfg(feature = "time")]
                time::register_time(ev)
            }
            "logging" =>
            {
                #[cfg(feature = "logging")]
                logging::register_logging(ev)
            }
            "process" =>
            {
                #[cfg(feature = "process")]
                process::register_process(ev)
            }
            "git" =>
            {
                #[cfg(feature = "git")]
                git::register_git(ev)
            }
            "fs" =>
            {
                #[cfg(feature = "fs")]
                fs::register_fs(ev)
            }
            "dataset" =>
            {
                #[cfg(feature = "dataset")]
                dataset::register_dataset(ev)
            }
            "frame" =>
            {
                #[cfg(feature = "frame")]
                frame::register_frame(ev)
            }
            "db" =>
            {
                #[cfg(feature = "db")]
                db::register_db(ev)
            }
            "containers" =>
            {
                #[cfg(feature = "containers")]
                containers::register_containers(ev)
            }
            "graphs" =>
            {
                #[cfg(feature = "graphs")]
                graphs::register_graphs(ev)
            }
            "crypto" =>
            {
                #[cfg(feature = "crypto")]
                crypto::register_crypto(ev)
            }
            "image" =>
            {
                #[cfg(feature = "image")]
                image::register_image(ev)
            }
            "visual" =>
            {
                #[cfg(feature = "visual")]
                visual::register_visual(ev)
            }
            "audio" =>
            {
                #[cfg(feature = "audio")]
                audio::register_audio(ev)
            }
            "media" =>
            {
                #[cfg(feature = "media")]
                media::register_media(ev)
            }
            "text" =>
            {
                #[cfg(feature = "text")]
                text::register_text(ev)
            }
            "text_fuzzy" =>
            {
                #[cfg(feature = "text_fuzzy")]
                text_fuzzy::register_text_fuzzy(ev)
            }
            "text_index" =>
            {
                #[cfg(feature = "text_index")]
                text_index::register_text_index(ev)
            }
            "collections" =>
            {
                #[cfg(feature = "collections")]
                collections::register_collections(ev)
            }
            "ml" =>
            {
                #[cfg(feature = "ml")]
                ml::register_ml(ev)
            }
            "nn" =>
            {
                #[cfg(feature = "nn")]
                nn::register_nn(ev)
            }
            "functional" =>
            {
                #[cfg(feature = "functional")]
                functional::register_functional(ev)
            }
            "package" =>
            {
                #[cfg(feature = "package")]
                package::register_package(ev)
            }
            "module" =>
            {
                #[cfg(feature = "module")]
                module::register_module(ev)
            }
            "project" =>
            {
                #[cfg(feature = "project")]
                project::register_project(ev)
            }
            "dev" =>
            {
                #[cfg(feature = "dev")]
                dev::register_dev(ev)
            }
            _ => {}
        }
    }
}

// Register only selected symbol names across modules. Always includes core + introspection.
pub fn register_selected(ev: &mut Evaluator, names: &std::collections::HashSet<&str>) {
    #[cfg(feature = "core")]
    lyra_runtime::eval::register_core(ev);
    lyra_runtime::eval::register_introspection(ev);
    let predicate = |n: &str| names.contains(n);

    #[cfg(feature = "string")]
    crate::string::register_string_filtered(ev, &predicate);
    #[cfg(feature = "math")]
    crate::math::register_math_filtered(ev, &predicate);
    #[cfg(feature = "algebra")]
    crate::algebra::register_algebra_filtered(ev, &predicate);
    #[cfg(feature = "list")]
    crate::list::register_list_filtered(ev, &predicate);
    #[cfg(feature = "tools")]
    crate::tools::register_tools_filtered(ev, &predicate);
    #[cfg(feature = "assoc")]
    crate::assoc::register_assoc_filtered(ev, &predicate);
    #[cfg(feature = "logic")]
    crate::logic::register_logic_filtered(ev, &predicate);
    #[cfg(feature = "concurrency")]
    crate::concurrency::register_concurrency_filtered(ev, &predicate);
    #[cfg(feature = "schema")]
    crate::schema::register_schema_filtered(ev, &predicate);
    #[cfg(feature = "explain")]
    crate::explain::register_explain_filtered(ev, &predicate);
    #[cfg(feature = "io")]
    crate::io::register_io_filtered(ev, &predicate);
    #[cfg(feature = "model")]
    crate::model::register_model_filtered(ev, &predicate);
    #[cfg(feature = "trace")]
    crate::trace::register_trace_filtered(ev, &predicate);
    #[cfg(feature = "metrics")]
    crate::metrics::register_metrics_filtered(ev, &predicate);
    #[cfg(feature = "memory")]
    crate::memory::register_memory_filtered(ev, &predicate);
    #[cfg(feature = "policy")]
    crate::policy::register_policy_filtered(ev, &predicate);
    #[cfg(feature = "workflow")]
    crate::workflow::register_workflow_filtered(ev, &predicate);
    #[cfg(feature = "vector")]
    crate::vector::register_vector_filtered(ev, &predicate);
    #[cfg(feature = "rag")]
    crate::rag::register_rag_filtered(ev, &predicate);
    #[cfg(feature = "net")]
    crate::net::register_net_filtered(ev, &predicate);
    #[cfg(feature = "time")]
    crate::time::register_time_filtered(ev, &predicate);
    #[cfg(feature = "logging")]
    crate::logging::register_logging_filtered(ev, &predicate);
    #[cfg(feature = "process")]
    crate::process::register_process_filtered(ev, &predicate);
    #[cfg(feature = "git")]
    crate::git::register_git_filtered(ev, &predicate);
    #[cfg(feature = "fs")]
    crate::fs::register_fs_filtered(ev, &predicate);
    #[cfg(feature = "dataset")]
    crate::dataset::register_dataset_filtered(ev, &predicate);
    #[cfg(feature = "frame")]
    crate::frame::register_frame_filtered(ev, &predicate);
    #[cfg(feature = "db")]
    crate::db::register_db_filtered(ev, &predicate);
    #[cfg(feature = "containers")]
    crate::containers::register_containers_filtered(ev, &predicate);
    #[cfg(feature = "graphs")]
    crate::graphs::register_graphs_filtered(ev, &predicate);
    #[cfg(feature = "crypto")]
    crate::crypto::register_crypto_filtered(ev, &predicate);
    #[cfg(feature = "image")]
    crate::image::register_image_filtered(ev, &predicate);
    #[cfg(feature = "visual")]
    crate::visual::register_visual_filtered(ev, &predicate);
    #[cfg(feature = "audio")]
    crate::audio::register_audio_filtered(ev, &predicate);
    #[cfg(feature = "media")]
    crate::media::register_media_filtered(ev, &predicate);
    #[cfg(feature = "text")]
    crate::text::register_text_filtered(ev, &predicate);
    #[cfg(feature = "text_fuzzy")]
    crate::text_fuzzy::register_text_fuzzy_filtered(ev, &predicate);
    #[cfg(feature = "text_index")]
    crate::text_index::register_text_index_filtered(ev, &predicate);
    #[cfg(feature = "collections")]
    crate::collections::register_collections_filtered(ev, &predicate);
    #[cfg(feature = "ndarray")]
    crate::ndarray::register_ndarray_filtered(ev, &predicate);
    #[cfg(feature = "ml")]
    crate::ml::register_ml_filtered(ev, &predicate);
    #[cfg(feature = "nn")]
    crate::nn::register_nn_filtered(ev, &predicate);
    #[cfg(feature = "functional")]
    crate::functional::register_functional_filtered(ev, &predicate);
    #[cfg(feature = "package")]
    crate::package::register_package_filtered(ev, &predicate);
    #[cfg(feature = "dev")]
    crate::dev::register_dev_filtered(ev, &predicate);
    #[cfg(feature = "testing")]
    crate::testing::register_testing_filtered(ev, &predicate);

    // Dispatchers last
    crate::dispatch::register_dispatch(ev);
}
