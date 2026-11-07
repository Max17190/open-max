//! Open Max core: a deliberately small, high-performance agent harness.
//! Talks to any OpenAI-compatible endpoint. UI-free; frontends consume a
//! single event channel.

pub mod agent;
mod batch;
pub mod chatgpt;
pub mod client;
mod compaction;
pub mod config;
pub mod doctor;
pub mod execution;
pub mod hooks;
pub mod ledger;
pub mod memory;
mod notes;
pub mod permissions;
pub mod prompt;
pub mod providers;
pub mod recall;
mod refreeze;
pub mod registry;
pub mod sessions;
pub mod skills;
pub mod spec;
pub mod state;
mod stream;
pub mod templates;
pub mod text;
pub mod tools;
mod transcript;
pub mod trust;
pub mod types;
