//! Open Max core: a deliberately small, high-performance agent harness.
//! Talks to any OpenAI-compatible endpoint. UI-free; frontends consume a
//! single event channel.

pub mod agent;
pub mod client;
pub mod config;
pub mod doctor;
pub(crate) mod execution;
pub mod hooks;
pub mod ledger;
pub mod memory;
pub mod permissions;
pub mod prompt;
pub mod providers;
pub mod recall;
pub mod registry;
pub mod sessions;
pub mod skills;
pub mod spec;
pub mod state;
pub mod templates;
pub mod text;
pub mod tools;
pub mod trust;
pub mod types;
