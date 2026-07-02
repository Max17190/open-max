//! Open Max core: a deliberately small agent harness for local models,
//! plus management of a local MLX inference server. UI-free; frontends
//! consume a single event channel.

pub mod agent;
pub mod client;
pub mod config;
pub mod fallback;
pub mod hf;
pub mod mlx;
pub mod prompt;
pub mod sessions;
pub mod state;
pub mod tools;
pub mod types;
