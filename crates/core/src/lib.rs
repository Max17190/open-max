//! Open Max core: a deliberately small, high-performance agent harness.
//! Talks to any OpenAI-compatible endpoint; optionally manages a local MLX
//! inference server. UI-free; frontends consume a single event channel.

pub mod agent;
pub mod client;
pub mod config;
pub mod fallback;
pub mod hf;
pub mod mlx;
pub mod prompt;
pub mod registry;
pub mod sessions;
pub mod skills;
pub mod state;
pub mod tools;
pub mod types;
