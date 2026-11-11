//! Streamed deltas, batched into frontend events at display rate.

use std::sync::{Arc, Mutex as StdMutex};
use std::time::Duration;

use crate::client::StreamDelta;
use crate::state::Core;
use crate::types::AgentEvent;

/// Hold a streamed delta at most this long before it goes to the UI: under
/// one frame of a 240 Hz display, so streamed text keeps up with the screen,
/// while a fast provider's deltas still coalesce into a few hundred events a
/// second rather than one per delta. A longer window caps text below display
/// rate whatever the UI's own pacing allows: 25 ms means 40 updates a second.
pub(crate) const FLUSH_INTERVAL: Duration = Duration::from_millis(4);

/// Buffers streamed deltas and flushes them as batched events. Flushes on
/// push once the batch window has elapsed, and via [`spawn_stale_flusher`]
/// when the stream goes quiet with a batch still buffered.
pub(crate) struct TokenBatcher {
    core: Arc<Core>,
    session_id: String,
    content: String,
    thinking: String,
    /// On the runtime clock, which the stale flusher's deadline sleeps on.
    last_flush: tokio::time::Instant,
    /// Wakes the stale flusher when a batch starts buffering.
    buffering: Option<tokio::sync::mpsc::Sender<()>>,
    /// What replies the client asked for again cost, for the caller to charge.
    pub(crate) resent: Vec<crate::client::Usage>,
}

impl TokenBatcher {
    pub(crate) fn new(core: Arc<Core>, session_id: String) -> Self {
        Self {
            core,
            session_id,
            content: String::new(),
            thinking: String::new(),
            last_flush: tokio::time::Instant::now(),
            buffering: None,
            resent: Vec::new(),
        }
    }

    pub(crate) fn push(&mut self, delta: StreamDelta) {
        let was_empty = self.content.is_empty() && self.thinking.is_empty();
        match delta {
            StreamDelta::Content(t) => self.content.push_str(&t),
            StreamDelta::Reasoning(t) => self.thinking.push_str(&t),
            StreamDelta::Usage(usage) => return self.resent.push(usage),
            // Flush first so the wire keeps the order the client saw: the
            // reasoning already streamed, then the notice that it is void.
            StreamDelta::Retry { attempt, max_attempts, reason } => {
                self.flush();
                self.core.send_agent(&self.session_id, AgentEvent::Retry { attempt, max_attempts, reason });
                return;
            }
        }
        if self.last_flush.elapsed() >= FLUSH_INTERVAL {
            self.flush();
        } else if was_empty {
            if let Some(wake) = &self.buffering {
                // Full means a wake is already pending; the flusher reads
                // the window when it takes that wake, so it covers this one.
                let _ = wake.try_send(());
            }
        }
    }

    /// Flush a batch the stream went quiet on. Pushes only flush when the
    /// next delta arrives, so without this a tail buffered just before the
    /// deltas stop - the model switching to tool-call arguments (which emit
    /// no deltas), or a stalled endpoint - stays invisible until the whole
    /// response ends: for a large write_file call, many seconds after the
    /// text was received.
    pub(crate) fn flush_if_stale(&mut self) {
        if (!self.content.is_empty() || !self.thinking.is_empty())
            && self.last_flush.elapsed() >= FLUSH_INTERVAL
        {
            self.flush();
        }
    }

    pub(crate) fn flush(&mut self) {
        if !self.content.is_empty() {
            self.core.send_agent(&self.session_id, AgentEvent::Token { text: std::mem::take(&mut self.content) });
        }
        if !self.thinking.is_empty() {
            self.core.send_agent(&self.session_id, AgentEvent::Thinking { text: std::mem::take(&mut self.thinking) });
        }
        self.last_flush = tokio::time::Instant::now();
    }
}

/// Flush each buffered batch when its window ends, for the lifetime of one
/// streaming request; the caller aborts the task as soon as the stream
/// returns. Abort can only land at an await point, so it never interrupts a
/// flush half way, and a wake racing the caller's final flush finds the
/// buffers already drained and does nothing.
///
/// The task sleeps toward a deadline only while a batch is buffered. Most of
/// a request has nothing to show (the model thinking before its first token,
/// or streaming tool-call arguments), and a fixed tick at the batch window
/// would wake the process 250 times a second through all of it.
///
/// The task holds the batcher weakly: a panicking turn unwinds past the
/// caller's abort, and dropping a JoinHandle detaches rather than aborts, so
/// a strong reference would leave a task parked (and pinning `Core`) for the
/// process lifetime. When the turn's own references drop, the batcher's
/// sender drops with them and the task exits.
pub(crate) fn spawn_stale_flusher(batcher: &Arc<StdMutex<TokenBatcher>>) -> tokio::task::JoinHandle<()> {
    let (wake, mut buffering) = tokio::sync::mpsc::channel(1);
    batcher.lock().unwrap().buffering = Some(wake);
    let batcher = Arc::downgrade(batcher);
    tokio::spawn(async move {
        while buffering.recv().await.is_some() {
            let Some(strong) = batcher.upgrade() else { return };
            let due = strong.lock().unwrap().last_flush + FLUSH_INTERVAL;
            drop(strong);
            tokio::time::sleep_until(due).await;
            let Some(strong) = batcher.upgrade() else { return };
            strong.lock().unwrap().flush_if_stale();
        }
    })
}
