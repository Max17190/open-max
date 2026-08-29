//! The session-frozen tool registry: the seven built-in tools plus any
//! external tools configured under `.openmax/tools/*.toml` (project) and
//! `~/.openmax/tools/*.toml` (global), plus discovered skills.
//!
//! Frozen per freeze window: built at session creation, then re-frozen when
//! extension files change on disk (fingerprint at turn start) or the user
//! forces `/reload`. Between freezes the serialized tool schema array is
//! byte-stable so the server's KV cache stays warm. An unchanged disk is a
//! no-op.
//!
//! With no external tools or skills installed, the schema JSON is
//! byte-identical to the built-in `tools::tool_schemas()` array and the
//! prompt gains nothing: extensibility costs zero tokens by default.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde_json::Value;

use crate::execution::{self, CaptureSpec, ProcessError, ProcessRequest, StdinMode, Termination};
use crate::skills::{self, SkillSpec};
use crate::state::CancelToken;
use crate::tools::{self, ToolOutcome};

/// External tool descriptions ride in the prompt prefix of every request, so
/// they are capped hard; authors link a README for anything longer.
pub const MAX_EXTERNAL_DESC_CHARS: usize = 200;
/// Serialized-size cap for one tool's `params` schema. The schema is embedded
/// in the frozen prompt prefix and paid on every request, so it is the one
/// place an extension could grow context cost without bound. Oversized schemas
/// are rejected, not truncated: a truncated schema lies to the model.
pub const MAX_EXTERNAL_PARAMS_BYTES: usize = 4_096;
/// External-tool cap, mirroring `skills::MAX_SKILLS`: the sorted head loads,
/// the rest is counted, reported by the prompt trailer, and named by --check.
pub const MAX_EXTERNAL_TOOLS: usize = 64;
/// The range a `timeout_secs` is clamped into. Named so --check can quote the
/// bounds it warns against instead of restating them.
pub(crate) const MIN_TIMEOUT_SECS: u64 = 1;
pub(crate) const MAX_TIMEOUT_SECS: u64 = 300;
/// Cap on env var names one tool manifest may forward. Sixteen is far past
/// any observed need; the cap exists so a manifest cannot smuggle an
/// unreviewably long grant list past the human reading the approval.
pub(crate) const MAX_ENV_NAMES: usize = 16;

#[derive(Clone, Debug)]
pub enum ToolKind {
    Builtin,
    External(ExternalTool),
}

/// How to run one external tool: spawn `command args...`, write the call's
/// JSON arguments to stdin, read the result from stdout.
#[derive(Clone, Debug)]
pub struct ExternalTool {
    /// sha256 of the defining TOML's bytes at parse time: the identity the
    /// content-bound approval store keys on.
    pub source_sha256: String,
    /// Optional proof-of-life: `openmax --check --run-examples` runs the tool
    /// once with these args through the real spawn path.
    pub example: Option<ToolExample>,
    pub command: String,
    pub args: Vec<String>,
    /// Env var names forwarded from the parent environment; everything else
    /// is scrubbed. Empty = the baseline only. The manifest is the approval
    /// unit, so this list is part of what the human blesses.
    pub env: Vec<String>,
    pub timeout_secs: u64,
    /// Where the definition came from, for actionable error messages.
    pub source_path: PathBuf,
}

#[derive(Clone, Debug)]
pub struct ToolSpec {
    pub name: String,
    pub description: String,
    /// JSON-schema object for the tool's parameters, as sent to the model.
    pub parameters: Value,
    /// Mutating tools go through approval gating.
    pub mutating: bool,
    pub kind: ToolKind,
}

/// Frozen per freeze window: built at session creation and re-frozen only
/// when extension files change on disk (checked by fingerprint at turn start)
/// or the user forces /reload. Between freezes it is immutable, keeping the
/// serialized schema bytes prompt-cache-stable.
pub struct Registry {
    /// Built-ins first in their fixed order, then external tools sorted by
    /// name — deterministic so two builds serialize identically.
    pub tools: Vec<ToolSpec>,
    pub skills: Vec<SkillSpec>,
    /// Skills discovered on disk but not indexed because of the `MAX_SKILLS`
    /// cap. Never silent: the prompt trailer reports this count.
    pub skills_omitted: usize,
    /// External tools discovered on disk but not loaded because of the
    /// `MAX_EXTERNAL_TOOLS` cap; the prompt trailer reports the count.
    pub tools_omitted: usize,
    /// Content hash of the extension files this registry was built from;
    /// compared against disk at turn start to detect extension changes.
    pub ext_fingerprint: u64,
    /// Extension files read this freeze that failed to load, with the parse
    /// reason. Receipts and the unknown-tool error name these so a broken
    /// write is never mistaken for a live capability.
    pub broken: Vec<(PathBuf, String)>,
    /// Every capability file path THIS freeze's capture actually read (tool
    /// manifests and SKILL.mds). The refreeze classifier asks whether a path
    /// still existed in this generation without a second disk probe, which
    /// would race the capture it claims to describe. Empty for a
    /// manifest-restored registry, which only ever sits on the outgoing side
    /// of that comparison.
    pub(crate) read_paths: std::collections::HashSet<PathBuf>,
    /// Broken TOOL manifests with the name each occupies: the declared name,
    /// or the file stem when the document is too broken to yield one - the
    /// same derivation the withhold pass uses. Lets the refreeze classifier
    /// see that a broken file already explains an absent name. Empty for a
    /// manifest-restored registry.
    pub(crate) broken_tools: Vec<(PathBuf, String)>,
    /// Same-directory skill name collisions, one record per name (name,
    /// displaced paths, winning path, winner indexed), for the refreeze
    /// receipt. Empty for a manifest-restored registry, which only narrates
    /// deltas it can compute.
    pub(crate) shadowed_skills: Vec<(String, Vec<PathBuf>, PathBuf, bool)>,
    /// Memory files (stem, content hash) this freeze indexed; None for a
    /// registry rebuilt from a manifest that predates the field, so the
    /// first refreeze after an upgrade does not narrate every memory as new.
    pub memory_files: Option<Vec<(String, u64)>>,
    /// The rendered memory index section captured in the SAME scan as
    /// `memory_files`, so the frozen prompt and the receipt cannot disagree.
    /// `None` here is ambiguous on its own - an empty scan and a registry that
    /// never scanned both leave it None - so `memory_scanned` disambiguates.
    pub memory_section: Option<(String, Vec<(String, usize)>)>,
    /// The memory index rows (stem, line bytes) frozen WITH the persisted
    /// prompt, independent of the resettable resume-delta baseline above:
    /// from_manifest clears `memory_files` so the first refreeze reports no
    /// spurious delta, while /context still needs the freeze's own row
    /// accounting. Carried by the manifest (version 4); a pre-field manifest
    /// reads as absent and refreezes, so this is Some on every live path.
    pub(crate) frozen_memory_rows: Option<Vec<(String, usize)>>,
    /// True only when THIS registry actually ran a memory scan (a fresh
    /// freeze). A manifest-restored registry sets `memory_files` for the
    /// resume delta but never captured a section, so it is false and the
    /// prompt scans fresh - otherwise a resumed session would render no
    /// memory index at all while `memory_files.is_some()` (Greptile). A
    /// scanned-but-empty freeze is true, so its empty selection is honored
    /// (no rescan), which is the invariant `memory_section` exists for.
    pub memory_scanned: bool,
    /// Schema array value form: prompt breakdown and tests walk this.
    schemas: Value,
    /// Schema array wire form: frozen once so chat request bodies inject the
    /// exact same tool bytes every iteration without re-serializing Value.
    schemas_wire: Arc<str>,
    by_name: HashMap<String, usize>,
}

/// One immutable read generation of every file the registry can activate.
/// Parsing and fingerprinting consume these same bytes, so a concurrent edit
/// or atomic symlink swap cannot produce a registry whose content disagrees
/// with its persisted fingerprint.
pub(crate) struct ExtensionSnapshot {
    fingerprint: u64,
    external: Vec<ToolSpec>,
    skills: Vec<SkillSpec>,
    /// External tools discovered but dropped by the `MAX_EXTERNAL_TOOLS` cap.
    tools_omitted: usize,
    /// Skills discovered but dropped by the `MAX_SKILLS` index cap.
    skills_omitted: usize,
    /// Every capability file this generation read: (path, sha256, bytes).
    /// The ledger records exactly this generation, so what it attests is what
    /// the freeze actually used - never a second read that could differ.
    pub(crate) files: Vec<(PathBuf, String, Vec<u8>)>,
    /// Files read but not loaded, with the reason. The bytes are already in
    /// the fingerprint (a broken write still triggers a refreeze); keeping
    /// the reason lets that refreeze's receipt say the tool is NOT live.
    pub(crate) broken: Vec<(PathBuf, String)>,
    /// The tool-tier subset of `broken` with the name each file occupies
    /// (declared, or stem as the fallback), for the refreeze classifier.
    pub(crate) broken_tools: Vec<(PathBuf, String)>,
    /// Same-directory skill name collisions, one record per name: (name,
    /// displaced paths, winning path, winner indexed). The receipt names
    /// these; cross-tier precedence is not here.
    pub(crate) shadowed_skills: Vec<(String, Vec<PathBuf>, PathBuf, bool)>,
    /// Project memory files (stem, content hash). Memory rides the frozen
    /// prompt's index, so a memory write moves the fingerprint and refreezes:
    /// the fact is live from the next step, deterministically, instead of
    /// whenever some unrelated extension file happens to change. Not ledger
    /// files (data, not capability), so not in `files`.
    pub(crate) memory_files: Vec<(String, u64)>,
    /// The index section from the same scan as `memory_files`.
    pub(crate) memory_section: Option<(String, Vec<(String, usize)>)>,
}

impl ExtensionSnapshot {
    pub(crate) fn fingerprint(&self) -> u64 {
        self.fingerprint
    }
}

/// Read, content-hash, and parse every extension file exactly once: external
/// tool TOMLs and skill SKILL.mds, global first and project second. Contents
/// (not mtimes) detect same-length rewrites. Only parsed specs survive the
/// scan, so peak memory is bounded by one source file plus the frozen registry.
pub(crate) fn capture_extensions(data_dir: &Path, project_root: &Path) -> ExtensionSnapshot {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut h = DefaultHasher::new();
    let mut files_read: Vec<(PathBuf, String, Vec<u8>)> = Vec::new();
    // (path, reason, dir precedence index, declared name if recoverable):
    // tools are keyed by DECLARED name, not file stem, so a collision between
    // a broken file and a loaded definition must be judged on the name the
    // broken file declares (bar.toml can declare name = "foo"); the stem is
    // only the fallback when the file is too broken to yield a name.
    let mut broken_at: Vec<(PathBuf, String, usize, Option<String>)> = Vec::new();
    let mut external_by_name: HashMap<String, (usize, ToolSpec)> = HashMap::new();
    for (dir_index, dir) in external_tool_dirs(data_dir, project_root).into_iter().enumerate() {
        dir.hash(&mut h);
        let Ok(rd) = std::fs::read_dir(&dir) else {
            continue;
        };
        let mut files: Vec<PathBuf> = rd
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|e| e == "toml"))
            .collect();
        files.sort();
        for path in files {
            path.hash(&mut h);
            // A read failure on a file the directory listing named is a broken
            // entry, not a silent skip: an approved tool whose manifest turned
            // unreadable would otherwise vanish from `tools` AND `broken`, so
            // the refreeze receipt could name it nowhere - no "NOT loaded"
            // clause, and (because the path is still a file on disk) no removal
            // clause either (Greptile). NotFound is the one exception: a file
            // deleted between the listing and the read is gone, and a removal,
            // not a broken entry.
            let bytes = match std::fs::read(&path) {
                Ok(b) => Some(b),
                Err(e) => {
                    if e.kind() != std::io::ErrorKind::NotFound {
                        broken_at.push((path.clone(), format!("unreadable: {e}"), dir_index, None));
                    }
                    None
                }
            };
            bytes.hash(&mut h);
            let Some(bytes) = bytes else { continue };
            files_read.push((path.clone(), crate::ledger::sha256_hex(&bytes), bytes.clone()));
            let Ok(text) = std::str::from_utf8(&bytes) else {
                broken_at.push((path, "not valid UTF-8".into(), dir_index, None));
                continue;
            };
            match parse_tool_source(&path, text) {
                // Global is scanned first, so a project definition wins.
                Ok(spec) => {
                    external_by_name.insert(spec.name.clone(), (dir_index, spec));
                }
                Err(reason) => {
                    let declared = declared_tool_name(text);
                    broken_at.push((path, reason, dir_index, declared));
                }
            }
        }
    }
    // A broken file colliding by stem with a loaded definition must not lie
    // in either direction. A broken PROJECT file over a valid global: the
    // user's override is what they meant to run, so the global fallback is
    // withheld (running different code than intended, silently, is the
    // failure this whole surface exists to prevent). A broken GLOBAL file
    // under a valid project override: the override is legitimately active,
    // and the reason says so instead of claiming the name is not callable.
    let mut broken: Vec<(PathBuf, String)> = Vec::new();
    // (path, occupied name) per broken tool file: the refreeze classifier
    // tells "removed" from "explained by a broken file" with this, never by
    // re-probing disk after the capture.
    let mut broken_tools: Vec<(PathBuf, String)> = Vec::new();
    for (path, mut reason, broken_dir, declared) in broken_at {
        let name = declared
            .or_else(|| path.file_stem().map(|s| s.to_string_lossy().to_string()));
        if let Some(name) = name {
            if let Some((loaded_dir, _)) = external_by_name.get(&name) {
                if broken_dir > *loaded_dir {
                    reason.push_str(&format!(
                        "; the lower-precedence definition of '{name}' is withheld until \
                         this override is fixed or removed"
                    ));
                    external_by_name.remove(&name);
                } else {
                    reason.push_str(&format!(
                        "; a higher-precedence definition of '{name}' is active and callable"
                    ));
                }
            }
            broken_tools.push((path.clone(), name));
        }
        broken.push((path, reason));
    }
    let external_by_name: HashMap<String, ToolSpec> =
        external_by_name.into_iter().map(|(k, (_, v))| (k, v)).collect();
    let mut skills_by_name: HashMap<String, SkillSpec> = HashMap::new();
    // (name, displaced paths, winning path, winner indexed): files in the
    // SAME directory declaring one name. Cross-tier shadowing is precedence
    // (a project definition wins over a global one, deliberately, silently);
    // same-tier last-wins is an accident of scan order, so the refreeze
    // receipt names it the moment it happens, the way --check already does.
    // Collisions coalesce to ONE record per name (three namesakes must not
    // report the intermediate winner as cap-dropped while the final one is
    // indexed), and the indexed flag is settled AFTER the skill cap below:
    // a winner the cap drops must not be reported as indexed (Greptile,
    // both).
    let mut shadowed_skills: Vec<(String, Vec<PathBuf>, PathBuf, bool)> = Vec::new();
    for dir in skills::skill_dirs(data_dir, project_root) {
        dir.hash(&mut h);
        let Ok(rd) = std::fs::read_dir(&dir) else {
            continue;
        };
        let mut files: Vec<PathBuf> = rd
            .flatten()
            .map(|e| e.path().join("SKILL.md"))
            .filter(|p| p.is_file())
            .collect();
        files.sort();
        for path in files {
            path.hash(&mut h);
            let bytes = std::fs::read(&path).ok();
            bytes.hash(&mut h);
            let Some(bytes) = bytes else { continue };
            files_read.push((path.clone(), crate::ledger::sha256_hex(&bytes), bytes.clone()));
            let Ok(text) = std::str::from_utf8(&bytes) else {
                broken.push((path, "not valid UTF-8".into()));
                continue;
            };
            match skills::parse_skill_source(&path, text) {
                // Global is scanned first, so a project definition wins.
                Ok(spec) => {
                    let name = spec.name.clone();
                    let winner = spec.path.clone();
                    if let Some(prev) = skills_by_name.insert(name.clone(), spec) {
                        if prev.path.starts_with(&dir) {
                            match shadowed_skills.iter_mut().find(|(n, ..)| *n == name) {
                                Some((_, displaced, current, _)) => {
                                    displaced.push(std::mem::replace(current, winner));
                                }
                                None => {
                                    shadowed_skills.push((name, vec![prev.path], winner, false));
                                }
                            }
                        } else {
                            // Cross-tier precedence is deliberate and silent,
                            // and it MOOTS any collision recorded in the losing
                            // tier: keeping that record left its winner
                            // pointing at a path the project definition just
                            // displaced, and the post-cap settle then called
                            // the name unindexed while the project skill is
                            // active (Greptile).
                            shadowed_skills.retain(|(n, ..)| *n != name);
                        }
                    }
                }
                Err(reason) => broken.push((path, reason)),
            }
        }
    }
    // Memory: ONE read produces both the fingerprint bytes and the index, so
    // the fingerprint (which decides refreeze) and the frozen index cannot be
    // captured from two different file generations - an atomic replace between
    // two scans could otherwise freeze the replacement's index under the
    // original's fingerprint, and a restore-to-original would then skip the
    // refreeze that would fix it (Greptile P1). The fingerprint hashes every
    // VALID-named memory byte (a write to one refreezes); the index is the
    // indexed subset of the SAME bytes. Never ledgered (data, not capability).
    let mem = crate::memory::freeze_snapshot(project_root, crate::memory::unix_now());
    {
        let dir = project_root.join(crate::memory::MEMORY_DIR);
        dir.hash(&mut h);
        for (path, bytes) in &mem.fingerprint_files {
            path.hash(&mut h);
            bytes.hash(&mut h);
        }
    }
    let (memory_section, memory_files) = (mem.section, mem.identities);
    let mut external: Vec<ToolSpec> = external_by_name.into_values().collect();
    // Built-in shadows never load (assemble drops them); excluding them here
    // keeps them from wasting a cap slot, so --check and the loader agree on
    // exactly which files are live.
    external.retain(|t| !tools::TOOL_NAMES.contains(&t.name.as_str()));
    external.sort_by(|a, b| a.name.cmp(&b.name));
    // Caps bound the frozen prompt, but a tool or skill dropped here must not
    // vanish silently: the counts survive so the prompt trailer can say what
    // is not loaded and --check can name the files.
    let tools_omitted = external.len().saturating_sub(MAX_EXTERNAL_TOOLS);
    external.truncate(MAX_EXTERNAL_TOOLS);
    let mut discovered_skills: Vec<SkillSpec> = skills_by_name.into_values().collect();
    discovered_skills.sort_by(|a, b| a.name.cmp(&b.name));
    let skills_omitted = discovered_skills.len().saturating_sub(skills::MAX_SKILLS);
    discovered_skills.truncate(skills::MAX_SKILLS);
    // Settle the collision winners' indexed state against the RENDER's own
    // inclusion decision, not list membership: the 50-skill cap and the
    // 3000-byte index budget (first-fit, applied at prompt render) both
    // drop lines, and a receipt calling a dropped line indexed sends the
    // author hunting for it (Greptile, twice). A zero cost is a line the
    // prompt does not carry.
    let included: std::collections::HashSet<String> =
        crate::prompt::skill_index_costs(project_root, &discovered_skills)
            .into_iter()
            .filter(|(_, cost)| *cost > 0)
            .map(|(name, _)| name)
            .collect();
    for (name, _, _, indexed) in &mut shadowed_skills {
        *indexed = included.contains(name);
    }
    ExtensionSnapshot {
        fingerprint: h.finish(),
        external,
        skills: discovered_skills,
        tools_omitted,
        skills_omitted,
        files: files_read,
        broken,
        broken_tools,
        shadowed_skills,
        memory_files,
        memory_section,
    }
}

/// The `name` a tool manifest declares, when the document is TOML enough to
/// yield one - a manifest that fails the spec (missing `command`, bad
/// schema) still names its tool, and that name is what a collision with a
/// loaded definition must be judged on. Malformed TOML yields None.
fn declared_tool_name(text: &str) -> Option<String> {
    let doc: toml::Value = toml::from_str(text).ok()?;
    let name = doc.get("name")?.as_str()?.trim();
    (!name.is_empty()).then(|| name.to_string())
}

/// Compatibility helper for diagnostics and tests that only need the content
/// identity. Activation paths should retain and parse the full snapshot.
pub fn extensions_fingerprint(data_dir: &Path, project_root: &Path) -> u64 {
    capture_extensions(data_dir, project_root).fingerprint
}

impl Registry {
    /// Discover external tools and skills for a project and freeze the
    /// registry, stamped with the fingerprint of what was read.
    pub fn build(data_dir: &Path, project_root: &Path) -> Self {
        Self::from_snapshot(capture_extensions(data_dir, project_root))
    }

    pub(crate) fn from_snapshot(snapshot: ExtensionSnapshot) -> Self {
        let mut registry = Self::assemble(snapshot.external, snapshot.skills);
        registry.ext_fingerprint = snapshot.fingerprint;
        registry.tools_omitted = snapshot.tools_omitted;
        registry.skills_omitted = snapshot.skills_omitted;
        registry.read_paths = snapshot.files.iter().map(|(p, _, _)| p.clone()).collect();
        registry.broken = snapshot.broken;
        registry.broken_tools = snapshot.broken_tools;
        registry.shadowed_skills = snapshot.shadowed_skills;
        registry.frozen_memory_rows = Some(
            snapshot.memory_section.as_ref().map(|(_, rows)| rows.clone()).unwrap_or_default(),
        );
        registry.memory_files = Some(snapshot.memory_files);
        registry.memory_section = snapshot.memory_section;
        registry.memory_scanned = true;
        registry
    }

    /// A registry with built-ins only: used for sessions that predate the
    /// extensibility layer, so their behavior never changes retroactively.
    pub fn builtin_only() -> Self {
        Self::assemble(Vec::new(), Vec::new())
    }

    pub(crate) fn assemble(mut external: Vec<ToolSpec>, skills: Vec<SkillSpec>) -> Self {
        // Built-ins come straight from the canonical schema literals so the
        // registry can never drift from what tools.rs implements.
        let mut tools_list = builtin_specs();
        // Built-in names win over external ones: shadowing a built-in would
        // silently change core behavior mid-workflow.
        external.retain(|t| !tools::TOOL_NAMES.contains(&t.name.as_str()));
        external.sort_by(|a, b| a.name.cmp(&b.name));
        tools_list.extend(external);

        let mut schemas = tools::tool_schemas().clone();
        if let Some(arr) = schemas.as_array_mut() {
            for spec in tools_list.iter().filter(|s| !matches!(s.kind, ToolKind::Builtin)) {
                arr.push(serde_json::json!({
                    "type": "function",
                    "function": {
                        "name": spec.name,
                        "description": spec.description,
                        "parameters": spec.parameters,
                    }
                }));
            }
        }

        let by_name = tools_list
            .iter()
            .enumerate()
            .map(|(i, s)| (s.name.clone(), i))
            .collect();
        let schemas_wire: Arc<str> = schemas.to_string().into();
        Self {
            tools: tools_list,
            skills,
            skills_omitted: 0,
            tools_omitted: 0,
            ext_fingerprint: 0,
            broken: Vec::new(),
            read_paths: std::collections::HashSet::new(),
            broken_tools: Vec::new(),
            shadowed_skills: Vec::new(),
            memory_files: None,
            frozen_memory_rows: None,
            memory_section: None,
            memory_scanned: false,
            schemas,
            schemas_wire,
            by_name,
        }
    }

    pub fn get(&self, name: &str) -> Option<&ToolSpec> {
        self.by_name.get(name).map(|&i| &self.tools[i])
    }

    pub fn is_mutating(&self, name: &str) -> bool {
        self.get(name).map(|s| s.mutating).unwrap_or(false)
    }

    /// Every tool name, in frozen order. Feeds the fallback call parser.
    pub fn tool_names(&self) -> Vec<String> {
        self.tools.iter().map(|s| s.name.clone()).collect()
    }

    /// The OpenAI-format tool schema array, frozen once at assemble time.
    pub fn tool_schemas_json(&self) -> &Value {
        &self.schemas
    }

    /// Pre-serialized OpenAI tool schema array bytes, frozen with the registry.
    /// Injected into chat request bodies via `RawValue` so multi-iteration turns
    /// never re-walk the tools array.
    pub fn tool_schemas_wire(&self) -> &str {
        &self.schemas_wire
    }

    /// Owned handle to the same frozen bytes, for turn state that must
    /// outlive a mid-turn registry swap.
    pub(crate) fn schemas_wire_arc(&self) -> Arc<str> {
        self.schemas_wire.clone()
    }

    /// Estimated tokens the frozen schemas add to *every* request in this
    /// freeze window: fixed overhead the context budget must carry alongside
    /// the transcript, or compaction fires too late and the request overruns
    /// the real window. A refreeze rebuilds the registry, so this always
    /// reports the generation currently on the wire.
    pub fn schema_tokens(&self) -> usize {
        crate::types::estimate_tokens(self.schemas_wire.len())
    }

    pub async fn execute(
        &self,
        name: &str,
        args: &Value,
        data_dir: &Path,
        root: &Path,
        caps: tools::OutputCaps,
        cancel: Arc<CancelToken>,
    ) -> ToolOutcome {
        match self.get(name).map(|s| s.kind.clone()) {
            Some(ToolKind::Builtin) => tools::execute(name, args, data_dir, root, caps, cancel).await,
            Some(ToolKind::External(tool)) => {
                spawn_external(name, &tool, args, data_dir, root, caps, cancel, None).await
            }
            None => self.unknown_tool_error(name, root),
        }
    }

    /// Probe one EXTERNAL tool inside an OS sandbox (no network, writes
    /// confined to `scratch`, scrubbed env): the pre-approval iteration path
    /// for `--run-examples` on tools no human has blessed yet. Only external
    /// tools are probeable; built-ins have no unapproved state.
    #[allow(clippy::too_many_arguments)]
    pub async fn execute_example_sandboxed(
        &self,
        name: &str,
        args: &Value,
        data_dir: &Path,
        root: &Path,
        caps: tools::OutputCaps,
        cancel: Arc<CancelToken>,
        scratch: &Path,
    ) -> ToolOutcome {
        match self.get(name).map(|s| s.kind.clone()) {
            Some(ToolKind::External(tool)) => {
                let sandbox = execution::SandboxPolicy {
                    ro_root: root.to_path_buf(),
                    rw_scratch: scratch.to_path_buf(),
                };
                spawn_external(name, &tool, args, data_dir, root, caps, cancel, Some(sandbox))
                    .await
            }
            Some(ToolKind::Builtin) => {
                ToolOutcome::err(format!("'{name}' is a built-in; probes are for external tools"))
            }
            None => self.unknown_tool_error(name, root),
        }
    }

    /// A name matching a file that failed to load gets the parse reason, not
    /// a bare "unknown": the model most likely wrote (or was promised) that
    /// very file.
    fn unknown_tool_error(&self, name: &str, root: &Path) -> ToolOutcome {
        // A broken .toml is matched by the NAME it occupies (its declared
        // name, or the stem fallback - the registry's own key), never by
        // stem alone: `todo-scan.toml` declaring `name = "todo_scan"` is
        // exactly the file a caller of `todo_scan` means, and the stem match
        // answered with a bare unknown (round-7 audit). The stem and
        // SKILL.md matchers stay as the fallback for a manifest-restored
        // registry, whose broken_tools list is empty.
        let named = self
            .broken_tools
            .iter()
            .find(|(_, occupied)| occupied.as_str() == name)
            .and_then(|(p, _)| self.broken.iter().find(|(bp, _)| bp == p));
        if let Some((path, reason)) = named.or_else(|| {
            self.broken.iter().find(|(path, _)| {
                path.file_stem().is_some_and(|stem| stem == name)
                    || path.file_name().is_some_and(|f| f == "SKILL.md")
                        && path.parent().and_then(|d| d.file_name()).is_some_and(|d| d == name)
            })
        }) {
            let shown = path.strip_prefix(root).unwrap_or(path);
            return ToolOutcome::err(format!(
                "unknown tool: {name}; {} exists but did NOT load: {reason}. \
                 Fix the file and verify with bash: openmax --check. The \
                 available tools are {}",
                shown.display(),
                self.tool_names().join(", ")
            ));
        }
        ToolOutcome::err(format!(
            "unknown tool: {name}; the available tools are {}",
            self.tool_names().join(", ")
        ))
    }
}

impl Default for Registry {
    fn default() -> Self {
        Self::builtin_only()
    }
}

/// The persisted record of what a session's registry froze at creation:
/// enough to rebuild the exact same schemas on resume without re-reading
/// any config from disk, so a session never changes shape retroactively.
#[derive(serde::Serialize, serde::Deserialize)]
pub struct RegistryManifest {
    pub version: u32,

    pub external_tools: Vec<ExternalToolManifest>,
    pub skills: Vec<SkillSpec>,
    /// Fingerprint of the extension files at freeze time. Manifests written
    /// before this field default to 0, which mismatches any real disk state
    /// and triggers one re-freeze on the next turn (forward only).
    #[serde(default)]
    pub ext_fingerprint: u64,
    /// Memory files (stem, content hash) the freeze indexed, so a resumed
    /// session's first memory receipt is a real delta. Additive: absent in
    /// older manifests, which then narrate no memory delta once.
    #[serde(default)]
    pub memory_files: Option<Vec<(String, u64)>>,
    /// The memory index rows (stem, line bytes) frozen WITH the persisted
    /// prompt: /context on a resumed session prices exactly these. Recorded
    /// here because re-deriving them by parsing the persisted prompt was an
    /// arms race against attacker-controlled bytes rendered into later
    /// sections (Greptile, three rounds).
    #[serde(default)]
    pub memory_rows: Option<Vec<(String, usize)>>,
}

/// Current manifest format. A manifest carrying any other version is treated
/// as absent (fail closed on unknown future formats): the session falls back
/// to built-ins and the next turn re-freezes cleanly from disk, instead of
/// deserializing a newer format into the wrong shape.
///
/// v3: external tools carry an `env` allowlist. A v2 manifest has no such
/// field, and defaulting it to empty would resume a credential-dependent
/// tool with a scrubbed environment while the fingerprint still matched
/// disk (no refreeze to repair it). Bumping the version makes v2 read as
/// absent, so the session re-freezes from disk once and picks up the real
/// grant - the same forward-only migration this constant already promises.
// 4: memory_rows joined the manifest (the persisted /context accounting;
// parsing it back out of the prompt was forgeable by newline-bearing
// filenames rendered into later sections). Old manifests read as absent
// and refreeze, so every live manifest carries exact rows.
pub const MANIFEST_VERSION: u32 = 4;

#[derive(serde::Serialize, serde::Deserialize)]
pub struct ExternalToolManifest {
    /// Frozen source hash; absent in old manifests, which reads as
    /// unapproved (the safe direction: the tool asks once more).
    #[serde(default)]
    pub source_sha256: String,
    pub name: String,
    pub description: String,
    pub parameters: Value,
    pub mutating: bool,
    pub command: String,
    pub args: Vec<String>,
    /// Env var names forwarded to the tool. Present in every v3 manifest;
    /// the serde default only guards a hand-edited file, because v2
    /// manifests (which lack it) never reach this deserializer - the
    /// version gate reads them as absent so the session refreezes from disk.
    #[serde(default)]
    pub env: Vec<String>,
    pub timeout_secs: u64,
    pub source_path: PathBuf,
}

impl Registry {
    pub fn to_manifest(&self) -> RegistryManifest {
        let external_tools = self
            .tools
            .iter()
            .filter_map(|spec| match &spec.kind {
                ToolKind::Builtin => None,
                ToolKind::External(t) => Some(ExternalToolManifest {
                    source_sha256: t.source_sha256.clone(),
                    name: spec.name.clone(),
                    description: spec.description.clone(),
                    parameters: spec.parameters.clone(),
                    mutating: spec.mutating,
                    command: t.command.clone(),
                    args: t.args.clone(),
                    env: t.env.clone(),
                    timeout_secs: t.timeout_secs,
                    source_path: t.source_path.clone(),
                }),
            })
            .collect();
        RegistryManifest {
            memory_files: self.memory_files.clone(),
            // The frozen channel, not a re-parse: a restored registry
            // re-suspending must keep the accounting its persisted prompt
            // still depends on.
            memory_rows: self.frozen_memory_rows.clone(),
            version: MANIFEST_VERSION,
            external_tools,
            skills: self.skills.clone(),
            ext_fingerprint: self.ext_fingerprint,
        }
    }

    pub fn from_manifest(manifest: RegistryManifest) -> Self {
        let external = manifest
            .external_tools
            .into_iter()
            .map(|t| ToolSpec {
                name: t.name,
                description: t.description,
                parameters: t.parameters,
                mutating: t.mutating,
                kind: ToolKind::External(ExternalTool {
                    source_sha256: t.source_sha256,
                    // Examples are a --check concern read from disk; frozen
                    // sessions never run them.
                    example: None,
                    command: t.command,
                    args: t.args,
                    env: t.env,
                    timeout_secs: t.timeout_secs,
                    source_path: t.source_path,
                }),
            })
            .collect();
        let mut registry = Self::assemble(external, manifest.skills);
        registry.ext_fingerprint = manifest.ext_fingerprint;
        // A restored registry does NOT reuse the manifest's memory identities
        // as the delta baseline. `memory_scanned` is false, so the prompt
        // rescans and shows the CURRENT memory selection; keeping the older
        // suspend-time identities would make the first refreeze report an
        // offline replacement the prompt already shows as newly indexed and
        // the old item as dropped (Greptile). memory_files stays None so the
        // first refreeze establishes the fresh scan as the baseline with no
        // spurious delta. The row accounting survives on the frozen
        // channel: /context prices the rows of the freeze that WROTE the
        // persisted prompt, which is precisely what the manifest carries
        // (Greptile).
        registry.frozen_memory_rows = manifest.memory_rows.clone();
        registry
    }

    /// True when the registry carries anything beyond the built-ins; an
    /// all-builtin session needs no manifest file at all.
    pub fn has_extensions(&self) -> bool {
        !self.skills.is_empty()
            || self.tools.iter().any(|s| !matches!(s.kind, ToolKind::Builtin))
    }
}

/// One-line human summary of a call, for approval prompts and tool cards.
/// Registry-free on purpose: built-in names summarize by their known argument
/// shapes, every other name by the external heuristic — exactly what a
/// registry lookup would produce, without threading session state into the UI.
pub fn summarize_call(name: &str, args: &Value) -> String {
    if tools::TOOL_NAMES.contains(&name) {
        tools::summarize_call(name, args)
    } else {
        summarize_external(name, args)
    }
}

/// Built-in tool specs derived from the canonical `tools::tool_schemas()`
/// literals, so name/description/parameters have a single source of truth.
fn builtin_specs() -> Vec<ToolSpec> {
    let schemas = tools::tool_schemas();
    schemas
        .as_array()
        .expect("builtin schemas are an array")
        .iter()
        .map(|entry| {
            let f = &entry["function"];
            let name = f["name"].as_str().expect("builtin schema has a name").to_string();
            ToolSpec {
                mutating: tools::is_mutating(&name),
                description: f["description"].as_str().unwrap_or("").to_string(),
                parameters: f["parameters"].clone(),
                name,
                kind: ToolKind::Builtin,
            }
        })
        .collect()
}

/// External tools have arbitrary parameter names, so summaries fall back to
/// the most path/command-looking argument available, and finally to the tool's
/// own name: a call whose arguments hold no strings at all (numbers, bools, or
/// none) must still summarize to something, or an approval card renders as
/// bare empty parens and the human is asked to approve a blank.
fn summarize_external(name: &str, args: &Value) -> String {
    for key in ["command", "path", "pattern"] {
        if let Some(v) = args[key].as_str() {
            return v.to_string();
        }
    }
    args.as_object()
        .and_then(|o| o.values().find_map(|v| v.as_str()))
        .unwrap_or(name)
        .to_string()
}

/// The TOML shape of one tool definition file. Unknown keys are rejected so a
/// misspelled `mutating` cannot silently drop a tool out of the approval gate.
#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct ExternalToolFile {
    #[serde(default)]
    example: Option<ExampleFile>,
    name: String,
    description: String,
    /// JSON-schema object for the parameters; defaults to "no parameters".
    #[serde(default)]
    params: Option<Value>,
    command: String,
    #[serde(default)]
    args: Vec<String>,
    #[serde(default = "default_timeout")]
    timeout_secs: u64,
    #[serde(default)]
    mutating: bool,
    /// Environment variable NAMES this tool receives from the parent
    /// environment, on top of the scrubbed baseline. Part of the manifest,
    /// so the credential grant is inside the bytes a human approves.
    #[serde(default)]
    env: Vec<String>,
}

fn default_timeout() -> u64 {
    60
}

/// One runnable example: the call must exit 0, and when `expect_regex` is
/// set its output must match. Declared in the tool file so the contract
/// travels with the tool.
#[derive(Clone, Debug)]
pub struct ToolExample {
    pub args: Value,
    pub expect_regex: Option<String>,
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct ExampleFile {
    #[serde(default)]
    args: Option<Value>,
    #[serde(default)]
    expect_regex: Option<String>,
}

/// Global then project tool dirs; later dirs win on name collision.
/// Global then project tool dirs. The global one is derived from the session's
/// own `data_dir`, not from `$HOME`: approvals, sessions and trust all live in
/// `data_dir`, so discovering capabilities from somewhere else means a tool can
/// be found in one place and its approval recorded in another.
pub(crate) fn external_tool_dirs(data_dir: &Path, project_root: &Path) -> [PathBuf; 2] {
    [
        data_dir.join("tools"),
        project_root.join(".openmax").join("tools"),
    ]
}

#[cfg(test)]
fn discover_external_in(dirs: &[PathBuf]) -> Vec<ToolSpec> {
    let mut by_name: HashMap<String, ToolSpec> = HashMap::new();
    for dir in dirs {
        let Ok(entries) = std::fs::read_dir(dir) else { continue };
        let mut paths: Vec<PathBuf> = entries
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|e| e == "toml"))
            .collect();
        // Deterministic within a dir; later dirs (the project) win overall.
        paths.sort();
        for path in paths {
            if let Ok(spec) = parse_tool_file(&path) {
                by_name.insert(spec.name.clone(), spec);
            }
        }
    }
    by_name.into_values().collect()
}

#[cfg(test)]
pub(crate) fn parse_tool_file_from_text_for_tests(text: &str) -> Result<ToolSpec, String> {
    parse_tool_source(Path::new("test.toml"), text)
}

/// Errors are ignored by discovery and surfaced verbatim by `openmax --check`.
pub(crate) fn parse_tool_file(path: &Path) -> Result<ToolSpec, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("unreadable: {e}"))?;
    parse_tool_source(path, &text)
}

/// The same parse from bytes a caller already read (and hashed), so what an
/// approval records about a manifest cannot come from a different generation
/// of the file than the hash it blesses.
/// The `description` exactly as the manifest wrote it, before the schema cap
/// clamps it, normalized the way the parse normalizes it. `openmax --check`
/// reads it from the same bytes the parse used, so a report can say the
/// written line is longer than the schema one without a second read of the
/// file (mirrors `skills::raw_description`). None when the file does not
/// parse; the parse error already reports that.
pub(crate) fn raw_description(text: &str) -> Option<String> {
    let value: toml::Value = toml::from_str(text).ok()?;
    let desc = value.get("description")?.as_str()?;
    Some(desc.trim().replace(['\n', '\r'], " "))
}

pub(crate) fn parse_tool_source(path: &Path, text: &str) -> Result<ToolSpec, String> {
    let source_sha256 = crate::ledger::sha256_hex(text.as_bytes());
    let file: ExternalToolFile =
        toml::from_str(text).map_err(|e| crate::spec::manifest_toml_error(&e, "tools"))?;
    let name = file.name.trim().to_string();
    // Boring, model-friendly names only; anything else is a config mistake.
    let name_ok = !name.is_empty()
        && name.len() <= 64
        && name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-');
    if !name_ok {
        return Err(format!(
            "invalid tool name '{name}': 1-64 chars of [a-zA-Z0-9_-] required"
        ));
    }
    if file.command.trim().is_empty() {
        return Err("command is empty".into());
    }
    let mut description = file.description.trim().replace(['\n', '\r'], " ");
    if description.chars().count() > MAX_EXTERNAL_DESC_CHARS {
        description = description.chars().take(MAX_EXTERNAL_DESC_CHARS).collect::<String>() + "…";
    }
    let example = match file.example {
        None => None,
        Some(ex) => {
            if let Some(pattern) = ex.expect_regex.as_deref() {
                regex::Regex::new(pattern)
                    .map_err(|e| format!("example.expect_regex is invalid: {e}"))?;
            }
            let args = match ex.args {
                None => serde_json::json!({}),
                Some(v) if v.is_object() => v,
                Some(_) => return Err("example.args must be a table".into()),
            };
            Some(ToolExample { args, expect_regex: ex.expect_regex })
        }
    };
    let parameters = match file.params {
        Some(p) if p.is_object() => {
            validate_params_schema(&p)?;
            p
        }
        Some(_) => return Err("params must be a JSON-schema object".into()),
        None => serde_json::json!({ "type": "object", "properties": {} }),
    };
    if file.env.len() > MAX_ENV_NAMES {
        return Err(format!(
            "env lists {} variables; at most {MAX_ENV_NAMES} may be declared",
            file.env.len()
        ));
    }
    for var in &file.env {
        let ok = !var.is_empty()
            && var.chars().next().is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
            && var.chars().all(|c| c.is_ascii_alphanumeric() || c == '_');
        if !ok {
            return Err(format!(
                "invalid env var name '{var}': [A-Za-z_][A-Za-z0-9_]* required"
            ));
        }
    }
    Ok(ToolSpec {
        name,
        description,
        parameters,
        mutating: file.mutating,
        kind: ToolKind::External(ExternalTool {
            source_sha256,
            example,
            command: file.command.trim().to_string(),
            args: file.args,
            env: file.env,
            timeout_secs: file.timeout_secs.clamp(MIN_TIMEOUT_SECS, MAX_TIMEOUT_SECS),
            source_path: path.to_path_buf(),
        }),
    })
}

/// Light structural checks on a tool's `params`, not a JSON-Schema engine:
/// the top level must be `type = "object"`, `properties` values must be
/// objects, and the serialized schema must fit the per-tool byte cap that
/// bounds its cost in the frozen prompt prefix.
fn validate_params_schema(params: &Value) -> Result<(), String> {
    match params.get("type").and_then(Value::as_str) {
        Some("object") => {}
        Some(other) => return Err(format!("params.type must be \"object\", not \"{other}\"")),
        None => return Err("params.type = \"object\" is required".into()),
    }
    if let Some(properties) = params.get("properties") {
        let Some(map) = properties.as_object() else {
            return Err("params.properties must be an object".into());
        };
        for (key, schema) in map {
            if !schema.is_object() {
                return Err(format!("params.properties.{key} must be an object"));
            }
        }
    }
    let bytes = params.to_string().len();
    if bytes > MAX_EXTERNAL_PARAMS_BYTES {
        return Err(format!(
            "params schema is {bytes} bytes; the cap is {MAX_EXTERNAL_PARAMS_BYTES} because every request pays for these bytes - shrink the schema or split the tool"
        ));
    }
    Ok(())
}

/// Run one external tool: spawn the command in the project root, hand it the
/// call's JSON arguments on stdin, and treat stdout as the result. Same
/// output caps and spill-to-file behavior as bash. One process per call,
/// nothing stays resident.
#[allow(clippy::too_many_arguments)]
async fn spawn_external(
    name: &str,
    tool: &ExternalTool,
    args: &Value,
    data_dir: &Path,
    root: &Path,
    caps: tools::OutputCaps,
    cancel: Arc<CancelToken>,
    sandbox: Option<execution::SandboxPolicy>,
) -> ToolOutcome {
    let request = ProcessRequest {
        program: tool.command.clone().into(),
        args: tool.args.iter().cloned().map(Into::into).collect(),
        cwd: root.to_path_buf(),
        stdin: StdinMode::json_line(args),
        timeout: std::time::Duration::from_secs(tool.timeout_secs),
        capture: CaptureSpec {
            head_bytes: 0,
            tail_bytes: caps.command_bytes,
            spill_dir: Some(data_dir.join("cmd-logs")),
            spill_bytes_per_stream: 16 * 1024 * 1024,
        },
        sandbox,
        env_allowlist: Some(tool.env.clone()),
    };

    match execution::run_process(request, cancel).await {
        Err(ProcessError::Spawn(e)) => ToolOutcome::err(format!(
                "failed to start external tool '{name}' (command '{}', defined in {}): {e}",
                tool.command,
                tool.source_path.display()
        )),
        Err(ProcessError::Wait(e)) => {
            ToolOutcome::err(format!("external tool '{name}' failed: {e}"))
        }
        // Only sandboxed probe runs can see this; session-path calls pass
        // sandbox: None. The caller (doctor.rs) words the refusal.
        Err(e @ ProcessError::SandboxUnavailable(_)) => ToolOutcome::err(e.to_string()),
        Ok(output) => match &output.termination {
            Termination::Cancelled => ToolOutcome::from_killed_process(
                format!("external tool '{name}' cancelled by user"),
                &output,
            ),
            // Same contract as bash: the tail is already captured when the
            // timeout fires, and it is the only clue to what the tool hung on.
            Termination::TimedOut => {
                let (text, truncated) = tools::render_process_output(&output, caps.command_bytes);
                ToolOutcome::from_process(
                    false,
                    format!(
                        "external tool '{name}' timed out after {}s; output until the kill:\n{text}",
                        tool.timeout_secs
                    ),
                    &output,
                    truncated,
                )
            }
            Termination::Exited(status) => {
                let (text, truncated) = tools::render_process_output(&output, caps.command_bytes);
                let (ok, text) = match status.success() {
                    true => (true, text),
                    false => (false, format!("{}\n{text}", tools::describe_exit(status))),
                };
                ToolOutcome::from_process(ok, text, &output, truncated)
            }
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::CancelToken;

    fn no_cancel() -> Arc<CancelToken> {
        Arc::new(CancelToken::default())
    }

    #[test]
    fn builtin_only_schemas_are_byte_identical_to_static() {
        let registry = Registry::builtin_only();
        // Byte identity, not just structural: this array is part of the
        // prompt-cache prefix.
        assert_eq!(
            registry.tool_schemas_json().to_string(),
            tools::tool_schemas().to_string()
        );
    }

    /// Approval prompts and the headless refusal line are built from this
    /// summary, so it must never come back empty: "declining danger ()" tells
    /// a human nothing about what they are being asked to approve.
    #[test]
    fn external_summaries_never_come_back_empty() {
        use serde_json::json;
        assert_eq!(summarize_call("deploy", &json!({"command": "ship"})), "ship");
        assert_eq!(summarize_call("deploy", &json!({"target": "prod"})), "prod");
        // No string argument at all falls back to the tool's own name.
        assert_eq!(summarize_call("danger", &json!({"count": 3})), "danger");
        assert_eq!(summarize_call("danger", &json!({})), "danger");
        assert_eq!(summarize_call("danger", &Value::Null), "danger");
    }

    #[test]
    fn tool_schemas_wire_matches_json_string() {
        let registry = Registry::builtin_only();
        assert_eq!(
            registry.tool_schemas_wire(),
            registry.tool_schemas_json().to_string()
        );
    }

    #[test]
    fn build_with_no_config_matches_builtin_only() {
        let dir = std::env::temp_dir().join(format!("omx-reg-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let registry = Registry::build(&dir.join("data"), &dir);
        assert_eq!(
            registry.tool_schemas_json().to_string(),
            Registry::builtin_only().tool_schemas_json().to_string()
        );
        assert!(registry.skills.is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn global_capabilities_come_from_the_session_data_dir_not_the_home_dir() {
        // Approvals, sessions and trust all live in `data_dir`. Discovering
        // capabilities from `$HOME` instead meant a tool could be found in one
        // place and its approval recorded in another, and it made the suite
        // fail for anyone who installed a global tool the documented way.
        let root = std::env::temp_dir().join(format!("omx-dd-{}", uuid::Uuid::new_v4()));
        let data_dir = root.join("data");
        let other_dir = root.join("elsewhere");
        let project = root.join("project");
        for dir in [data_dir.join("tools"), other_dir.join("tools"), project.clone()] {
            std::fs::create_dir_all(dir).unwrap();
        }
        let manifest = |name: &str| {
            format!("name = \"{name}\"\ndescription = \"probe\"\ncommand = \"/bin/echo\"\n")
        };
        std::fs::write(data_dir.join("tools").join("mine.toml"), manifest("from_data_dir")).unwrap();
        std::fs::write(other_dir.join("tools").join("theirs.toml"), manifest("from_elsewhere"))
            .unwrap();

        let names = Registry::build(&data_dir, &project).tool_names();
        assert!(
            names.iter().any(|n| n == "from_data_dir"),
            "a tool under the session's own data dir must be found: {names:?}"
        );
        assert!(
            !names.iter().any(|n| n == "from_elsewhere"),
            "a tool under a different data dir must NOT leak in: {names:?}"
        );

        // The same directory decides the fingerprint, or a refreeze would miss
        // a change to the very files it just loaded.
        let before = extensions_fingerprint(&data_dir, &project);
        std::fs::write(data_dir.join("tools").join("mine.toml"), manifest("renamed")).unwrap();
        assert_ne!(
            before,
            extensions_fingerprint(&data_dir, &project),
            "an edit under the data dir must move the fingerprint"
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    /// A skill sorted past MAX_SKILLS is dropped from the index but never
    /// silently: the omission count survives the snapshot into the registry.
    #[test]
    fn skills_beyond_cap_are_counted_not_silent() {
        let project = temp_dir("skill-cap");
        for i in 0..(crate::skills::MAX_SKILLS + 3) {
            let dir = project.join(".agents/skills").join(format!("s{i:03}"));
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(
                dir.join("SKILL.md"),
                format!("---\nname: skill-{i:03}\ndescription: d\n---\nbody\n"),
            )
            .unwrap();
        }
        let registry = Registry::build(&project.join("data"), &project);
        assert_eq!(registry.skills.len(), crate::skills::MAX_SKILLS);
        assert_eq!(registry.skills_omitted, 3);
        let _ = std::fs::remove_dir_all(project);
    }

    #[test]
    fn oversized_params_schema_is_rejected_not_truncated() {
        let mut properties = String::new();
        for i in 0..200 {
            properties.push_str(&format!(
                "[params.properties.field_{i:03}]\ntype = \"string\"\ndescription = \"a reasonably long description that pads the schema out\"\n"
            ));
        }
        let text = format!(
            "name = \"big\"\ndescription = \"d\"\ncommand = \"/bin/true\"\n\n[params]\ntype = \"object\"\n{properties}"
        );
        let err = parse_tool_source(Path::new("big.toml"), &text).unwrap_err();
        assert!(err.contains("bytes"), "{err}");
        assert!(err.contains(&MAX_EXTERNAL_PARAMS_BYTES.to_string()), "{err}");
    }

    #[test]
    fn params_structural_validation_rejects_wrong_shapes() {
        let missing_type = "name = \"t\"\ndescription = \"d\"\ncommand = \"/bin/true\"\n\n[params]\n[params.properties.a]\ntype = \"string\"\n";
        let err = parse_tool_source(Path::new("t.toml"), missing_type).unwrap_err();
        assert!(err.contains("params.type"), "{err}");

        let wrong_type = "name = \"t\"\ndescription = \"d\"\ncommand = \"/bin/true\"\n\n[params]\ntype = \"array\"\n";
        let err = parse_tool_source(Path::new("t.toml"), wrong_type).unwrap_err();
        assert!(err.contains("\"array\""), "{err}");

        let bad_property = "name = \"t\"\ndescription = \"d\"\ncommand = \"/bin/true\"\n\n[params]\ntype = \"object\"\n[params.properties]\na = \"string\"\n";
        let err = parse_tool_source(Path::new("t.toml"), bad_property).unwrap_err();
        assert!(err.contains("params.properties.a"), "{err}");
    }

    /// A tool sorted past MAX_EXTERNAL_TOOLS never loads, but the count
    /// survives so the prompt trailer can say what is missing.
    #[test]
    fn external_tools_beyond_cap_are_counted_not_hidden() {
        let project = temp_dir("toolcap");
        let dir = project.join(".openmax").join("tools");
        std::fs::create_dir_all(&dir).unwrap();
        for i in 0..(MAX_EXTERNAL_TOOLS + 2) {
            std::fs::write(
                dir.join(format!("tool-{i:03}.toml")),
                format!("name = \"tool-{i:03}\"\ndescription = \"d\"\ncommand = \"/bin/true\"\n"),
            )
            .unwrap();
        }
        let snapshot = capture_extensions(&project.join("data"), &project);
        assert!(snapshot.external.len() <= MAX_EXTERNAL_TOOLS);
        assert!(snapshot.tools_omitted >= 2, "{}", snapshot.tools_omitted);
        // The sorted head loads: the lexicographically first names survive.
        assert!(snapshot.external.iter().any(|t| t.name == "tool-000"));
        assert!(!snapshot.external.iter().any(|t| t.name == format!(
            "tool-{:03}",
            MAX_EXTERNAL_TOOLS + 1
        )));
        let _ = std::fs::remove_dir_all(project);
    }

    #[test]
    fn schemas_are_deterministic_across_builds() {
        let a = Registry::builtin_only();
        let b = Registry::builtin_only();
        assert_eq!(a.tool_schemas_json().to_string(), b.tool_schemas_json().to_string());
        assert_eq!(a.tool_names(), b.tool_names());
    }

    /// A manifest that parses cleanly but omits a required key is still a
    /// hard error, and it must say why in terms of the key. It used to be
    /// announced as `invalid TOML: TOML parse error at line 1, column 1`
    /// with carets under a line the author wrote correctly, which reads as
    /// an instruction to go edit that line.
    #[test]
    fn a_missing_tool_field_is_named_not_blamed_on_a_line() {
        let text = "name = \"deploy\"\ncommand = \"./deploy.sh\"\n";
        let err = parse_tool_source(Path::new("deploy.toml"), text).unwrap_err();
        assert!(err.contains("missing required field 'description'"), "{err}");
        assert!(err.contains("name, description, command"), "{err}");
        assert!(!err.contains("invalid TOML"), "{err}");
        assert!(!err.contains("line 1, column 1"), "{err}");
        assert!(!err.contains('^'), "{err}");

        // A genuine syntax error keeps the location and the caret: there the
        // file really is malformed, at that column.
        let err = parse_tool_source(Path::new("deploy.toml"), "name = [not toml").unwrap_err();
        assert!(err.contains("invalid TOML"), "{err}");
        assert!(err.contains("line 1, column 9"), "{err}");
        assert!(err.contains('^'), "{err}");

        // So does a key of the wrong type: the value it points at is the
        // value to fix.
        let wrong_type = "name = 3\ndescription = \"d\"\ncommand = \"c\"\n";
        let err = parse_tool_source(Path::new("deploy.toml"), wrong_type).unwrap_err();
        assert!(err.contains("invalid TOML"), "{err}");
        assert!(err.contains("invalid type"), "{err}");
        assert!(err.contains('^'), "{err}");
    }

    /// A collision whose WINNER falls past the skill cap must not be
    /// recorded as indexed: neither file made the prompt, and the receipt
    /// wording branches on this flag (Greptile).
    #[test]
    fn a_collision_winner_past_the_cap_is_not_marked_indexed() {
        let dir = std::env::temp_dir().join(format!("omx-shadowcap-{}", uuid::Uuid::new_v4()));
        let data = dir.join("data");
        let project = dir.join("project");
        for i in 0..crate::skills::MAX_SKILLS {
            let d = project.join(".agents/skills").join(format!("s{i:02}"));
            std::fs::create_dir_all(&d).unwrap();
            std::fs::write(
                d.join("SKILL.md"),
                format!("---\nname: s{i:02}\ndescription: filler\n---\nB.\n"),
            )
            .unwrap();
        }
        for d in ["zz-a", "zz-b"] {
            let s = project.join(".agents/skills").join(d);
            std::fs::create_dir_all(&s).unwrap();
            std::fs::write(
                s.join("SKILL.md"),
                "---\nname: zz-common\ndescription: d\n---\nB.\n",
            )
            .unwrap();
        }
        let snap = capture_extensions(&data, &project);
        assert_eq!(snap.shadowed_skills.len(), 1, "{:?}", snap.shadowed_skills);
        let (name, _, winner, winner_indexed) = &snap.shadowed_skills[0];
        assert_eq!(name, "zz-common");
        assert!(winner.ends_with("zz-b/SKILL.md"), "{winner:?}");
        assert!(!*winner_indexed, "the cap dropped the winner; it is not indexed");
        let _ = std::fs::remove_dir_all(dir);
    }

    /// A collision winner inside the 50-skill list can still lose its LINE
    /// to the 3000-byte index budget (first-fit at render); the flag must
    /// follow the render's own inclusion decision, or the receipt claims an
    /// index line the prompt does not carry (Greptile).
    #[test]
    fn a_collision_winner_past_the_byte_budget_is_not_marked_indexed() {
        let dir = std::env::temp_dir().join(format!("omx-shadowbb-{}", uuid::Uuid::new_v4()));
        let data = dir.join("data");
        let project = dir.join("project");
        let long = "x".repeat(200);
        for i in 0..14 {
            let s = project.join(".agents/skills").join(format!("a{i:02}"));
            std::fs::create_dir_all(&s).unwrap();
            std::fs::write(
                s.join("SKILL.md"),
                format!("---\nname: a{i:02}\ndescription: {long}\n---\nB.\n"),
            )
            .unwrap();
        }
        for d in ["zz-a", "zz-b"] {
            let s = project.join(".agents/skills").join(d);
            std::fs::create_dir_all(&s).unwrap();
            std::fs::write(
                s.join("SKILL.md"),
                format!("---\nname: zz-common\ndescription: {long}\n---\nB.\n"),
            )
            .unwrap();
        }
        let snap = capture_extensions(&data, &project);
        assert_eq!(snap.shadowed_skills.len(), 1, "{:?}", snap.shadowed_skills);
        let (name, _, winner, winner_indexed) = snap.shadowed_skills[0].clone();
        assert_eq!(name, "zz-common");
        // Premise: the winner survives the 50-skill list but its line does
        // not fit the byte budget.
        let registry = Registry::from_snapshot(snap);
        assert!(registry.skills.iter().any(|s| s.name == "zz-common"), "within the 50");
        let costs = crate::prompt::skill_index_costs(&project, &registry.skills);
        assert_eq!(
            costs.iter().find(|(n, _)| n == "zz-common").map(|(_, c)| *c),
            Some(0),
            "the byte budget drops the line: {costs:?}"
        );
        assert!(winner.ends_with("zz-b/SKILL.md"), "{winner:?}");
        assert!(!winner_indexed, "a budget-dropped line is not indexed");
        let _ = std::fs::remove_dir_all(dir);
    }

    /// A project definition taking a name MOOTS the losing tier's collision
    /// record: two global namesakes collide, the project override wins by
    /// precedence, and keeping the global record made the receipt call the
    /// name unindexed while the project skill is active (Greptile).
    #[test]
    fn a_project_override_moots_a_global_collision_record() {
        let dir = std::env::temp_dir().join(format!("omx-shadowx-{}", uuid::Uuid::new_v4()));
        let data = dir.join("data");
        let project = dir.join("project");
        for d in ["aa", "zz"] {
            let s = data.join("skills").join(d);
            std::fs::create_dir_all(&s).unwrap();
            std::fs::write(
                s.join("SKILL.md"),
                "---\nname: common\ndescription: global\n---\nB.\n",
            )
            .unwrap();
        }
        let s = project.join(".agents/skills/common");
        std::fs::create_dir_all(&s).unwrap();
        std::fs::write(s.join("SKILL.md"), "---\nname: common\ndescription: project\n---\nB.\n")
            .unwrap();
        let snap = capture_extensions(&data, &project);
        assert!(
            snap.shadowed_skills.is_empty(),
            "precedence moots the losing tier's collision: {:?}",
            snap.shadowed_skills
        );
        let registry = Registry::from_snapshot(snap);
        let skill = registry.skills.iter().find(|s| s.name == "common").expect("indexed");
        assert!(skill.path.starts_with(&project), "the project definition is the live one");
        let _ = std::fs::remove_dir_all(dir);
    }

    /// Three namesakes coalesce to ONE record: the final winner plus every
    /// path it displaced. Recording the intermediate winner as its own entry
    /// let the receipt call it cap-dropped while the final entry said the
    /// name IS indexed, a contradiction about one name (Greptile).
    #[test]
    fn three_namesakes_coalesce_to_the_final_winner() {
        let dir = std::env::temp_dir().join(format!("omx-shadow3-{}", uuid::Uuid::new_v4()));
        let data = dir.join("data");
        let project = dir.join("project");
        for d in ["aa", "mm", "zz"] {
            let s = project.join(".agents/skills").join(d);
            std::fs::create_dir_all(&s).unwrap();
            std::fs::write(
                s.join("SKILL.md"),
                "---\nname: common\ndescription: d\n---\nB.\n",
            )
            .unwrap();
        }
        let snap = capture_extensions(&data, &project);
        assert_eq!(snap.shadowed_skills.len(), 1, "one record per name: {:?}", snap.shadowed_skills);
        let (name, displaced, winner, winner_indexed) = &snap.shadowed_skills[0];
        assert_eq!(name, "common");
        assert_eq!(displaced.len(), 2, "{displaced:?}");
        assert!(displaced[0].ends_with("aa/SKILL.md"), "{displaced:?}");
        assert!(displaced[1].ends_with("mm/SKILL.md"), "{displaced:?}");
        assert!(winner.ends_with("zz/SKILL.md"), "{winner:?}");
        assert!(*winner_indexed);
        let _ = std::fs::remove_dir_all(dir);
    }

    /// Two files in ONE directory declaring the same skill name collapse to
    /// whichever sorts last; that accident is recorded for the receipt.
    /// Cross-tier shadowing (a project skill over a global namesake) is
    /// deliberate precedence and is NOT recorded.
    #[test]
    fn a_same_tier_skill_namesake_is_recorded_and_precedence_is_not() {
        let dir = std::env::temp_dir().join(format!("omx-shadow-{}", uuid::Uuid::new_v4()));
        let data = dir.join("data");
        let project = dir.join("project");
        for (d, body) in [
            ("aa-review", "---\nname: code-review\ndescription: first\n---\nA.\n"),
            ("zz-review", "---\nname: code-review\ndescription: second\n---\nB.\n"),
        ] {
            let skill = project.join(".agents/skills").join(d);
            std::fs::create_dir_all(&skill).unwrap();
            std::fs::write(skill.join("SKILL.md"), body).unwrap();
        }
        let snap = capture_extensions(&data, &project);
        assert_eq!(snap.shadowed_skills.len(), 1, "{:?}", snap.shadowed_skills);
        let (name, displaced, winner, winner_indexed) = &snap.shadowed_skills[0];
        assert_eq!(name, "code-review");
        assert_eq!(displaced.len(), 1);
        assert!(displaced[0].ends_with("aa-review/SKILL.md"), "{displaced:?}");
        assert!(winner.ends_with("zz-review/SKILL.md"), "{winner:?}");
        assert!(*winner_indexed, "under the cap, the winner is in the index");

        // The same name split across TIERS is precedence, not a collision.
        let dir2 = std::env::temp_dir().join(format!("omx-shadow2-{}", uuid::Uuid::new_v4()));
        let data2 = dir2.join("data");
        let project2 = dir2.join("project");
        for (root, d) in [(data2.join("skills"), "review"), (project2.join(".agents/skills"), "review")] {
            let skill = root.join(d);
            std::fs::create_dir_all(&skill).unwrap();
            std::fs::write(
                skill.join("SKILL.md"),
                "---\nname: code-review\ndescription: d\n---\nBody.\n",
            )
            .unwrap();
        }
        let snap2 = capture_extensions(&data2, &project2);
        assert!(snap2.shadowed_skills.is_empty(), "{:?}", snap2.shadowed_skills);
        let _ = std::fs::remove_dir_all(dir);
        let _ = std::fs::remove_dir_all(dir2);
    }

    /// A caller of `todo_scan` means the broken `todo-scan.toml` that
    /// DECLARES that name; the unknown-tool error must link them by the
    /// registry's own key (declared name, stem fallback), not by stem alone,
    /// or the model concludes the file was never written (round-7 audit).
    #[tokio::test]
    async fn a_broken_manifest_is_named_when_its_declared_name_differs_from_its_stem() {
        let dir = std::env::temp_dir().join(format!("omx-f9-{}", uuid::Uuid::new_v4()));
        let data = dir.join("data");
        let project = dir.join("project");
        std::fs::create_dir_all(project.join(".openmax/tools")).unwrap();
        // Hyphen file, underscore tool: parses as TOML, fails the spec.
        std::fs::write(
            project.join(".openmax/tools/todo-scan.toml"),
            "name = \"todo_scan\"\ndescription = \"d\"\n",
        )
        .unwrap();
        let registry = Registry::build(&data, &project);
        assert!(registry.get("todo_scan").is_none());
        let out = registry
            .execute(
                "todo_scan",
                &serde_json::json!({}),
                &data,
                &project,
                crate::tools::OutputCaps::default(),
                std::sync::Arc::new(crate::state::CancelToken::default()),
            )
            .await;
        assert!(!out.ok);
        assert!(out.output.contains("did NOT load"), "{}", out.output);
        assert!(out.output.contains("todo-scan.toml"), "{}", out.output);
        let _ = std::fs::remove_dir_all(dir);
    }

    /// A same-name collision between a broken file and a loaded definition
    /// must not lie in either direction: a broken PROJECT override withholds
    /// the global fallback (running different code than the user configured,
    /// silently, is the failure this surface exists to prevent), while a
    /// broken GLOBAL under a valid project override leaves the override
    /// callable and says so.
    #[test]
    fn a_broken_project_override_withholds_the_global_fallback() {
        let dir = std::env::temp_dir().join(format!("openmax-collide-{}", uuid::Uuid::new_v4()));
        let data = dir.join("data");
        let project = dir.join("project");
        std::fs::create_dir_all(data.join("tools")).unwrap();
        std::fs::create_dir_all(project.join(".openmax/tools")).unwrap();
        let valid = "name = \"wordcount\"\ndescription = \"d\"\ncommand = \"/bin/echo\"\n";
        let broken = "name = \"wordcount\"\ndescription = \"d\"\n";

        // Broken project override + valid global: withheld, and the reason
        // says so instead of claiming the name is not callable elsewhere.
        std::fs::write(data.join("tools/wordcount.toml"), valid).unwrap();
        std::fs::write(project.join(".openmax/tools/wordcount.toml"), broken).unwrap();
        let registry = Registry::build(&data, &project);
        assert!(registry.get("wordcount").is_none(), "the global fallback must be withheld");
        let (_, reason) = registry
            .broken
            .iter()
            .find(|(p, _)| p.starts_with(&project))
            .expect("the broken override is recorded");
        assert!(reason.contains("withheld"), "{reason}");

        // Reverse: broken global + valid project override stays callable.
        std::fs::write(data.join("tools/wordcount.toml"), broken).unwrap();
        std::fs::write(project.join(".openmax/tools/wordcount.toml"), valid).unwrap();
        let registry = Registry::build(&data, &project);
        assert!(registry.get("wordcount").is_some(), "the valid override must stay callable");
        let (_, reason) = registry
            .broken
            .iter()
            .find(|(p, _)| p.starts_with(&data))
            .expect("the broken global is recorded");
        assert!(reason.contains("higher-precedence"), "{reason}");

        // Tools are keyed by DECLARED name, not stem: a broken project file
        // named bar.toml that declares name = "wordcount" still collides
        // with the valid global wordcount, and the fallback is withheld.
        std::fs::remove_file(project.join(".openmax/tools/wordcount.toml")).unwrap();
        std::fs::write(data.join("tools/wordcount.toml"), valid).unwrap();
        std::fs::write(project.join(".openmax/tools/bar.toml"), broken).unwrap();
        let registry = Registry::build(&data, &project);
        assert!(
            registry.get("wordcount").is_none(),
            "a broken override under a different filename must still withhold the fallback"
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    /// The env allowlist is enforced at spawn: declared names arrive,
    /// undeclared parent env - API keys included - does not. Both variables
    /// are set on the test process, so the only difference is the manifest.
    #[tokio::test]
    async fn an_external_tool_receives_only_its_declared_env() {
        let dir = std::env::temp_dir().join(format!("openmax-envtool-{}", uuid::Uuid::new_v4()));
        let project = dir.join("project");
        std::fs::create_dir_all(project.join(".openmax/tools")).unwrap();
        std::fs::write(
            project.join(".openmax/tools/envcheck.toml"),
            "name = \"envcheck\"\ndescription = \"d\"\ncommand = \"/bin/sh\"\nargs = [\"-c\", \"printf '%s|%s' \\\"$OPENMAX_TEST_KEEP\\\" \\\"$OPENMAX_TEST_DROP\\\"\"]\nenv = [\"OPENMAX_TEST_KEEP\"]\n",
        )
        .unwrap();
        std::env::set_var("OPENMAX_TEST_KEEP", "kept");
        std::env::set_var("OPENMAX_TEST_DROP", "leaked");
        let registry = Registry::build(&dir.join("data"), &project);
        let out = registry
            .execute(
                "envcheck",
                &serde_json::json!({}),
                &dir.join("data"),
                &project,
                tools::OutputCaps::default(),
                Arc::new(CancelToken::default()),
            )
            .await;
        std::env::remove_var("OPENMAX_TEST_KEEP");
        std::env::remove_var("OPENMAX_TEST_DROP");
        assert!(out.ok, "{}", out.output);
        assert_eq!(out.output.trim(), "kept|", "declared env arrives; the rest is scrubbed");
    }

    #[test]
    fn env_names_are_validated_and_capped() {
        let path = Path::new("/p/.openmax/tools/t.toml");
        let bad = "name = \"t\"\ndescription = \"d\"\ncommand = \"/bin/echo\"\nenv = [\"BAD-NAME\"]\n";
        let err = parse_tool_source(path, bad).unwrap_err();
        assert!(err.contains("invalid env var name"), "{err}");
        let many: Vec<String> = (0..17).map(|i| format!("\"VAR_{i}\"")).collect();
        let over = format!(
            "name = \"t\"\ndescription = \"d\"\ncommand = \"/bin/echo\"\nenv = [{}]\n",
            many.join(", ")
        );
        let err = parse_tool_source(path, &over).unwrap_err();
        assert!(err.contains("at most 16"), "{err}");
        // The list round-trips through the frozen manifest so a resumed
        // session keeps the same grant.
        let good = "name = \"t\"\ndescription = \"d\"\ncommand = \"/bin/echo\"\nenv = [\"GITHUB_TOKEN\"]\n";
        let spec = parse_tool_source(path, good).unwrap();
        let registry = Registry::assemble(vec![spec], Vec::new());
        let manifest = registry.to_manifest();
        let restored = Registry::from_manifest(manifest);
        match &restored.get("t").unwrap().kind {
            ToolKind::External(t) => assert_eq!(t.env, vec!["GITHUB_TOKEN"]),
            ToolKind::Builtin => panic!("external expected"),
        }
    }

    #[test]
    fn unknown_tool_file_key_is_rejected() {
        // `mutatng` is the typo that matters: silently defaulting `mutating`
        // to false would take the tool out of the approval gate.
        let text = r#"
name = "deploy"
description = "ship it"
command = "./deploy.sh"
mutatng = true
"#;
        let err = parse_tool_source(Path::new("deploy.toml"), text).unwrap_err();
        assert!(err.contains("mutatng"), "{err}");
    }

    #[test]
    fn seven_builtins() {
        let names = Registry::builtin_only().tool_names();
        assert_eq!(
            names,
            vec![
                "list_dir",
                "read_file",
                "write_file",
                "edit_file",
                "glob",
                "grep",
                "bash",
            ]
        );
        assert!(!names.iter().any(|n| n == "task"));
    }

    #[test]
    fn builtin_lookups_match_tools_module() {
        let registry = Registry::builtin_only();
        assert_eq!(
            registry.tool_names(),
            tools::TOOL_NAMES.iter().map(|s| s.to_string()).collect::<Vec<_>>()
        );
        for name in tools::TOOL_NAMES {
            assert_eq!(registry.is_mutating(name), tools::is_mutating(name), "{name}");
        }
        assert!(!registry.is_mutating("nope"));
    }

    #[tokio::test]
    async fn unknown_tool_error_lists_names() {
        let registry = Registry::builtin_only();
        let out = registry
            .execute("nope", &serde_json::json!({}), Path::new("/nonexistent-openmax-data"), Path::new("."), tools::OutputCaps::default(), no_cancel())
            .await;
        assert!(!out.ok);
        assert!(out.output.contains("bash"), "should list valid tools: {}", out.output);
    }

    // ---------- external tools ----------

    fn temp_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("omx-{tag}-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn write_tool(dir: &Path, file: &str, body: &str) {
        std::fs::write(dir.join(file), body).unwrap();
    }

    fn write_script(dir: &Path, file: &str, body: &str) -> PathBuf {
        use std::os::unix::fs::PermissionsExt;
        let path = dir.join(file);
        std::fs::write(&path, body).unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
        path
    }

    fn registry_from(global: &Path, project: &Path) -> Registry {
        Registry::assemble(
            discover_external_in(&[global.to_path_buf(), project.to_path_buf()]),
            Vec::new(),
        )
    }

    #[test]
    fn project_tool_wins_over_global_on_collision() {
        let global = temp_dir("glob");
        let project = temp_dir("proj");
        write_tool(&global, "hello.toml", "name = \"hello\"\ndescription = \"global\"\ncommand = \"/bin/false\"\n");
        write_tool(&project, "hello.toml", "name = \"hello\"\ndescription = \"project\"\ncommand = \"/bin/true\"\n");
        let registry = registry_from(&global, &project);
        let spec = registry.get("hello").expect("hello discovered");
        assert_eq!(spec.description, "project");
        match &spec.kind {
            ToolKind::External(t) => assert_eq!(t.command, "/bin/true"),
            _ => panic!("expected external"),
        }
        let _ = std::fs::remove_dir_all(global);
        let _ = std::fs::remove_dir_all(project);
    }

    #[test]
    fn externals_sort_after_builtins_and_schemas_stay_deterministic() {
        let global = temp_dir("glob");
        let project = temp_dir("proj");
        write_tool(&project, "zz.toml", "name = \"zz_tool\"\ndescription = \"z\"\ncommand = \"/bin/true\"\n");
        write_tool(&project, "aa.toml", "name = \"aa_tool\"\ndescription = \"a\"\ncommand = \"/bin/true\"\n");
        let a = registry_from(&global, &project);
        let b = registry_from(&global, &project);
        let names = a.tool_names();
        let builtin_count = tools::TOOL_NAMES.len();
        assert_eq!(&names[..builtin_count], tools::TOOL_NAMES);
        assert_eq!(&names[builtin_count..], &["aa_tool", "zz_tool"]);
        assert_eq!(
            a.tool_schemas_json().to_string(),
            b.tool_schemas_json().to_string(),
            "schema serialization must be deterministic"
        );
        let _ = std::fs::remove_dir_all(global);
        let _ = std::fs::remove_dir_all(project);
    }

    #[test]
    fn snapshot_activates_the_same_generation_it_fingerprinted() {
        let project = temp_dir("snapshot");
        let tools_dir = project.join(".openmax").join("tools");
        std::fs::create_dir_all(&tools_dir).unwrap();
        let path = tools_dir.join("generation.toml");
        std::fs::write(
            &path,
            "name = \"generation\"\ndescription = \"first\"\ncommand = \"/bin/true\"\n",
        )
        .unwrap();

        let snapshot = capture_extensions(&project.join("data"), &project);
        let first_fingerprint = snapshot.fingerprint();
        std::fs::write(
            &path,
            "name = \"generation\"\ndescription = \"later\"\ncommand = \"/bin/true\"\n",
        )
        .unwrap();

        let frozen = Registry::from_snapshot(snapshot);
        assert_eq!(frozen.ext_fingerprint, first_fingerprint);
        assert_eq!(frozen.get("generation").unwrap().description, "first");

        let current = Registry::build(&project.join("data"), &project);
        assert_ne!(current.ext_fingerprint, first_fingerprint);
        assert_eq!(current.get("generation").unwrap().description, "later");
        let _ = std::fs::remove_dir_all(project);
    }

    #[test]
    fn external_description_capped_and_builtin_shadowing_rejected() {
        let global = temp_dir("glob");
        let project = temp_dir("proj");
        let long_desc = "d".repeat(500);
        write_tool(&project, "long.toml", &format!("name = \"long_tool\"\ndescription = \"{long_desc}\"\ncommand = \"/bin/true\"\n"));
        // Shadowing a built-in would silently change core behavior: dropped.
        write_tool(&project, "bash.toml", "name = \"bash\"\ndescription = \"evil\"\ncommand = \"/bin/true\"\n");
        // Malformed files are skipped, never fatal.
        write_tool(&project, "broken.toml", "name = \"broken\ncommand=");
        let registry = registry_from(&global, &project);
        let spec = registry.get("long_tool").unwrap();
        assert!(spec.description.chars().count() <= MAX_EXTERNAL_DESC_CHARS + 1);
        match &registry.get("bash").unwrap().kind {
            ToolKind::Builtin => {}
            _ => panic!("built-in bash must not be shadowed"),
        }
        assert!(registry.get("broken").is_none());
        let _ = std::fs::remove_dir_all(global);
        let _ = std::fs::remove_dir_all(project);
    }

    #[tokio::test]
    async fn external_tool_round_trips_args_over_stdin() {
        let project = temp_dir("proj");
        let tools_dir = project.join(".openmax").join("tools");
        std::fs::create_dir_all(&tools_dir).unwrap();
        let script = write_script(&project, "echo_args.sh", "#!/bin/sh\nread -r line\necho \"got: $line\"\n");
        write_tool(
            &tools_dir,
            "echo.toml",
            &format!(
                "name = \"echo_args\"\ndescription = \"echoes\"\ncommand = \"{}\"\n\n[params]\ntype = \"object\"\n[params.properties.message]\ntype = \"string\"\n",
                script.display()
            ),
        );
        let registry = Registry::assemble(discover_external_in(&[tools_dir]), Vec::new());
        let out = registry
            .execute("echo_args", &serde_json::json!({"message": "hi"}), Path::new("/nonexistent-openmax-data"), &project, tools::OutputCaps::default(), no_cancel())
            .await;
        assert!(out.ok, "{}", out.output);
        assert!(out.output.contains("got: {\"message\":\"hi\"}"), "{}", out.output);
        let _ = std::fs::remove_dir_all(project);
    }

    /// The args payload is one newline-terminated line: a tool reading it
    /// with `read -r` under `&&` (or `set -e`) must see a complete line, or
    /// correct line-oriented shell fails at EOF with the payload fully
    /// delivered.
    #[tokio::test]
    async fn strict_line_readers_get_a_terminated_payload() {
        let project = temp_dir("strictline");
        let tools_dir = project.join(".openmax").join("tools");
        std::fs::create_dir_all(&tools_dir).unwrap();
        let script =
            write_script(&project, "strict.sh", "#!/bin/sh\nread -r line && echo \"ok: $line\"\n");
        write_tool(
            &tools_dir,
            "strict.toml",
            &format!(
                "name = \"strict\"\ndescription = \"strict line reader\"\ncommand = \"{}\"\n",
                script.display()
            ),
        );
        let registry = Registry::assemble(discover_external_in(&[tools_dir]), Vec::new());
        let out = registry
            .execute("strict", &serde_json::json!({"k": "v"}), Path::new("/nonexistent-openmax-data"), &project, tools::OutputCaps::default(), no_cancel())
            .await;
        assert!(out.ok, "read -r must succeed on a terminated line: {}", out.output);
        assert!(out.output.contains("ok: {\"k\":\"v\"}"), "{}", out.output);
        let _ = std::fs::remove_dir_all(project);
    }

    /// The successful branch must carry the same process metadata the failing
    /// one does: a hook cannot be told a clipped tool was quiet.
    #[tokio::test]
    async fn successful_external_tool_reports_what_it_produced() {
        let project = temp_dir("proj");
        let tools_dir = project.join(".openmax").join("tools");
        std::fs::create_dir_all(&tools_dir).unwrap();
        let script = write_script(
            &project,
            "noisy.sh",
            "#!/bin/sh\nread -r line\nfor i in $(seq 1 2000); do echo \"noise line $i padded out a bit\"; done\n",
        );
        write_tool(
            &tools_dir,
            "noisy.toml",
            &format!(
                "name = \"noisy\"\ndescription = \"prints a lot\"\ncommand = \"{}\"\n\n[params]\ntype = \"object\"\n",
                script.display()
            ),
        );
        let registry = Registry::assemble(discover_external_in(&[tools_dir]), Vec::new());
        let out = registry
            .execute("noisy", &serde_json::json!({}), Path::new("/nonexistent-openmax-data"), &project, tools::OutputCaps::default(), no_cancel())
            .await;

        assert!(out.ok, "{}", out.output);
        let produced = out.process_bytes.expect("a successful external tool reports its size");
        assert!(produced > out.output.len() as u64, "the result is a bounded rendering");
        assert!(out.process_truncated);
        let _ = std::fs::remove_dir_all(project);
    }

    #[tokio::test]
    async fn external_tool_timeout_and_failure_shape() {
        let project = temp_dir("proj");
        let tools_dir = project.join(".openmax").join("tools");
        std::fs::create_dir_all(&tools_dir).unwrap();
        // The timeout must outlast script startup by a wide margin, or a
        // loaded runner kills the script before `echo` runs and there is
        // legitimately no tail to report.
        let slow = write_script(&project, "slow.sh", "#!/bin/sh\necho started-then-hung\nsleep 30\n");
        let fail = write_script(&project, "fail.sh", "#!/bin/sh\necho oops >&2\nexit 3\n");
        write_tool(&tools_dir, "slow.toml", &format!("name = \"slow\"\ndescription = \"s\"\ncommand = \"{}\"\ntimeout_secs = 3\n", slow.display()));
        write_tool(&tools_dir, "fail.toml", &format!("name = \"fail\"\ndescription = \"f\"\ncommand = \"{}\"\n", fail.display()));
        let registry = Registry::assemble(discover_external_in(std::slice::from_ref(&tools_dir)), Vec::new());

        let out = registry.execute("slow", &serde_json::json!({}), Path::new("/nonexistent-openmax-data"), &project, tools::OutputCaps::default(), no_cancel()).await;
        assert!(!out.ok);
        assert!(out.output.contains("timed out after 3s"), "{}", out.output);
        assert!(
            out.output.contains("started-then-hung"),
            "the captured tail must survive the kill: {}",
            out.output
        );

        let out = registry.execute("fail", &serde_json::json!({}), Path::new("/nonexistent-openmax-data"), &project, tools::OutputCaps::default(), no_cancel()).await;
        assert!(!out.ok);
        assert!(out.output.starts_with("exit code 3"), "{}", out.output);
        assert!(out.output.contains("[stderr]") && out.output.contains("oops"), "{}", out.output);

        #[cfg(unix)]
        {
            let sig = write_script(&project, "sig.sh", "#!/bin/sh\nkill -9 $$\n");
            write_tool(&tools_dir, "sig.toml", &format!("name = \"sig\"\ndescription = \"g\"\ncommand = \"{}\"\n", sig.display()));
            let registry = Registry::assemble(discover_external_in(std::slice::from_ref(&tools_dir)), Vec::new());
            let out = registry.execute("sig", &serde_json::json!({}), Path::new("/nonexistent-openmax-data"), &project, tools::OutputCaps::default(), no_cancel()).await;
            assert!(!out.ok);
            assert!(
                out.output.starts_with("killed by signal 9 (SIGKILL)"),
                "a signal kill is named, not faked as an exit code: {}",
                out.output
            );
        }

        let out = registry.execute("missing_binary", &serde_json::json!({}), Path::new("/nonexistent-openmax-data"), &project, tools::OutputCaps::default(), no_cancel()).await;
        assert!(!out.ok && out.output.contains("unknown tool"));
        let _ = std::fs::remove_dir_all(project);
    }

    #[tokio::test]
    async fn external_tool_spawn_failure_is_actionable() {
        let project = temp_dir("proj");
        let tools_dir = project.join(".openmax").join("tools");
        std::fs::create_dir_all(&tools_dir).unwrap();
        write_tool(&tools_dir, "ghost.toml", "name = \"ghost\"\ndescription = \"g\"\ncommand = \"/nonexistent/binary\"\n");
        let registry = Registry::assemble(discover_external_in(std::slice::from_ref(&tools_dir)), Vec::new());
        let out = registry.execute("ghost", &serde_json::json!({}), Path::new("/nonexistent-openmax-data"), &project, tools::OutputCaps::default(), no_cancel()).await;
        assert!(!out.ok);
        assert!(out.output.contains("ghost") && out.output.contains("/nonexistent/binary"), "{}", out.output);
        assert!(out.output.contains("ghost.toml"), "must point at the defining file: {}", out.output);
        let _ = std::fs::remove_dir_all(project);
    }
}
