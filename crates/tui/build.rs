//! Link options for the `openmax` binary on macOS, where the dynamic loader's
//! work happens before `main` and so before the first frame
//! (`first_frame_holds_its_ratio_to_a_baseline_binary` in
//! crates/tui/tests/thesis.rs times it;
//! `the_binary_names_only_libsystem_caps_its_imports_and_chains_its_fixups`
//! holds what these change).
//!
//! `-dead_strip_dylibs` drops a library no symbol is bound from. The `libc`
//! crate links libiconv on Apple targets for iconv bindings nothing here
//! calls, and the loader opens every library the binary names.
//!
//! `-fixup_chains` writes the binary's pointer fixups as chained fixups
//! instead of rebase opcodes. With opcodes the loader walks every rebased
//! pointer in the constant data before `main`; chained, macOS 13 and later
//! apply them as each page is first touched, and only the pages touched
//! pay. The linker chooses this format itself only for a deployment target
//! of macOS 12 or later. Asked for it below that, as here for macOS 11, the
//! oldest aarch64 macOS, it writes the chains in DYLD_CHAINED_PTR_64, the
//! pointer format from before macOS 12's: that the macOS 11 loader reads
//! them rests on that choice of the linker's, as no launch there is tested.
//! x86_64's default deployment target, macOS 10.12, predates the format, so
//! x86_64 keeps opcodes.

fn main() {
    println!("cargo::rerun-if-changed=build.rs");
    let var = |name: &str| std::env::var(name).unwrap_or_default();
    if var("CARGO_CFG_TARGET_OS") != "macos" {
        return;
    }
    println!("cargo::rustc-link-arg-bin=openmax=-Wl,-dead_strip_dylibs");
    if var("CARGO_CFG_TARGET_ARCH") == "aarch64" {
        println!("cargo::rustc-link-arg-bin=openmax=-Wl,-fixup_chains");
    }
}
