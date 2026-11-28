//! What an interactive run's commands left running.
//!
//! Each tool call's process group is stopped when the call returns, but a
//! process that starts a session of its own (`setsid`, a dev server that
//! daemonizes) is out of the harness's reach by design. Such a process keeps
//! the environment it inherited, so the run marks its own environment with a
//! value no other run shares, every `bash` call and hook inherits it, and a
//! process that still carries it once the run's tools are stopped is one the
//! run's commands started and left running. This only reports: nothing here
//! signals a process.

use std::time::Duration;

/// The variable the run's environment carries, read back from each process.
const MARKER: &str = "OPENMAX_RUN";

/// Marks this run's environment, which every command it starts inherits:
/// the pid and the time it started, so no later run reads another's mark.
/// Called while this is the only thread, as `set_var` needs.
pub(crate) fn mark_run() {
    let start = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_nanos());
    std::env::set_var(MARKER, format!("{}-{start}", std::process::id()));
}

/// A process still running with this run's mark.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Left {
    pub pid: u32,
    pub name: String,
    /// The TCP ports it listens on, where they can be read (Linux and macOS).
    pub ports: Vec<u16>,
}

/// The running processes, other than this one, whose environment holds this
/// run's mark, by pid. Reads only what a process of the same user may read,
/// so a process that runs as another user, or cleared its environment, is
/// not among them.
pub(crate) fn still_running() -> Vec<Left> {
    std::env::var_os(MARKER).map_or_else(Vec::new, |tag| marked(std::os::unix::ffi::OsStrExt::as_bytes(tag.as_os_str())))
}

/// The running processes, other than this one, marked with `tag`.
fn marked(tag: &[u8]) -> Vec<Left> {
    let entry = [MARKER.as_bytes(), b"=", tag].concat();
    let mut left = platform::scan(&entry, std::process::id());
    // A process names itself: its name reaches the shell and the transcript
    // as text, never as a line break or an escape sequence.
    for p in &mut left {
        p.name = open_max_core::text::one_line(&p.name);
    }
    left.sort_by_key(|p| p.pid);
    left
}

/// What [`still_running`] finds twice, `settle` apart, so that a process the
/// run's exit had just killed and that was still on its way out is not
/// reported. With nothing found the first time, no wait.
pub(crate) fn left_after_exit(settle: Duration) -> Vec<Left> {
    let first = still_running();
    if first.is_empty() {
        return first;
    }
    std::thread::sleep(settle);
    still_running().into_iter().filter(|p| first.iter().any(|f| f.pid == p.pid)).collect()
}

/// One process as a line's text: its name, then the ports it listens on.
pub(crate) fn describe(p: &Left) -> String {
    let ports: Vec<String> = p.ports.iter().map(u16::to_string).collect();
    match ports.is_empty() {
        true => p.name.clone(),
        false => format!("{}, listening on TCP {}", p.name, ports.join(", ")),
    }
}

/// The command that stops them all.
pub(crate) fn stop_command(left: &[Left]) -> String {
    let pids: Vec<String> = left.iter().map(|p| p.pid.to_string()).collect();
    format!("kill {}", pids.join(" "))
}

/// The report printed on the shell once the session has ended: nothing when
/// nothing is left.
pub(crate) fn exit_report(left: &[Left]) -> Vec<String> {
    if left.is_empty() {
        return Vec::new();
    }
    let mut lines = vec!["openmax: still running, started by this run's commands:".to_string()];
    lines.extend(left.iter().map(|p| format!("  pid {}  {}", p.pid, describe(p))));
    lines.push(format!("openmax: to stop them: {}", stop_command(left)));
    lines
}

#[cfg(target_os = "linux")]
mod platform {
    use super::Left;
    use std::path::Path;

    pub(super) fn scan(entry: &[u8], own: u32) -> Vec<Left> {
        let entries = std::fs::read_dir("/proc").into_iter().flatten().flatten();
        entries
            .filter_map(|e| {
                let pid: u32 = e.file_name().to_str()?.parse().ok()?;
                let dir = e.path();
                let environ = std::fs::read(dir.join("environ")).ok()?;
                if pid == own || !environ.split(|&b| b == 0).any(|var| var == entry) {
                    return None;
                }
                let stat = String::from_utf8_lossy(&std::fs::read(dir.join("stat")).ok()?).into_owned();
                let (head, rest) = stat.rsplit_once(')')?;
                let name = head.split_once(" (")?.1.to_string();
                // Exited and not yet waited for, or on its way out: it runs nothing.
                if matches!(rest.split_whitespace().next(), Some("Z" | "X" | "x")) {
                    return None;
                }
                Some(Left { pid, name, ports: listening_ports(&dir) })
            })
            .collect()
    }

    /// The TCP ports the process at `dir` holds a listening socket on: the
    /// sockets `net/tcp` and `net/tcp6` of its network namespace list in
    /// state 0A (listen), matched by inode to the sockets among its open
    /// descriptors.
    fn listening_ports(dir: &Path) -> Vec<u16> {
        let mut listening = std::collections::HashMap::new();
        for table in ["net/tcp", "net/tcp6"] {
            let text = std::fs::read_to_string(dir.join(table)).unwrap_or_default();
            for line in text.lines().skip(1) {
                let f: Vec<&str> = line.split_whitespace().collect();
                let (Some(local), Some(&"0A"), Some(inode)) = (f.get(1), f.get(3), f.get(9)) else { continue };
                let port = local.rsplit(':').next().and_then(|p| u16::from_str_radix(p, 16).ok());
                if let (Some(port), Ok(inode)) = (port, inode.parse::<u64>()) {
                    listening.insert(inode, port);
                }
            }
        }
        let fds = std::fs::read_dir(dir.join("fd")).into_iter().flatten().flatten();
        let mut ports: Vec<u16> = fds
            .filter_map(|fd| {
                let link = std::fs::read_link(fd.path()).ok()?;
                let inode = link.to_str()?.strip_prefix("socket:[")?.strip_suffix(']')?.parse::<u64>().ok()?;
                listening.get(&inode).copied()
            })
            .collect();
        ports.sort_unstable();
        ports.dedup();
        ports
    }
}

/// macOS: the process list, each process's environment and its sockets come
/// from libproc and sysctl. Of those calls the binary binds only
/// `proc_pidinfo` at launch; the rest are looked up with `dlsym` here, when a
/// session ends or `/status` asks, so the loader binds no more symbols at
/// every launch (see the import cap in docs/development.md).
#[cfg(target_os = "macos")]
mod platform {
    use super::Left;
    use libc::{c_int, c_uint, c_void, size_t};

    type ListAllPids = unsafe extern "C" fn(*mut c_void, c_int) -> c_int;
    type Sysctl = unsafe extern "C" fn(*mut c_int, c_uint, *mut c_void, *mut size_t, *mut c_void, size_t) -> c_int;
    type PidFdInfo = unsafe extern "C" fn(c_int, c_int, c_int, *mut c_void, c_int) -> c_int;

    /// `RTLD_DEFAULT` in `<dlfcn.h>`: search every image the process loaded.
    const RTLD_DEFAULT: *mut c_void = -2isize as *mut c_void;
    /// `PROC_PIDFDSOCKETINFO` in `<sys/proc_info.h>`.
    const PROC_PIDFDSOCKETINFO: c_int = 3;
    /// `sizeof(struct socket_fdinfo)` and the offsets read from it, as the
    /// macOS SDK's `<sys/proc_info.h>` lays it out. A reply of another size
    /// is another layout, and its ports are not read.
    const SOCKET_FDINFO: usize = 792;
    const SOI_FAMILY: usize = 184;
    const SOI_KIND: usize = 256;
    const TCPSI_LPORT: usize = 268;
    const TCPSI_STATE: usize = 344;
    /// `SOCKINFO_TCP` and `TSI_S_LISTEN`.
    const SOCKINFO_TCP: i32 = 2;
    const TSI_S_LISTEN: i32 = 1;

    fn symbol<T>(name: &std::ffi::CStr) -> Option<T> {
        // SAFETY: dlsym only reads the name; a null result is no symbol.
        let found = unsafe { libc::dlsym(RTLD_DEFAULT, name.as_ptr()) };
        // SAFETY: each caller names a libSystem function whose C signature
        // is the fn pointer type `T` it asks for.
        (!found.is_null()).then(|| unsafe { std::mem::transmute_copy::<*mut c_void, T>(&found) })
    }

    pub(super) fn scan(entry: &[u8], own: u32) -> Vec<Left> {
        let (Some(list), Some(sysctl)) = (symbol::<ListAllPids>(c"proc_listallpids"), symbol::<Sysctl>(c"sysctl")) else {
            return Vec::new();
        };
        let fdinfo = symbol::<PidFdInfo>(c"proc_pidfdinfo");
        // SAFETY: a null buffer asks only for the count.
        let count = unsafe { list(std::ptr::null_mut(), 0) }.max(0) as usize;
        let mut pids = vec![0 as c_int; count + 64];
        let bytes = (pids.len() * std::mem::size_of::<c_int>()) as c_int;
        // SAFETY: the buffer holds `bytes` bytes.
        let listed = unsafe { list(pids.as_mut_ptr().cast(), bytes) }.max(0) as usize;
        pids.truncate(listed.min(pids.len()));
        let argmax = arg_max(sysctl);
        pids.into_iter()
            .filter_map(|pid| u32::try_from(pid).ok().filter(|&p| p != 0 && p != own))
            .filter_map(|pid| {
                if !environment(sysctl, pid, argmax)?.iter().any(|var| var.as_slice() == entry) {
                    return None;
                }
                let name = running_name(pid)?;
                let ports = fdinfo.map(|info| listening_ports(info, pid)).unwrap_or_default();
                Some(Left { pid, name, ports })
            })
            .collect()
    }

    fn arg_max(sysctl: Sysctl) -> usize {
        let mut mib = [libc::CTL_KERN, libc::KERN_ARGMAX];
        let (mut value, mut size) = (0 as c_int, std::mem::size_of::<c_int>());
        // SAFETY: `value` holds `size` bytes.
        let read = unsafe { sysctl(mib.as_mut_ptr(), 2, (&raw mut value).cast(), &mut size, std::ptr::null_mut(), 0) };
        if read == 0 && value > 0 { value as usize } else { 1 << 20 }
    }

    /// The environment `pid` started with, from `KERN_PROCARGS2`: argc, the
    /// executable's path, padding, argc arguments, then the environment up
    /// to an empty string. None for a process of another user, or one that
    /// has exited.
    fn environment(sysctl: Sysctl, pid: u32, argmax: usize) -> Option<Vec<Vec<u8>>> {
        let mut mib = [libc::CTL_KERN, libc::KERN_PROCARGS2, pid as c_int];
        let (mut buf, mut size) = (vec![0u8; argmax], argmax);
        // SAFETY: `buf` holds `size` bytes.
        let read = unsafe { sysctl(mib.as_mut_ptr(), 3, buf.as_mut_ptr().cast(), &mut size, std::ptr::null_mut(), 0) };
        if read != 0 || size < 4 {
            return None;
        }
        let argc = i32::from_ne_bytes(buf[..4].try_into().ok()?).max(0) as usize;
        let mut strings = buf[4..size.min(buf.len())].split(|&b| b == 0);
        strings.next()?;
        let mut strings = strings.skip_while(|s| s.is_empty()).skip(argc);
        Some(strings.by_ref().take_while(|s| !s.is_empty()).map(<[u8]>::to_vec).collect())
    }

    /// The name of `pid` while it runs, from `proc_pidinfo`'s BSD info;
    /// None once it has exited.
    fn running_name(pid: u32) -> Option<String> {
        // SAFETY: proc_bsdinfo is plain data, valid zeroed.
        let mut info: libc::proc_bsdinfo = unsafe { std::mem::zeroed() };
        let size = std::mem::size_of::<libc::proc_bsdinfo>() as c_int;
        // SAFETY: `info` holds `size` bytes.
        let read = unsafe { libc::proc_pidinfo(pid as c_int, libc::PROC_PIDTBSDINFO, 0, (&raw mut info).cast(), size) };
        if read != size || info.pbi_status == libc::SZOMB {
            return None;
        }
        let text = |chars: &[libc::c_char]| {
            let bytes: Vec<u8> = chars.iter().take_while(|&&c| c != 0).map(|&c| c as u8).collect();
            String::from_utf8_lossy(&bytes).into_owned()
        };
        let name = text(&info.pbi_name);
        Some(if name.is_empty() { text(&info.pbi_comm) } else { name })
    }

    /// The TCP ports `pid` holds a listening socket on, from each socket
    /// descriptor's `socket_fdinfo`.
    fn listening_ports(fdinfo: PidFdInfo, pid: u32) -> Vec<u16> {
        // SAFETY: a null buffer asks only for the size the list needs.
        let need = unsafe { libc::proc_pidinfo(pid as c_int, libc::PROC_PIDLISTFDS, 0, std::ptr::null_mut(), 0) };
        let mut fds = vec![libc::proc_fdinfo { proc_fd: 0, proc_fdtype: 0 }; need.max(0) as usize / 8 + 16];
        let bytes = (fds.len() * std::mem::size_of::<libc::proc_fdinfo>()) as c_int;
        // SAFETY: the buffer holds `bytes` bytes.
        let read = unsafe { libc::proc_pidinfo(pid as c_int, libc::PROC_PIDLISTFDS, 0, fds.as_mut_ptr().cast(), bytes) };
        fds.truncate((read.max(0) as usize / std::mem::size_of::<libc::proc_fdinfo>()).min(fds.len()));
        let mut ports: Vec<u16> = fds
            .iter()
            .filter(|fd| fd.proc_fdtype == libc::PROX_FDTYPE_SOCKET as u32)
            .filter_map(|fd| {
                let mut info = [0u64; SOCKET_FDINFO / 8];
                // SAFETY: `info` holds SOCKET_FDINFO bytes.
                let read = unsafe { fdinfo(pid as c_int, fd.proc_fd, PROC_PIDFDSOCKETINFO, info.as_mut_ptr().cast(), SOCKET_FDINFO as c_int) };
                let bytes: Vec<u8> = info.iter().flat_map(|w| w.to_ne_bytes()).collect();
                let int = |at: usize| i32::from_ne_bytes(bytes[at..at + 4].try_into().unwrap());
                let tcp = matches!(int(SOI_FAMILY), libc::AF_INET | libc::AF_INET6) && int(SOI_KIND) == SOCKINFO_TCP;
                (read as usize == SOCKET_FDINFO && tcp && int(TCPSI_STATE) == TSI_S_LISTEN)
                    .then(|| u16::from_be(int(TCPSI_LPORT) as u16))
            })
            .collect();
        ports.sort_unstable();
        ports.dedup();
        ports
    }
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
mod platform {
    pub(super) fn scan(_: &[u8], _: u32) -> Vec<super::Left> {
        Vec::new()
    }
}

