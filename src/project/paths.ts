/**
 * `path` with every separator as `/`, as Vite's `normalizePath` spells a path where it enters: a `relative()`
 * result, a Glob or recursive `readdir` entry, or a tool's output. Unlike Vite's, it replaces `\` on every
 * host, so a Windows-shaped path normalizes the same on macOS, where tests inject `path.win32`; Shallot's
 * tooling never meets a POSIX file name containing `\`.
 */
export function normalizePath(path: string): string {
    return path.replaceAll("\\", "/");
}
