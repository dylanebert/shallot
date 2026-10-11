/**
 * The running Bun as a child process: spawn `bunCommand(args)` with `...BUN_CHILD`. Windows CI runs Bun
 * from a copy named node.exe so Dawn can load (src/engine/runtime/dawn-host.ts), and Bun named node
 * acts as Node; `argv0` keeps the child Bun.
 */
export const bunCommand = (args: string[]) => [process.execPath, ...args];
export const BUN_CHILD = { argv0: "bun" } as const;
