import { BUN_CHILD, bunCommand } from "./bun-child";

const child = Bun.spawn(bunCommand(["test", "gpu.test", ...process.argv.slice(2)]), {
    ...BUN_CHILD,
    argv0: process.platform === "win32" ? "bun.exe" : undefined,
    env: { ...process.env, SHALLOT_GPU_COMPILE_ORACLE: "1" },
    stdin: "inherit",
    stdout: "inherit",
    stderr: "inherit",
});
process.exit(await child.exited);
