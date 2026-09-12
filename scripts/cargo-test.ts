import { resolve } from "node:path";

const root = resolve(import.meta.dir, "..");

/** Run one compiled Cargo test population; compilation is resolved by the cargo requirement. */
export function runCargoTest(packageName: string, ...args: string[]): void {
    const command = ["cargo", "test", "-p", packageName, ...args];
    const proc = Bun.spawnSync(command, {
        cwd: root,
        stdout: "inherit",
        stderr: "inherit",
    });
    if (!proc.success) {
        throw new Error(`${command.join(" ")} exited with ${proc.exitCode ?? "unknown status"}`);
    }
}
