import { BUN_CHILD, bunCommand } from "./bun-child";

const INSTALL_TIMEOUT = 15_000;

export function spawnPackedInstall(
    command: string[],
    cwd: string,
    label: string,
    timeout = INSTALL_TIMEOUT,
): string {
    const result = Bun.spawnSync(command, {
        ...BUN_CHILD,
        cwd,
        stdout: "pipe",
        stderr: "pipe",
        timeout,
        killSignal: "SIGKILL",
    });
    const output = `${result.stdout.toString()}${result.stderr.toString()}`;
    if (result.exitedDueToTimeout) {
        throw new Error(
            `${label} timed out after ${timeout} ms (last 8192 bytes per stream):\n` +
                `stdout:\n${result.stdout.subarray(-8192).toString()}\n` +
                `stderr:\n${result.stderr.subarray(-8192).toString()}`,
        );
    }
    if (result.exitCode !== 0) throw new Error(`${label} failed:\n${output}`);
    return output;
}

export function packedInstall(args: string[], cwd: string, label: string): string {
    return spawnPackedInstall(bunCommand([args[0]!, "--verbose", ...args.slice(1)]), cwd, label);
}
