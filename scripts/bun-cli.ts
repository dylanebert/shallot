type SpawnOptions = Bun.SpawnOptions.SpawnSyncOptions<"ignore", "pipe", "pipe">;

/** Run a Bun CLI subcommand through the active runtime, even when Bun is copied as node.exe. */
export function spawnBunCli(args: string[], options: SpawnOptions) {
    return Bun.spawnSync<"ignore", "pipe", "pipe">([process.execPath, ...args], {
        ...options,
        ...(process.platform === "win32" ? { argv0: "bun.exe" } : {}),
    });
}
