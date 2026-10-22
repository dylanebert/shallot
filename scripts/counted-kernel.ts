import { resolve } from "node:path";

const ROOT = resolve(import.meta.dir, "..");
// Its own target directory, so the feature build and the plain build never replace each other's artifact.
const TARGET = resolve(ROOT, "target/count-allocations");

/**
 * The physics kernel's wasm bytes built with the `count-allocations` feature, which exports
 * `allocationCount` and `allocationControl`. Cargo rebuilds only what changed since the last call.
 */
export async function countedKernelBytes(): Promise<ArrayBuffer> {
    const build = Bun.spawnSync(
        [
            "cargo",
            "build",
            "--frozen",
            "-p",
            "shallot-physics",
            "--release",
            "--target",
            "wasm32-unknown-unknown",
            "--features",
            "count-allocations",
            "--target-dir",
            TARGET,
        ],
        { cwd: ROOT },
    );
    if (build.exitCode !== 0)
        throw new Error(
            `${build.stderr.toString()}\nCounted kernel requires fetched crates. Run cargo fetch --locked in ${ROOT}.`,
        );
    return Bun.file(
        resolve(TARGET, "wasm32-unknown-unknown/release/shallot_physics.wasm"),
    ).arrayBuffer();
}
