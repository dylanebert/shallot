import { readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { run } from "./native";

const target = join(tmpdir(), "shallot-box3d-functions-wasm");
run(
    [
        "cargo",
        "build",
        "--release",
        "--target",
        "wasm32-unknown-unknown",
        "-p",
        "shallot-physics",
        "--features",
        "box3d-oracle",
        "--target-dir",
        target,
    ],
    { RUSTFLAGS: "-C target-feature=+simd128 --remap-path-prefix=crates/physics/=" },
);
export const oracleWasm = new Uint8Array(
    readFileSync(join(target, "wasm32-unknown-unknown/release/shallot_physics.wasm")),
);

export async function assertPublicOracleKernel(): Promise<void> {
    const { kernel } = await import("../../src/standard/physics/kernel/kernel");
    const k = kernel(undefined) as unknown as { box3dOracleRun?: unknown };
    if (typeof k.box3dOracleRun !== "function") {
        throw new Error(
            "public oracle requires --preload ./diagnostics/box3d-parity/oracle-preload.ts",
        );
    }
}
