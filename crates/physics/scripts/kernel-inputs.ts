import { createHash } from "node:crypto";
import { readdirSync, readFileSync } from "node:fs";
import { resolve } from "node:path";

export const KERNEL_ARTIFACTS = [
    "src/standard/physics/kernel/kernel.wasm.ts",
    "src/standard/physics/kernel/kernel.shared.wasm.ts",
];
export const KERNEL_BUILD_COMMAND = "bun run crates/physics/scripts/build-kernel.ts";

/** Fingerprints tree inputs, not machine-local toolchains or build outputs. */
export function kernelInputHash(root: string): string {
    const files = [
        "Cargo.toml",
        "Cargo.lock",
        "rust-toolchain.toml",
        "package.json",
        "bun.lock",
        "scripts/wasm-opt.ts",
        "crates/physics/Cargo.toml",
        "crates/physics/.cargo/config.toml",
        "crates/physics/scripts/build-kernel.ts",
        "crates/physics/scripts/kernel-inputs.ts",
    ];
    function sources(dir: string): void {
        for (const entry of readdirSync(resolve(root, dir), { withFileTypes: true })) {
            const path = `${dir}/${entry.name}`;
            if (entry.isDirectory()) sources(path);
            else files.push(path);
        }
    }
    sources("crates/physics/src");
    const hash = createHash("sha256");
    for (const path of files.sort()) {
        const bytes = readFileSync(resolve(root, path));
        hash.update(`${path}\0${bytes.length}\0`);
        hash.update(bytes);
    }
    return hash.digest("hex");
}
