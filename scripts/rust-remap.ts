import { spawnSync } from "node:child_process";
import { homedir } from "node:os";
import { join, resolve } from "node:path";

/** Stable source prefixes for rust-src and Cargo registry files embedded in WASM panic locations. */
export function rustSourceRemaps(toolchain?: string): string[] {
    const args = toolchain ? [`+${toolchain}`, "--print", "sysroot"] : ["--print", "sysroot"];
    const result = spawnSync("rustc", args, {
        cwd: resolve(import.meta.dir, ".."),
        encoding: "utf8",
    });
    if (result.status !== 0)
        throw new Error(`rustc ${args.join(" ")} failed while resolving the rust-src path`);

    const sysroot = result.stdout.trim();
    const rustSrc = resolve(sysroot, "lib/rustlib/src/rust");
    const cargoHome = resolve(process.env.CARGO_HOME ?? join(homedir(), ".cargo"));
    const registrySrc = resolve(cargoHome, "registry/src");
    return [
        `--remap-path-prefix=${rustSrc}=/rust-src`,
        `--remap-path-prefix=${registrySrc}=/cargo/registry/src`,
    ];
}
