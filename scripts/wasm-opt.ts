import { existsSync } from "node:fs";
import { resolve } from "node:path";
import { $ } from "bun";

// One optimizer for every shipped WASM (audio and both physics kernels): the `binaryen` devDependency
// pins its version, so the emitted bytes don't depend on whatever wasm-opt a seat has on PATH.

const bin = resolve(import.meta.dir, "../node_modules/.bin/wasm-opt");

/** Enables the union of features the shipped modules use; enabling one a module lacks is a no-op. */
export const FLAGS = [
    "-O3",
    "--enable-simd",
    "--enable-threads",
    "--enable-bulk-memory",
    "--enable-mutable-globals",
    "--enable-nontrapping-float-to-int",
    "--enable-sign-ext",
];

/** The pinned wasm-opt's version line, for generated-file headers. */
export async function version(): Promise<string> {
    return (await $`${locate()} --version`.text()).trim();
}

function locate(): string {
    if (!existsSync(bin))
        throw new Error(`wasm-opt missing at ${bin}: run \`bun install\` (binaryen devDependency)`);
    return bin;
}

/** Optimizes `input` into `output` with the shared flag set. */
export async function optimize(input: string, output: string): Promise<void> {
    await $`${locate()} ${FLAGS} ${input} -o ${output}`;
}
