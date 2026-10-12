import { expect, test } from "bun:test";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import {
    KERNEL_ARTIFACTS,
    KERNEL_BUILD_COMMAND,
    kernelInputHash,
} from "../crates/physics/scripts/kernel-inputs";

test("checked kernel artifacts match their build inputs", () => {
    const root = resolve(import.meta.dir, "..");
    const stamp = `// Build inputs: sha256:${kernelInputHash(root)}`;
    let failed = false;
    for (const artifact of KERNEL_ARTIFACTS) {
        const contents = readFileSync(resolve(root, artifact), "utf8");
        if (contents.split("\n", 3)[1] === stamp) continue;
        console.error(
            `${artifact}: build-input stamp missing or stale. Regenerate: ${KERNEL_BUILD_COMMAND}`,
        );
        failed = true;
    }
    expect(failed).toBe(false);
});

test("committed WASM contains no host paths", () => {
    const root = resolve(import.meta.dir, "..");
    const inlinedKernels = KERNEL_ARTIFACTS.map((artifact) => {
        const contents = readFileSync(resolve(root, artifact), "utf8");
        const base64 = contents.match(
            /export const KERNEL(?:_SHARED)?_WASM_BASE64 =\s+"([^"]+)";/,
        )?.[1];
        expect(base64).toBeDefined();
        return [artifact, Buffer.from(base64!, "base64")] as const;
    });
    const wasm = [
        [
            "crates/audio/pkg/shallot_audio.wasm",
            readFileSync(resolve(root, "crates/audio/pkg/shallot_audio.wasm")),
        ] as const,
        ...inlinedKernels,
    ];
    const forbidden = ["/Users/", "/home/", ".rustup", ".cargo"].map((path) => Buffer.from(path));
    const failures: string[] = [];
    for (const [artifact, bytes] of wasm) {
        for (const path of forbidden) {
            if (bytes.includes(path)) failures.push(`${artifact}: contains ${path.toString()}`);
        }
    }
    if (failures.length) console.error(failures.join("\n"));
    expect(failures).toEqual([]);
});
