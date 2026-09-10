import { createHash } from "node:crypto";
import { existsSync, readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";

export const nativeSourceHash = "afc3b46db7b2e1b62602a146dca0749e0df5f456a3b742bca38505aa17537ee1";
export const nativePatchHash = "c1ce7d567b2e83b7e6cae33a29e507aa150c20178a77f335f45a4492b06b6bb3";
export const nativeHash = "2f573c74b9f8cb96aa86c17e7a4c47ab9b210edd8335526126e40c8d8288d86c";

/** Load only the carried acquisition repair, without executing the optional upstream peer. */
export async function loadNative(): Promise<typeof import("bun-webgpu")> {
    if (typeof Bun === "undefined") throw new Error("Shallot native setup requires Bun");
    createRequire(import.meta.url).resolve("bun-webgpu");
    const file = resolve(
        fileURLToPath(import.meta.resolve("@dylanebert/shallot/harness/browser")),
        "../native.js",
    );
    if (!existsSync(file)) {
        throw new Error(
            "Shallot native projection is missing. In a source checkout, run bun packages/shallot/scripts/tooling.ts from the repository root after install or pack. For an installed package, reinstall @dylanebert/shallot.",
        );
    }
    const hash = createHash("sha256").update(readFileSync(file)).digest("hex");
    if (hash !== nativeHash) throw new Error("Shallot native projection hash mismatch");
    return import(file);
}
