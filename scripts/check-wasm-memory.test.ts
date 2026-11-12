import { expect, test } from "bun:test";
import { resolve } from "node:path";
import { KERNELS, kernelBytes, memoryMaximum, UNCAPPED } from "./wasm-limits";
import { WASM_MAX_MEMORY } from "./wasm-memory";

const root = resolve(import.meta.dir, "..");

test("every built kernel declares a memory maximum within the 2 GiB cap", () => {
    const over = KERNELS.flatMap((path) => {
        const maximum = memoryMaximum(kernelBytes(resolve(root, path)));
        const cap = UNCAPPED[path as keyof typeof UNCAPPED] ?? WASM_MAX_MEMORY;
        return maximum !== undefined && maximum <= cap
            ? []
            : [`${path}: memory maximum ${maximum ?? "absent"} exceeds ${cap}`];
    });
    expect(over).toEqual([]);
});
