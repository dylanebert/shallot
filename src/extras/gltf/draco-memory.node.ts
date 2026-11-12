import { expect, setDefaultTimeout, test } from "bun:test";
import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import { TIMEOUT } from "@dylanebert/shallot/testing";

setDefaultTimeout(TIMEOUT.node);

const root = resolve(import.meta.dir, "../../..");
const rows = async (name: string) =>
    (await readFile(resolve(root, `crates/draco/tests/records/${name}.tsv`), "utf8"))
        .trim()
        .split("\n")
        .filter((line) => !line.startsWith("#"))
        .map((line) => line.split("\t"));

test("an input past the 2 GiB memory cap is refused by name, and the kernel stays usable", async () => {
    const bytes = await readFile(resolve(root, "crates/draco/tests/fixtures/standard-closed.drc"));
    const [, faces, , , , , , points] = (await rows("faces")).find(
        (row) => row[0] === "fixtures/standard-closed",
    )!;
    type Exports = { alloc(length: number): number; release(ptr: number, length: number): void };
    const Instance = WebAssembly.Instance;
    let kernel: Exports | undefined;
    WebAssembly.Instance = class extends Instance {
        constructor(...args: ConstructorParameters<typeof Instance>) {
            super(...args);
            kernel = this.exports as unknown as Exports;
        }
    };
    try {
        const entry: typeof import("./draco") = await import(`./draco.ts?cap=${Date.now()}`);
        const decode = () => entry.decodeDraco(bytes, Number(faces), Number(points), []);
        expect(decode().faces.length).toBe(Number(faces) * 3);
        // Hold every block the allocator gives, halving the size, until even one byte fails.
        const held: [number, number][] = [];
        for (let size = 2 ** 28; size >= 1; ) {
            const pointer = kernel!.alloc(size);
            if (pointer) {
                expect(pointer).toBeGreaterThan(0);
                held.push([pointer, size]);
            } else size /= 2;
        }
        expect(() => decode()).toThrow("Draco input allocation failed");
        for (const [pointer, size] of held) kernel!.release(pointer, size);
        expect(decode().faces.length).toBe(Number(faces) * 3);
    } finally {
        WebAssembly.Instance = Instance;
    }
});
