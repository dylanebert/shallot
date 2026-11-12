import { expect, test } from "bun:test";
import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import { type DracoType, decodeDraco } from "./draco";

const root = resolve(import.meta.dir, "../../..");
const rows = async (name: string) =>
    (await readFile(resolve(root, `crates/draco/tests/records/${name}.tsv`), "utf8"))
        .trim()
        .split("\n")
        .filter((line) => !line.startsWith("#"))
        .map((line) => line.split("\t"));

test("JavaScript entry refuses invalid requests and malformed streams by name", async () => {
    const bytes = await readFile(resolve(root, "crates/draco/tests/fixtures/standard-closed.drc"));
    const [, faces, , , , , , points] = (await rows("faces")).find(
        (row) => row[0] === "fixtures/standard-closed",
    )!;
    const [f, p] = [Number(faces), Number(points)];
    expect(() => decodeDraco(bytes, f + 1, p, [])).toThrow("Draco face count mismatch");
    expect(() => decodeDraco(bytes, -1, p, [])).toThrow("invalid Draco expected face count");
    expect(() => decodeDraco(bytes, f, 0.5, [])).toThrow("invalid Draco expected point count");
    expect(() => decodeDraco(bytes, f, p, [{ id: 99, type: "float32" }])).toThrow(
        /Draco attribute/,
    );
    expect(() => decodeDraco(bytes, f, p, [{ id: 0, type: "half" as DracoType }])).toThrow(
        "invalid Draco output data type",
    );
    expect(() => decodeDraco(bytes.subarray(0, 20), f, p, [])).toThrow(/Draco/);
    expect(() => decodeDraco(new Uint8Array(8), f, p, [])).toThrow("invalid Draco signature");
    // The kernel stays usable after each refusal.
    expect(decodeDraco(bytes, f, p, []).faces.length).toBe(f * 3);
});

test("importing the entry instantiates no WebAssembly until a stream is decoded", async () => {
    const Instance = WebAssembly.Instance;
    let instances = 0;
    WebAssembly.Instance = class extends Instance {
        constructor(...args: ConstructorParameters<typeof Instance>) {
            super(...args);
            instances++;
        }
    };
    try {
        // A fresh module instance, so no earlier test's decode has instantiated it.
        const entry: typeof import("./draco") = await import(`./draco.ts?lazy=${Date.now()}`);
        expect(instances).toBe(0);
        expect(() => entry.decodeDraco(new Uint8Array(8), 0, 0, [])).toThrow();
        expect(instances).toBe(1);
    } finally {
        WebAssembly.Instance = Instance;
    }
});
