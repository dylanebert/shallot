import { expect, test } from "bun:test";
import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";
import inputs from "../../../crates/draco/tests/records/inputs.json";
import { syncFiles } from "../../../examples/gltf/sync";
import { type DracoType, decodeDraco } from "./draco";

const root = resolve(import.meta.dir, "../../..");
const types: DracoType[] = [
    "int8",
    "uint8",
    "int16",
    "uint16",
    "int32",
    "uint32",
    "int64",
    "uint64",
    "float32",
    "float64",
    "bool",
];
const sha256 = (bytes: ArrayBufferView) =>
    createHash("sha256")
        .update(new Uint8Array(bytes.buffer, bytes.byteOffset, bytes.byteLength))
        .digest("hex");
const rows = async (name: string) =>
    (await readFile(resolve(root, `crates/draco/tests/records/${name}.tsv`), "utf8"))
        .trim()
        .split("\n")
        .filter((line) => !line.startsWith("#"))
        .map((line) => line.split("\t"));

test("JavaScript entry decodes every recorded stream equal to its record", async () => {
    const assets = resolve(root, "test-results/draco-assets");
    await syncFiles(
        inputs.files,
        process.env.GLTF_SAMPLE_MODELS
            ? pathToFileURL(`${resolve(process.env.GLTF_SAMPLE_MODELS)}/`)
            : new URL(
                  `https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/${inputs.commit}/Models/`,
              ),
        assets,
    );
    const ranges = new Map(inputs.streams.map((stream) => [stream.name, stream]));
    const attributes = await rows("attributes");
    let outputs = 0;
    let streams = 0;
    let refusals = 0;
    for (const [name, faces, length, faceHash, , inputHash, , points] of await rows("faces")) {
        const stream = ranges.get(name);
        const bytes = stream
            ? (await readFile(resolve(assets, stream.file))).subarray(
                  stream.offset,
                  stream.offset + stream.length,
              )
            : await readFile(resolve(root, `crates/draco/tests/${name}.drc`));
        expect(sha256(bytes), `${name} input`).toBe(inputHash);
        const own = attributes.filter((row) => row[0] === name);
        const request = (row: string[]) => ({
            id: Number(row[2]),
            type: types[Number(row[7]) - 1],
        });
        const accepted = own.filter((row) => row[8] !== "refused");
        const decoded = decodeDraco(bytes, Number(faces), Number(points), accepted.map(request));
        expect([decoded.faces.byteLength, sha256(decoded.faces)], `${name} faces`).toEqual([
            Number(length),
            faceHash,
        ]);
        for (const [index, row] of accepted.entries()) {
            const { components, data } = decoded.attributes[index];
            expect(
                [components, data.byteLength, sha256(data)],
                `${name} attribute ${row[1]} as ${request(row).type}`,
            ).toEqual([Number(row[5]), Number(row[8]), row[9]]);
            outputs++;
        }
        for (const row of own.filter((row) => row[8] === "refused")) {
            expect(
                () => decodeDraco(bytes, Number(faces), Number(points), [request(row)]),
                `${name} attribute ${row[1]} as ${request(row).type}`,
            ).toThrow(/Draco/);
            refusals++;
        }
        streams++;
    }
    expect(streams).toBe(394);
    expect(outputs + refusals).toBe(attributes.length);
    console.log(`${streams} streams, ${outputs} attribute outputs, ${refusals} refusals`);
});

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
