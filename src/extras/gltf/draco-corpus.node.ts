import { expect, setDefaultTimeout, test } from "bun:test";
import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import inputs from "../../../crates/draco/tests/records/inputs.json";
import { syncFiles } from "../../../examples/gltf/sync";
import { type DracoType, decodeDraco } from "./draco";

setDefaultTimeout(TIMEOUT.node);

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
