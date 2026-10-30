import { beforeAll, expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";

setDefaultTimeout(TIMEOUT.node);

import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { assets, models, primitives } from "../../../diagnostics/mikktspace-records/inputs";
import manifest from "../../../examples/gltf/assets.json";
import { syncFiles } from "../../../examples/gltf/sync";
import records from "./fixtures/mikktspace.json";
import { generateTangents } from "./tangents";

beforeAll(async () => {
    await syncFiles(
        manifest.models
            .filter((model) => models.includes(model.name))
            .flatMap((model) => model.files),
        new URL(`${manifest.commit}/`, manifest.source),
        assets,
    );
});

for (const record of records.records)
    test(`${record.name}: every corner equals pinned C MikkTSpace bits with glTF handedness`, async () => {
        const input = primitives(record.name).find(
            (p) => p.mesh === record.mesh && p.primitive === record.primitive,
        )!;
        const result = await generateTangents({ ...input, name: record.name });
        const bytes = readFileSync(resolve(import.meta.dir, "fixtures", record.file));
        const reference = new Uint32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
        const bits = new Uint32Array(result.tangent.buffer);
        const corners = new Uint32Array(result.indices.length * 4);
        for (let i = 0; i < result.indices.length; i++)
            corners.set(bits.subarray(result.indices[i] * 4, result.indices[i] * 4 + 4), i * 4);
        expect(record.corners).toBe(result.indices.length);
        expect(corners).toEqual(reference);
        expect(result.vertices.length).toBe(input.vertices.length);
        expect(result.indices).toEqual(input.indices);
    });
