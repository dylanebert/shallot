import { expect, setDefaultTimeout, test } from "bun:test";
import { readdirSync, readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import manifest from "../../../examples/gltf/assets.json";
import { syncFiles } from "../../../examples/gltf/sync";
import { bunChild } from "../../../scripts/bun-child";
import { type GltfJson, type GltfMesh, type GltfScene, parseGltf } from "./gltf";
import { loadGltf, resolveGltf } from "./load";

setDefaultTimeout(TIMEOUT.node);
const variants = [
    ["Avocado", "glTF-Draco"],
    ["Box", "glTF-Draco"],
    ["BrainStem", "glTF-Draco"],
    ["CesiumMan", "glTF-Draco"],
    ["CesiumMilkTruck", "glTF-Draco"],
    ["Duck", "glTF-Draco"],
    ["Lantern", "glTF-Draco"],
    ["MorphPrimitivesTest", "glTF-Draco"],
    ["RiggedFigure", "glTF-Draco"],
    ["RiggedSimple", "glTF-Draco"],
    // Checked here, not in the picker: its KTX2 transfer functions break KHR_texture_basisu's MUSTs
    // (a data texture tagged sRGB, an sRGB red-green texture), which the importer refuses by name.
    // loadGltf fetches no images, so its geometry, the corpus's only Draco TEXCOORD_1, is compared.
    ["CarConcept", "glTF-KTX-BasisU-Draco"],
] as const;
/** Asset differences between a Draco variant and its base, not decoding differences. */
const assetMaterials: Record<string, Record<number, [number, number]>> = {
    // CarConcept's Draco variant assigns InteriorDashSides "Interior 1", the glTF "Interior 3 Carmine".
    CarConcept: { 8: [3, 6] },
};
const assets = fileURLToPath(new URL("../../../examples/gltf/public/assets/", import.meta.url));
await syncFiles(
    manifest.models
        .filter((model) => variants.some(([name]) => name === model.name))
        .flatMap((model) => model.files),
    new URL(`${manifest.commit}/`, manifest.source),
    assets,
);
// Google's shipped glTF decoder at the unit's pin, the oracle for each stream's quantization.
const pin = "554628821c3448319d7d4ae48db170b8c63e1aaf";
const oracle = fileURLToPath(new URL("../../../test-results/draco-oracle/", import.meta.url));
await syncFiles(
    [
        {
            path: "draco_wasm_wrapper_gltf.js",
            sha256: "9927d2886d34906aec5258d091a162f3d9f59ab7e84fd6a81961987bca3c363f",
        },
        {
            path: "draco_decoder_gltf.wasm",
            sha256: "39b78cec9398af525d76f7eecf336ccf829891cec9524905e55fe686499834bf",
        },
    ],
    new URL(`https://raw.githubusercontent.com/google/draco/${pin}/javascript/`),
    oracle,
);
const wrapper = { exports: {} as unknown };
// The wrapper is CommonJS; the worktree's package.json would load it as a module.
new Function(
    "module",
    "exports",
    "require",
    "__dirname",
    readFileSync(`${oracle}draco_wasm_wrapper_gltf.js`, "utf8"),
)(wrapper, wrapper.exports, require, oracle);
const draco: any = await (wrapper.exports as (options: object) => Promise<unknown>)({
    wasmBinary: readFileSync(`${oracle}draco_decoder_gltf.wasm`),
});

const read: typeof fetch = (async (url: URL | string) =>
    new Response(readFileSync(new URL(String(url))))) as typeof fetch;
const file = (model: string, variant: string) => {
    const dir = `${assets}Models/${model}/${variant}/`;
    return new URL(`file://${dir}${readdirSync(dir).find((name) => name.endsWith(".gltf"))}`);
};

/** Per primitive, each extension semantic's tolerance: its quantization step, or zero. */
function steps(url: URL): Map<string, number>[] {
    const json = JSON.parse(readFileSync(url, "utf8")) as GltfJson;
    const buffers = (json.buffers ?? []).map((buffer) => readFileSync(new URL(buffer.uri!, url)));
    const decoder = new draco.Decoder();
    for (const type of ["POSITION", "NORMAL", "COLOR", "TEX_COORD", "GENERIC"])
        decoder.SkipAttributeTransform(draco[type]);
    const result = (json.meshes ?? []).flatMap((mesh) =>
        mesh.primitives.map((prim) => {
            const def = prim.extensions!.KHR_draco_mesh_compression as {
                bufferView: number;
                attributes: Record<string, number>;
            };
            const view = json.bufferViews![def.bufferView];
            const bytes = buffers[view.buffer].subarray(
                view.byteOffset ?? 0,
                (view.byteOffset ?? 0) + view.byteLength,
            );
            const geometry = new draco.Mesh();
            if (!decoder.DecodeArrayToMesh(bytes, bytes.length, geometry).ok())
                throw new Error("oracle refused a corpus stream");
            const tolerances = new Map<string, number>();
            for (const [semantic, id] of Object.entries(def.attributes)) {
                const attribute = decoder.GetAttributeByUniqueId(geometry, id);
                const quantization = new draco.AttributeQuantizationTransform();
                const octahedron = new draco.AttributeOctahedronTransform();
                // Normals admit two octahedral steps of 2 / (2^bits - 1) in each unit-vector
                // component; the corpus reaches 1.81, while positions and coordinates stay
                // within half of their one step.
                tolerances.set(
                    semantic,
                    quantization.InitFromAttribute(attribute)
                        ? quantization.range() / (2 ** quantization.quantization_bits() - 1)
                        : octahedron.InitFromAttribute(attribute)
                          ? 4 / (2 ** octahedron.quantization_bits() - 1)
                          : 0,
                );
                draco.destroy(quantization);
                draco.destroy(octahedron);
            }
            draco.destroy(geometry);
            return tolerances;
        }),
    );
    draco.destroy(decoder);
    return result;
}

async function geometry(url: URL): Promise<GltfScene> {
    const { json, buffers } = await resolveGltf(url, read);
    return parseGltf(json, buffers);
}

/** Each vertex's channels with the semantic whose step bounds it. */
function channels(mesh: GltfMesh): [string, Float32Array, number][] {
    const v = mesh.vertices;
    const pick = (offsets: number[]) =>
        Float32Array.from({ length: (v.length / 8) * offsets.length }, (_, i) => {
            const vertex = Math.floor(i / offsets.length);
            return v[vertex * 8 + offsets[i % offsets.length]];
        });
    const list: [string, Float32Array | undefined, number][] = [
        ["POSITION", pick([0, 1, 2]), 3],
        ["TEXCOORD_0", pick([3, 7]), 2],
        ["NORMAL", pick([4, 5, 6]), 3],
        ["TEXCOORD_1", mesh.uv1, 2],
        ["COLOR_0", mesh.color, 4],
        ["JOINTS_0", mesh.joints, 4],
        ["WEIGHTS_0", mesh.weights, 4],
        ["TANGENT", mesh.tangent, 4],
        ...(mesh.morphTargets ?? []).flatMap((target, t) =>
            (["position", "normal", "tangent"] as const).map(
                (key): [string, Float32Array | undefined, number] => [
                    `target ${t} ${key}`,
                    target[key],
                    3,
                ],
            ),
        ),
    ];
    return list.filter((c): c is [string, Float32Array, number] => c[1] !== undefined);
}

/** Matches every decoded triangle to a distinct base triangle, corners in cyclic order. */
function matchTriangles(
    name: string,
    actual: GltfMesh,
    expected: GltfMesh,
    tolerance: Map<string, number>,
): void {
    const a = channels(actual);
    const b = channels(expected);
    expect(
        a.map(([semantic]) => semantic),
        `${name} channels`,
    ).toEqual(b.map(([semantic]) => semantic));
    expect(actual.indices.length, `${name} corners`).toBe(expected.indices.length);
    const step = (semantic: string) => tolerance.get(semantic) ?? 0;
    const same = (va: number, vb: number) =>
        a.every(([semantic, values, width], c) => {
            for (let k = 0; k < width; k++)
                if (
                    !(
                        Math.abs(values[va * width + k] - b[c][1][vb * width + k]) <=
                        step(semantic) * 1.0001 + 1e-7
                    )
                )
                    return false;
            return true;
        });
    const cell = Math.max(step("POSITION"), 1e-6) * 4;
    const key = (values: Float32Array, vertex: number, dx = 0, dy = 0, dz = 0) =>
        `${Math.floor(values[vertex * 3] / cell) + dx},${Math.floor(values[vertex * 3 + 1] / cell) + dy},${Math.floor(values[vertex * 3 + 2] / cell) + dz}`;
    const positions = (mesh: [string, Float32Array, number][]) => mesh[0][1];
    const grid = new Map<string, number[]>();
    for (let t = 0; t < expected.indices.length / 3; t++) {
        const k = key(positions(b), expected.indices[t * 3]);
        grid.get(k)?.push(t) ?? grid.set(k, [t]);
    }
    // Candidates per decoded triangle, then a maximum matching: a mesh may repeat a triangle
    // within one step, so the first candidate is not always the right one.
    const candidates = Array.from({ length: actual.indices.length / 3 }, (_, t) => {
        const corners = [0, 1, 2].map((c) => actual.indices[t * 3 + c]);
        const found = new Set<number>();
        for (let rotation = 0; rotation < 3; rotation++)
            for (let dx = -1; dx <= 1; dx++)
                for (let dy = -1; dy <= 1; dy++)
                    for (let dz = -1; dz <= 1; dz++)
                        for (const candidate of grid.get(
                            key(positions(a), corners[rotation], dx, dy, dz),
                        ) ?? [])
                            if (
                                [0, 1, 2].every((c) =>
                                    same(
                                        corners[(rotation + c) % 3],
                                        expected.indices[candidate * 3 + c],
                                    ),
                                )
                            )
                                found.add(candidate);
        return [...found];
    });
    const owner = new Int32Array(expected.indices.length / 3).fill(-1);
    const augment = (t: number, seen: Uint8Array): boolean => {
        for (const candidate of candidates[t]) {
            if (seen[candidate]) continue;
            seen[candidate] = 1;
            if (owner[candidate] < 0 || augment(owner[candidate], seen)) {
                owner[candidate] = t;
                return true;
            }
        }
        return false;
    };
    let unmatched = 0;
    for (const [t, list] of candidates.entries()) {
        const direct = list.find((candidate) => owner[candidate] < 0);
        if (direct !== undefined) owner[direct] = t;
        else if (!augment(t, new Uint8Array(owner.length))) unmatched++;
    }
    expect(unmatched, `${name} unmatched triangles`).toBe(0);
}

/** A scene's records without geometry; JSON drops the -0 the base assets' nodes carry. */
const records = (scene: GltfScene, materials: Record<number, unknown> = {}) =>
    JSON.parse(
        JSON.stringify({
            ...scene,
            images: scene.images.map((image) => image.uri?.replace(/^.*\/|\.\w+$/g, "")),
            meshes: scene.meshes.map((mesh, i) => ({
                name: mesh.name,
                material: i in materials ? "asset" : mesh.material,
                streams: Object.keys(mesh).sort(),
                targets: mesh.morphTargets?.length,
            })),
        }),
    );

for (const [model, variant] of variants)
    test(`${model}/${variant}: scene equals glTF but geometry; geometry matches triangle for triangle within each stream's quantization step`, async () => {
        const compressed = file(model, variant);
        const base = file(model, "glTF");
        const materials = assetMaterials[model];
        const actual = await loadGltf(compressed, read);
        const expected = await loadGltf(base, read);
        expect(records(actual, materials)).toEqual(records(expected, materials));
        for (const [i, [draco, gltf]] of Object.entries(materials ?? {})) {
            expect(actual.meshes[Number(i)].material).toBe(draco);
            expect(expected.meshes[Number(i)].material).toBe(gltf);
        }
        const tolerances = steps(compressed);
        const a = await geometry(compressed);
        const b = await geometry(base);
        expect(a.meshes.length).toBe(tolerances.length);
        for (const [i, mesh] of a.meshes.entries())
            matchTriangles(`${model} ${mesh.name}`, mesh, b.meshes[i], tolerances[i]);
    });

const decoderImported = async (url: URL) => {
    const entry = new URL("./index.ts", import.meta.url).href;
    const child = Bun.spawn({
        ...bunChild([
            "-e",
            `const {loadGltf} = await import(${JSON.stringify(entry)});
             const {readFileSync} = await import("node:fs");
             await loadGltf(${JSON.stringify(url.href)}, async (url) => new Response(readFileSync(new URL(String(url)))));
             console.log(Object.keys(require.cache).some(p => /draco(?:\\.wasm)?\\.ts$/.test(p)));`,
        ]),
        stdout: "pipe",
        stderr: "pipe",
    });
    expect(await child.exited, await new Response(child.stderr).text()).toBe(0);
    return (await new Response(child.stdout).text()).trim();
};

test("a scene with no Draco primitive imports no decoder", async () => {
    expect(await decoderImported(file("Box", "glTF"))).toBe("false");
    expect(await decoderImported(file("Box", "glTF-Draco"))).toBe("true");
});
