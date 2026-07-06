import { existsSync } from "node:fs";
import { join } from "node:path";
import { decodeDraco, loadDraco } from "../src/extras/gltf/draco";
import { isGlb, parseGlb } from "../src/extras/gltf/glb";
import { type GltfJson, type GltfScene, parse } from "../src/extras/gltf/gltf";
import { decodeMeshopt, loadMeshopt } from "../src/extras/gltf/meshopt";

// the shared corpus walk for the glTF conformance suite (roadmap "glTF import — conformance + regression
// suite"). The CPU half of `loadGltf` — fetch → glb-split → resolve buffers → inject the Draco codec → `parse`
// — run over the Khronos glTF-Sample-Assets corpus. `parse` is deviceless (no GPU, no State; KTX2 transcode
// is GPU-side, so a KTX variant parses on the CPU like any other), so the walk rides `bun test` and the
// matrix generator alike — both import THIS, so the test and the report can't drift. The corpus is a git
// submodule outside the distributed package, so everything here is presence-gated; nothing reaches into
// `src/extras/gltf` beyond its public parse surface.

/** the corpus, symlinked nowhere — read straight from the `reference/gltf-sample-assets` submodule. Absent
 *  on a fresh clone that didn't init submodules; {@link corpusPresent} gates every consumer. */
export const CORPUS = join(import.meta.dir, "../../../../reference/gltf-sample-assets/Models");

/** true when the corpus submodule is checked out — the loud-skip gate for the test + the generator. */
export function corpusPresent(): boolean {
    return existsSync(join(CORPUS, "model-index.json"));
}

/** one `model-index.json` entry — the corpus's own catalog of each model + its packed-asset variants. */
export interface CorpusModel {
    label: string;
    name: string;
    tags?: string[];
    /** variant name (also the subdirectory) → the entry file within it (`Foo.gltf` / `Foo.glb`). */
    variants: Record<string, string>;
}

export type Status = "supported" | "partial" | "unsupported";

/** the pinned outcome for one (model, variant): the skipped-feature keys, the decoded geometry count, and a
 *  status DERIVED from them — re-derived on read so a hand-edit that desyncs status from data fails loud. */
export interface MatrixEntry {
    unsupported: string[];
    meshes: number;
    status: Status;
}

/** model → variant → {@link MatrixEntry}, the committed `gltf-matrix.json`. */
export type Matrix = Record<string, Record<string, MatrixEntry>>;

/** one walked (model, variant): the parsed scene, or the parse error that a healthy importer must not hit. */
export interface CorpusEntry {
    model: string;
    variant: string;
    scene?: GltfScene;
    error?: string;
}

export async function corpusModels(): Promise<CorpusModel[]> {
    return JSON.parse(await Bun.file(join(CORPUS, "model-index.json")).text());
}

// the variants worth parsing: the base geometry path plus every codec the importer claims to handle (Draco,
// KTX, Meshopt — meshopt rides KHR_mesh_quantization, dequantized in readFloats). WEBP / JPG paths stay out of
// scope (the importer doesn't decode them and they'd parse as garbage geometry, not a clean skip), so the
// matrix stays focused on what the importer actually contracts.
export function pickVariants(model: CorpusModel): string[] {
    const keys = Object.keys(model.variants);
    const picked = new Set<string>();
    // both base paths when present (external .bin AND the .glb container — the most-shipped format), else the
    // self-contained embedded fallback
    for (const k of ["glTF", "glTF-Binary"]) if (keys.includes(k)) picked.add(k);
    if (picked.size === 0 && keys.includes("glTF-Embedded")) picked.add("glTF-Embedded");
    for (const k of keys)
        if (k.includes("Draco") || k.includes("KTX") || k.includes("Meshopt")) picked.add(k);
    return [...picked];
}

// resolve one glTF buffer to its bytes — the .glb BIN chunk (no uri), a base64 data-URI, or a file next to the
// .gltf. Mirrors index.ts `resolveBuffer`; reads from disk so the walk stays deviceless (no fetch, no GPU).
async function resolveBuffer(
    buffer: { uri?: string; byteLength: number; extensions?: Record<string, unknown> },
    dir: string,
    bin?: ArrayBuffer,
): Promise<ArrayBuffer> {
    const uri = buffer.uri;
    if (!uri) {
        if (bin) return bin;
        // a meshopt fallback buffer carries no bytes (the compressed bufferViews redirect to the source); the
        // missing uri is expected, zero-fill it — mirrors index.ts resolveBuffer
        if (
            buffer.extensions?.EXT_meshopt_compression ||
            buffer.extensions?.KHR_meshopt_compression
        )
            return new ArrayBuffer(buffer.byteLength);
        throw new Error("[gltf] buffer has no uri and no .glb BIN chunk");
    }
    if (uri.startsWith("data:")) {
        const buf = Buffer.from(uri.slice(uri.indexOf(",") + 1), "base64");
        return buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
    }
    return Bun.file(join(dir, decodeURIComponent(uri))).arrayBuffer();
}

/** parse one (model, variant) through the importer's deviceless path. Mirrors `loadGltf`'s CPU half exactly —
 *  glb-split, buffer resolve, and the Draco / meshopt codecs injected only when the asset carries them. */
export async function parseVariant(model: CorpusModel, variant: string): Promise<GltfScene> {
    const dir = join(CORPUS, model.name, variant);
    const bytes = await Bun.file(join(dir, model.variants[variant])).arrayBuffer();
    const { json, bin } = isGlb(bytes)
        ? parseGlb(bytes)
        : { json: JSON.parse(new TextDecoder().decode(bytes)) as GltfJson, bin: undefined };
    const buffers = await Promise.all((json.buffers ?? []).map((b) => resolveBuffer(b, dir, bin)));
    const needsDraco = (json.meshes ?? []).some((m) =>
        m.primitives.some((p) => p.extensions?.KHR_draco_mesh_compression),
    );
    let draco: typeof decodeDraco | undefined;
    if (needsDraco) {
        await loadDraco();
        draco = decodeDraco;
    }
    const needsMeshopt = (json.bufferViews ?? []).some(
        (bv) => bv.extensions?.EXT_meshopt_compression || bv.extensions?.KHR_meshopt_compression,
    );
    let meshopt: typeof decodeMeshopt | undefined;
    if (needsMeshopt) {
        await loadMeshopt();
        meshopt = decodeMeshopt;
    }
    return parse(json, buffers, draco, meshopt);
}

/** walk every (model, representative-variant) in the corpus, isolating a parse failure to its own entry so one
 *  bad model surfaces as a red row rather than aborting the sweep. */
export async function walkCorpus(): Promise<CorpusEntry[]> {
    const models = await corpusModels();
    const out: CorpusEntry[] = [];
    for (const model of models) {
        for (const variant of pickVariants(model)) {
            try {
                out.push({ model: model.name, variant, scene: await parseVariant(model, variant) });
            } catch (e) {
                out.push({ model: model.name, variant, error: String(e) });
            }
        }
    }
    return out;
}

/** the sorted, stable feature keys a scene skipped (drops the count-bearing `detail` strings that would churn). */
export function features(scene: GltfScene): string[] {
    return scene.unsupported.map((u) => u.feature).sort();
}

/** the derived status — the single rule both the matrix and the test compute, never authored by hand. */
export function status(scene: GltfScene): Status {
    if (scene.meshes.length === 0) return "unsupported";
    return scene.unsupported.length === 0 ? "supported" : "partial";
}

/** the pinned outcome for one scene. */
export function entryOf(scene: GltfScene): MatrixEntry {
    return { unsupported: features(scene), meshes: scene.meshes.length, status: status(scene) };
}
