import { decodeDraco, loadDraco } from "../../src/extras/gltf/draco";
import { isGlb, parseGlb } from "../../src/extras/gltf/glb";
import { type GltfJson, type GltfScene, parse } from "../../src/extras/gltf/gltf";
import { decodeMeshopt, loadMeshopt } from "../../src/extras/gltf/meshopt";
import { cachedFiles, load } from "../assets";

// the shared corpus walk for the glTF conformance generator. The CPU half of `loadGltf` — glb-split → resolve
// buffers → inject the Draco/meshopt codecs → `parse` — run over the Khronos glTF-Sample-Assets models pinned
// in `assets.json`, read from the assets cache by hash. `parse` is deviceless (no GPU, no State; KTX2
// transcode is GPU-side, so a KTX variant parses on the CPU like any other); nothing here reaches into
// `src/extras/gltf` beyond its public parse surface.

const KHRONOS = "KhronosGroup/glTF-Sample-Assets/";

/** one pinned Khronos model: each variant's `assets.json` entry and the `.gltf`/`.glb` file it opens. */
export interface CorpusModel {
    name: string;
    variants: Record<string, { asset: string; entry: string }>;
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

/** the pinned Khronos models, grouped from each entry's `dest` (`gltf-samples/<Model>/<variant>`). */
export function corpusModels(): CorpusModel[] {
    const models = new Map<string, CorpusModel>();
    for (const asset of load()) {
        if (!asset.url.includes(KHRONOS) || !asset.files) continue;
        const [, model, variant] = asset.dest.split("/");
        const entry = asset.files.find((f) => /\.(gltf|glb)$/.test(f.path))?.path;
        if (!model || !variant || !entry)
            throw new Error(
                `assets.json ${asset.name}: expected gltf-samples/<Model>/<variant> with a .gltf/.glb`,
            );
        let found = models.get(model);
        if (!found) models.set(model, (found = { name: model, variants: {} }));
        found.variants[variant] = { asset: asset.name, entry };
    }
    return [...models.values()].sort((x, y) => x.name.localeCompare(y.name));
}

/** the corpus entries absent from the assets cache; the walk refuses until this is empty. */
export function uncached(): string[] {
    return corpusModels()
        .flatMap((m) => Object.values(m.variants).map((v) => v.asset))
        .filter((name) => {
            try {
                cachedFiles(name);
                return false;
            } catch {
                return true;
            }
        });
}

// resolve one glTF buffer to its bytes — the .glb BIN chunk (no uri), a base64 data-URI, or a file next to the
// .gltf. Mirrors index.ts `resolveBuffer`; reads the cache so the walk stays deviceless (no fetch, no GPU).
async function resolveBuffer(
    buffer: { uri?: string; byteLength: number; extensions?: Record<string, unknown> },
    files: Map<string, string>,
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
    const path = files.get(decodeURIComponent(uri));
    if (!path) throw new Error(`[gltf] buffer ${uri} is not a file of the pinned asset`);
    return Bun.file(path).arrayBuffer();
}

/** parse one (model, variant) through the importer's deviceless path. Mirrors `loadGltf`'s CPU half exactly —
 *  glb-split, buffer resolve, and the Draco / meshopt codecs injected only when the asset carries them. */
export async function parseVariant(model: CorpusModel, variant: string): Promise<GltfScene> {
    const { asset, entry } = model.variants[variant];
    const files = cachedFiles(asset);
    const bytes = await Bun.file(files.get(entry)!).arrayBuffer();
    const { json, bin } = isGlb(bytes)
        ? parseGlb(bytes)
        : { json: JSON.parse(new TextDecoder().decode(bytes)) as GltfJson, bin: undefined };
    const buffers = await Promise.all(
        (json.buffers ?? []).map((b) => resolveBuffer(b, files, bin)),
    );
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
    const out: CorpusEntry[] = [];
    for (const model of corpusModels()) {
        for (const variant of Object.keys(model.variants)) {
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
