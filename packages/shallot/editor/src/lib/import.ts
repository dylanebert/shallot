import type { GltfImport } from "@dylanebert/shallot";
import type { Node } from "@dylanebert/shallot/editor";
import { formatFields } from "@dylanebert/shallot/scene/core";
import { mintId } from "./bundles";

// The model-import pipeline behind the viewport drop + the Add menu's Model… picker. Pure logic lives
// here (grouping a drop into models + sidecars, minting the document nodes); the DOM half (DataTransfer
// directory traversal, upload fetches) is thin and lives beside it. App.svelte orchestrates: collect →
// group → upload → live loadGltf → mintNodes → one compound doc.add.

const MODEL_EXT = /\.(glb|gltf)$/i;

/** one dropped/picked file, path-relative to the drop root (a directory drop keeps its folder prefix). */
export interface ImportFile {
    path: string;
    bytes: ArrayBuffer;
}

/** one importable model from a drop: its root `.glb`/`.gltf` plus the sidecar files the root references
 *  (present in the drop), and the referenced names the drop is missing. */
export interface ModelGroup {
    root: ImportFile;
    sidecars: ImportFile[];
    missing: string[];
}

// the external (non-data:) uris a .gltf references — its buffers + images, the sidecar set that must
// travel with it. Percent-decoded to match real file names (the spec allows encoded uris).
function externalUris(gltf: {
    buffers?: { uri?: string }[];
    images?: { uri?: string }[];
}): string[] {
    const refs = [...(gltf.buffers ?? []), ...(gltf.images ?? [])];
    return refs.flatMap((r) => {
        if (!r.uri || r.uri.startsWith("data:")) return [];
        try {
            return [decodeURIComponent(r.uri)];
        } catch {
            return [r.uri];
        }
    });
}

// resolve a uri against its referencing file's directory, normalizing `./` — the path the sidecar must
// sit at within the drop
function siblingPath(rootPath: string, uri: string): string {
    const dir = rootPath.slice(0, rootPath.lastIndexOf("/") + 1);
    const joined = dir + uri;
    const parts: string[] = [];
    for (const seg of joined.split("/")) {
        if (seg === "" || seg === ".") continue;
        if (seg === ".." && parts.length) parts.pop();
        else parts.push(seg);
    }
    return parts.join("/");
}

/**
 * group dropped files into importable models: each `.glb`/`.gltf` is a root, and a `.gltf` claims the
 * sidecars (`.bin`, textures) its json references from the dropped set — a referenced file the drop
 * lacks lands in `missing`, so the caller can fail naming it. Files no root references are ignored.
 * Returns no groups when the drop holds no model file.
 */
export function groupModels(files: ImportFile[]): ModelGroup[] {
    const byPath = new Map(files.map((f) => [f.path, f]));
    const groups: ModelGroup[] = [];
    for (const root of files) {
        if (!MODEL_EXT.test(root.path)) continue;
        if (root.path.toLowerCase().endsWith(".glb")) {
            groups.push({ root, sidecars: [], missing: [] });
            continue;
        }
        let uris: string[];
        try {
            uris = externalUris(JSON.parse(new TextDecoder().decode(root.bytes)));
        } catch {
            // not parseable json — let the importer's decode produce the real error downstream
            groups.push({ root, sidecars: [], missing: [] });
            continue;
        }
        const sidecars: ImportFile[] = [];
        const missing: string[] = [];
        for (const uri of uris) {
            const path = siblingPath(root.path, uri);
            const file = byPath.get(path);
            if (file) sidecars.push(file);
            else missing.push(uri);
        }
        groups.push({ root, sidecars, missing });
    }
    return groups;
}

// the model's outliner id base: the file stem, lowercased, non-id characters folded to dashes
function idBase(src: string): string {
    const file = src.slice(src.lastIndexOf("/") + 1);
    const stem = file.replace(MODEL_EXT, "");
    return (
        stem
            .toLowerCase()
            .replace(/[^a-z0-9_-]+/g, "-")
            .replace(/^-+|-+$/g, "") || "model"
    );
}

/**
 * mint the document nodes for an imported asset — one flat `part` + `transform` + `color` node per node
 * placement (baked TRS from the descriptor, offset by `at`), ids deduped against the document. An asset
 * with no placements gets one node per primitive at `at`. The output is indistinguishable from
 * hand-authored nodes: the mesh is named (`src#i`), the surface + material decorations stay runtime
 * (the route sync's), so the scene round-trips through the declarative load.
 */
export function mintNodes(
    imp: GltfImport,
    src: string,
    at: [number, number, number],
    doc: { nodes: Node[] },
): Node[] {
    const placements = imp.instances.length
        ? imp.instances
        : imp.meshes.map((_, i) => ({
              handle: i,
              pos: [0, 0, 0] as [number, number, number],
              rot: [0, 0, 0, 1] as [number, number, number, number],
              scale: [1, 1, 1] as [number, number, number],
          }));
    const base = idBase(src);
    const used = new Set<string>();
    return placements.map((p) => {
        const handle = imp.meshes[p.handle];
        const attrs = [
            { name: "part", value: formatFields("part", { mesh: handle.name }) },
            {
                name: "transform",
                value: formatFields("transform", {
                    pos: [p.pos[0] + at[0], p.pos[1] + at[1], p.pos[2] + at[2], 0],
                    rot: p.rot,
                    scale: [p.scale[0], p.scale[1], p.scale[2], 1],
                }),
            },
            { name: "color", value: formatFields("color", { rgba: handle.color }) },
        ];
        return { id: mintId(base, doc, used), attrs, children: [] };
    });
}

/** upload one file through `POST /__api/asset`, returning the final public-relative src (deduped on a
 *  byte-differing collision). Throws with the server's message on a rejected write. */
export async function uploadAsset(name: string, bytes: ArrayBuffer): Promise<string> {
    const res = await fetch(`/__api/asset?name=${encodeURIComponent(name)}`, {
        method: "POST",
        body: bytes,
    });
    const data = (await res.json()) as { src?: string; error?: string };
    if (!res.ok || !data.src) throw new Error(data.error ?? `upload failed (${res.status})`);
    return data.src;
}

/**
 * upload one model group: sidecars first, then the root. A sidecar must land at its exact referenced
 * path — a byte-differing collision there means the name already belongs to a different file, and a
 * deduped rename would leave the root pointing at the wrong bytes, so it throws. The root's own dedupe
 * rename is safe (its references point at the just-placed sidecars). Returns the root's final src, the
 * `loadGltf` key.
 */
export async function uploadModel(group: ModelGroup): Promise<string> {
    if (group.missing.length) {
        throw new Error(
            `${group.root.path} references missing file(s): ${group.missing.join(", ")}`,
        );
    }
    for (const sidecar of group.sidecars) {
        const src = await uploadAsset(sidecar.path, sidecar.bytes);
        if (src !== sidecar.path) {
            throw new Error(
                `"${sidecar.path}" already exists with different contents — rename the file or folder and retry`,
            );
        }
    }
    return uploadAsset(group.root.path, group.root.bytes);
}

// a FileSystemEntry (webkitGetAsEntry) file read, promisified
function entryFile(entry: FileSystemFileEntry): Promise<File> {
    return new Promise((resolve, reject) => entry.file(resolve, reject));
}

function entryDir(reader: FileSystemDirectoryReader): Promise<FileSystemEntry[]> {
    return new Promise((resolve, reject) => reader.readEntries(resolve, reject));
}

// walk one entry (file or directory) into ImportFiles, paths rooted at the drop (a dropped folder keeps
// its own name as the prefix — the natural per-model directory)
async function walkEntry(entry: FileSystemEntry, prefix: string, out: ImportFile[]): Promise<void> {
    if (entry.isFile) {
        const file = await entryFile(entry as FileSystemFileEntry);
        out.push({ path: prefix + entry.name, bytes: await file.arrayBuffer() });
        return;
    }
    if (entry.isDirectory) {
        const reader = (entry as FileSystemDirectoryEntry).createReader();
        // readEntries returns batches of ≤100 — drain until empty
        for (;;) {
            const batch = await entryDir(reader);
            if (batch.length === 0) break;
            for (const child of batch) await walkEntry(child, `${prefix}${entry.name}/`, out);
        }
    }
}

/** read a drop's files, traversing dropped directories (Chrome's `webkitGetAsEntry` — the editor is
 *  Chrome-only). A dropped folder's files keep the folder prefix, so a `.gltf` set imports as a unit. */
export async function collectFiles(dt: DataTransfer): Promise<ImportFile[]> {
    const out: ImportFile[] = [];
    const entries = Array.from(dt.items, (item) => item.webkitGetAsEntry?.()).filter(
        (e): e is FileSystemEntry => !!e,
    );
    if (entries.length) {
        for (const entry of entries) await walkEntry(entry, "", out);
        return out;
    }
    for (const file of Array.from(dt.files)) {
        out.push({ path: file.name, bytes: await file.arrayBuffer() });
    }
    return out;
}
