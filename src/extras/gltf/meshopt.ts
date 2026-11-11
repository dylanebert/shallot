import type { GltfJson } from "./gltf";
import { decodeMeshoptView } from "./kernel";

const names = ["KHR_meshopt_compression", "EXT_meshopt_compression"] as const;
interface CompressedView {
    buffer: number;
    byteOffset?: number;
    byteLength: number;
    byteStride: number;
    count: number;
    mode: string;
    filter?: string;
}
function extension(extensions: Record<string, unknown> | undefined, owner: string) {
    const present = names.filter((name) => extensions?.[name] !== undefined);
    if (present.length === 2)
        throw new Error(`${owner}: KHR_meshopt_compression excludes EXT_meshopt_compression`);
    return present[0];
}
function range(offset: number, length: number, capacity: number): boolean {
    return (
        Number.isSafeInteger(offset) &&
        offset >= 0 &&
        Number.isSafeInteger(length) &&
        length > 0 &&
        offset + length <= capacity
    );
}

/** Validates meshopt references before fetching and identifies buffers whose views all decode. */
export function meshoptFallbacks(json: GltfJson, hasBin: boolean): Set<number> {
    const buffers = json.buffers ?? [];
    const views = json.bufferViews ?? [];
    const compressed = views.map((view, i) => extension(view.extensions, `bufferView ${i}`));
    const sources = new Set<number>();
    for (const [i, view] of views.entries()) {
        const name = compressed[i];
        if (!name) continue;
        const def = view.extensions![name] as CompressedView;
        const parent = buffers[view.buffer];
        const source = buffers[def?.buffer];
        if (
            !Number.isInteger(view.buffer) ||
            !parent ||
            !range(view.byteOffset ?? 0, view.byteLength, parent.byteLength)
        )
            throw new Error(`bufferView ${i}: ${name} invalid fallback buffer range`);
        if (
            !Number.isInteger(def?.buffer) ||
            (def.byteOffset !== undefined && !Number.isSafeInteger(def.byteOffset)) ||
            !source ||
            !range(def.byteOffset ?? 0, def.byteLength, source.byteLength)
        )
            throw new Error(`bufferView ${i}: ${name} invalid compressed buffer range`);
        if (
            (def.filter !== undefined && typeof def.filter !== "string") ||
            !Number.isSafeInteger(def.count) ||
            def.count < 1 ||
            !Number.isInteger(def.byteStride) ||
            def.byteStride < 1 ||
            def.byteStride > 256 ||
            def.count * def.byteStride !== view.byteLength ||
            view.byteLength > 0x7ffffff8 ||
            (name === names[1] &&
                view.byteStride !== undefined &&
                view.byteStride !== def.byteStride)
        )
            throw new Error(`bufferView ${i}: ${name} invalid count, byteStride or byteLength`);
        sources.add(def.buffer);
    }
    const skipped = new Set<number>();
    for (const [i, buffer] of buffers.entries()) {
        const name = extension(buffer.extensions, `buffer ${i}`);
        const marker = name ? (buffer.extensions![name] as { fallback?: boolean }) : undefined;
        if (marker?.fallback !== undefined && typeof marker.fallback !== "boolean")
            throw new Error(`buffer ${i}: ${name} invalid fallback`);
        const references = views
            .map((view, j) => (view.buffer === i ? j : -1))
            .filter((j) => j >= 0);
        if (marker?.fallback && (sources.has(i) || references.some((j) => compressed[j] !== name)))
            throw new Error(`buffer ${i}: ${name} invalid fallback references`);
        if (
            marker?.fallback &&
            !buffer.uri &&
            !(i === 0 && hasBin) &&
            !json.extensionsRequired?.includes(name!)
        )
            throw new Error(`buffer ${i}: ${name} must be required for a placeholder`);
        if (
            !sources.has(i) &&
            references.every((j) => compressed[j]) &&
            (marker?.fallback || references.length > 0)
        ) {
            for (const j of references) {
                if (
                    !buffer.uri &&
                    !(i === 0 && hasBin) &&
                    !json.extensionsRequired?.includes(compressed[j]!)
                )
                    throw new Error(
                        `buffer ${i}: ${compressed[j]} must be required for a placeholder`,
                    );
            }
            skipped.add(i);
        }
    }
    return skipped;
}

/** Replaces compressed views with private decoded buffers before any accessor or image reads. */
export function decodeMeshoptViews(json: GltfJson, buffers: ArrayBuffer[]): void {
    for (const [i, view] of (json.bufferViews ?? []).entries()) {
        const name = extension(view.extensions, `bufferView ${i}`);
        if (!name) continue;
        const def = view.extensions![name] as CompressedView;
        const source = buffers[def.buffer];
        if (!range(def.byteOffset ?? 0, def.byteLength, source.byteLength))
            throw new Error(`bufferView ${i}: ${name} compressed buffer is truncated`);
        const output = new Uint8Array(view.byteLength);
        decodeMeshoptView(
            i,
            def,
            new Uint8Array(source, def.byteOffset ?? 0, def.byteLength),
            output,
            name === names[0],
        );
        view.buffer = buffers.length;
        view.byteOffset = 0;
        buffers.push(output.buffer);
    }
}
