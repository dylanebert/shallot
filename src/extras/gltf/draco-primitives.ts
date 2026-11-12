import type { DracoType } from "./draco";
import type { GltfJson } from "./gltf";

const NAME = "KHR_draco_mesh_compression";
const COMPONENTS = { SCALAR: 1, VEC2: 2, VEC3: 3, VEC4: 4, MAT2: 4, MAT3: 9, MAT4: 16 };
const TYPES: Record<number, [DracoType, number]> = {
    5120: ["int8", 1],
    5121: ["uint8", 1],
    5122: ["int16", 2],
    5123: ["uint16", 2],
    5125: ["uint32", 4],
    5126: ["float32", 4],
};
/** The largest decoded accessor admitted, in bytes: the bound meshopt places on a decoded view. */
const MAX_BYTES = 0x7ffffff8;

type Accessor = NonNullable<GltfJson["accessors"]>[number];
interface Draco {
    bufferView: number;
    attributes: Record<string, number>;
}

/** Replaces each Draco primitive's indices and extension attributes with decoded accessors. */
export async function decodeDracoPrimitives(json: GltfJson, buffers: ArrayBuffer[]): Promise<void> {
    const primitives = (json.meshes ?? []).flatMap((mesh, m) =>
        mesh.primitives.flatMap((prim, p) => {
            const def = prim.extensions?.[NAME] as Draco | undefined;
            return def === undefined ? [] : [{ prim, def, owner: `mesh ${m} primitive ${p}` }];
        }),
    );
    if (primitives.length === 0) return;
    const { decodeDraco } = await import("./draco");
    const accessors = (json.accessors ??= []);
    const views = (json.bufferViews ??= []);
    for (const { prim, def, owner } of primitives) {
        const mode = prim.mode ?? 4;
        if (mode !== 4)
            throw new Error(
                `${owner}: ${NAME} refuses mode ${mode === 5 ? "TRIANGLE_STRIP" : mode}; only TRIANGLES decode`,
            );
        const view = json.bufferViews?.[def?.bufferView];
        const buffer = view && buffers[view.buffer];
        const offset = view?.byteOffset ?? 0;
        if (
            !Number.isInteger(def?.bufferView) ||
            !buffer ||
            !Number.isSafeInteger(offset) ||
            !Number.isSafeInteger(view.byteLength) ||
            offset < 0 ||
            view.byteLength < 0 ||
            offset + view.byteLength > buffer.byteLength
        )
            throw new Error(`${owner}: ${NAME} invalid bufferView`);
        const attributes = Object.entries(def.attributes ?? {});
        const targets = attributes.map(([semantic, id]) => {
            const index = (prim.attributes as Record<string, number>)[semantic];
            const accessor = accessors[index];
            if (!accessor)
                throw new Error(`${owner}: ${NAME} attribute ${semantic} has no accessor`);
            return { semantic, id, index, accessor };
        });
        const position = accessors[prim.attributes.POSITION];
        // Without indices, POSITION's count gives the triangles; the decoded faces still index
        // the decoded points, which Draco may reorder and merge, as three.js DRACOLoader reads.
        const indices: Accessor | undefined =
            prim.indices === undefined
                ? position && { componentType: 5125, count: position.count, type: "SCALAR" }
                : accessors[prim.indices];
        if (!position || !indices) throw new Error(`${owner}: ${NAME} missing accessor`);
        for (const { semantic, accessor } of [
            ...targets,
            { semantic: "indices", accessor: indices },
        ]) {
            const type = TYPES[accessor.componentType];
            const components = COMPONENTS[accessor.type];
            if (
                !type ||
                !components ||
                !Number.isSafeInteger(accessor.count) ||
                accessor.count < 0 ||
                accessor.count * components * type[1] > MAX_BYTES
            )
                throw new Error(`${owner}: ${NAME} ${semantic} accessor count or type invalid`);
        }
        if (indices.count % 3 !== 0)
            throw new Error(`${owner}: ${NAME} indices count is not a multiple of 3`);
        const decoded = decodeDraco(
            new Uint8Array(buffer, offset, view.byteLength),
            indices.count / 3,
            position.count,
            targets.map(({ id, accessor }) => ({ id, type: TYPES[accessor.componentType][0] })),
        );
        const point = (data: ArrayBufferView, { sparse: _, ...accessor }: Accessor) => {
            views.push({ buffer: buffers.length, byteLength: data.byteLength });
            buffers.push(data.buffer as ArrayBuffer);
            return accessors.push({ ...accessor, bufferView: views.length - 1, byteOffset: 0 }) - 1;
        };
        const attributesOut = prim.attributes as Record<string, number>;
        for (const [i, { semantic, index, accessor }] of targets.entries()) {
            const { components, data } = decoded.attributes[i];
            const count = data.length / components;
            if (components !== COMPONENTS[accessor.type] || count !== accessor.count)
                throw new Error(
                    `${owner}: ${NAME} ${semantic} accessor ${index} count ${accessor.count} ${accessor.type} differs from decoded ${count} of ${components} components`,
                );
            attributesOut[semantic] = point(data, accessor);
        }
        prim.indices = point(decoded.faces, {
            ...indices,
            componentType: 5125,
            type: "SCALAR",
        });
        delete prim.extensions![NAME];
    }
}
