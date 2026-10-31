import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import manifest from "../../examples/gltf/assets.json";
import { adversarial, adversarialNames } from "./adversarial";

export const models = [...manifest.models.map(model => model.name), ...adversarialNames];
export const assets = resolve(import.meta.dir, "../../examples/gltf/public/assets");

/** Reads the pinned fixtures' dense float attributes and unsigned triangle indices. */
export function primitives(name: string): { mesh: number; primitive: number; vertices: Float32Array; indices: Uint32Array }[] {
    if (adversarialNames.includes(name)) return [adversarial(name)];
    const path = resolve(assets, `Models/${name}/glTF/${name}.gltf`);
    const gltf = JSON.parse(readFileSync(path, "utf8"));
    const buffers = gltf.buffers.map((buffer: { uri: string }) => {
        const bytes = readFileSync(resolve(dirname(path), buffer.uri));
        return new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    });
    const accessor = (id: number): number[] => {
        const a = gltf.accessors[id];
        const view = gltf.bufferViews[a.bufferView];
        const data = buffers[view.buffer];
        const width = ({ SCALAR: 1, VEC2: 2, VEC3: 3 } as Record<string, number>)[a.type];
        const size = ({ 5121: 1, 5123: 2, 5125: 4, 5126: 4 } as Record<number, number>)[a.componentType];
        if (!width || !size || a.sparse || a.normalized) throw new Error("unsupported reference accessor");
        return Array.from({ length: a.count * width }, (_, i) => {
            const offset = (view.byteOffset ?? 0) + (a.byteOffset ?? 0) + Math.floor(i / width) * (view.byteStride ?? width * size) + (i % width) * size;
            return a.componentType === 5126 ? data.getFloat32(offset, true) : size === 4 ? data.getUint32(offset, true) : size === 2 ? data.getUint16(offset, true) : data.getUint8(offset);
        });
    };
    const result = [];
    for (let mesh = 0; mesh < gltf.meshes.length; mesh++) {
        for (let primitive = 0; primitive < gltf.meshes[mesh].primitives.length; primitive++) {
            const p = gltf.meshes[mesh].primitives[primitive];
            const texture = gltf.materials?.[p.material]?.normalTexture;
            if ((p.mode ?? 4) < 4 || (!texture || p.attributes.TANGENT !== undefined) && !["NormalTangentTest", "NormalTangentMirrorTest"].includes(name)) continue;
            if ((p.mode ?? 4) !== 4 || texture?.texCoord) throw new Error(`reference reader needs topology or UV set support for ${name}`);
            const positions = accessor(p.attributes.POSITION);
            const normals = accessor(p.attributes.NORMAL);
            const uv = accessor(p.attributes.TEXCOORD_0);
            const count = positions.length / 3;
            const vertices = new Float32Array(count * 8);
            for (let i = 0; i < count; i++) vertices.set([...positions.slice(i * 3, i * 3 + 3), uv[i * 2], ...normals.slice(i * 3, i * 3 + 3), uv[i * 2 + 1]], i * 8);
            const indices = Uint32Array.from(p.indices === undefined ? Array.from({ length: count }, (_, i) => i) : accessor(p.indices));
            result.push({ mesh, primitive, vertices, indices });
        }
    }
    return result;
}
