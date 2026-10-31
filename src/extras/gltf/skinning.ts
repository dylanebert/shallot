import tgpu from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import { MeshPlugin } from "../../core/mesh";
import { Xform, xformNormal, xformPoint } from "../../core/rendering";
import { component, type Plugin, type System, u32, type World } from "../../engine";
import {
    applyPbrLighting,
    extendedMaterial,
    extensionLayout,
    MaterialPlugin,
    Materials,
    MeshPreprocessSystem,
    type materialLayout,
    pbrInputFromStandardMaterial,
} from "../../standard/rendering";

/** Four u16 joint indices and four float32 weights, addressed by family-absolute vidx. */
export const SkinInfluences = d.struct({ joints: d.vec2u, weights: d.vec4f });
/** Bindings required by skinningVertex, in both materialLayout and extensionLayout. */
export const skinningBindings = {
    skin: { type: "attribute", element: SkinInfluences },
    jointPalette: { type: "texture", filterable: false },
    skinOffsets: { type: "texture", sampleType: "uint" },
} as const;
const PaletteWidth = 256;
const JointsPerRow = PaletteWidth / 4;

/** Palette allocation owned by this entity's membership. firstJoint is derived; jointCount is
 * supplied on attachment and must remain unchanged until removal. Removing or destroying the entity frees it. */
export const SkinnedMesh = component("SkinnedMesh", { firstJoint: u32, jointCount: u32 });

/** CPU staging and world-owned textures. extent is the allocated joint high-water mark, including holes. */
export const SkinPalette = {
    create: (world: World) => new Palette(world),
};
class Palette {
    enabled = false;
    extent = 0;
    matrices = new Float32Array(0);
    private _free: { first: number; count: number }[] = [];
    private _paletteHeight = 0;
    private _offsetHeight = 0;
    private _offsets = new Uint32Array(0);
    jointPalette!: GPUTexture;
    skinOffsets!: GPUTexture;
    readonly bindings = {
        jointPalette: undefined as unknown as GPUTexture,
        skinOffsets: undefined as unknown as GPUTexture,
    };
    private readonly _paletteTarget: GPUTexelCopyTextureInfo = {
        texture: undefined as unknown as GPUTexture,
    };
    private readonly _offsetTarget: GPUTexelCopyTextureInfo = {
        texture: undefined as unknown as GPUTexture,
    };
    private readonly _paletteLayout = { bytesPerRow: PaletteWidth * 16 };
    private readonly _offsetLayout = { bytesPerRow: PaletteWidth * 4 };
    private readonly _paletteSize = { width: PaletteWidth, height: 1 };
    private readonly _offsetSize = { width: PaletteWidth, height: 1 };
    private _world: World;
    constructor(world: World) {
        this._world = world;
    }
    allocate(count: number): number {
        if (!Number.isInteger(count) || count <= 0)
            throw new Error("SkinnedMesh requires a positive jointCount");
        const hole = this._free.findIndex((range) => range.count >= count);
        if (hole < 0 && this.extent + count > JointsPerRow * 8192)
            throw new Error("SkinnedMesh palette exceeds WebGPU's default texture dimensions");
        let first: number;
        if (hole >= 0) {
            const range = this._free[hole];
            first = range.first;
            range.first += count;
            range.count -= count;
            if (!range.count) this._free.splice(hole, 1);
        } else {
            first = this.extent;
            this.extent += count;
        }
        this.growPalette(Math.max(1, Math.ceil(this.extent / JointsPerRow)));
        return first;
    }
    release(first: number, count: number): void {
        this._free.push({ first, count });
        this._free.sort((a, b) => a.first - b.first);
        for (let i = this._free.length - 2; i >= 0; i--) {
            const a = this._free[i],
                b = this._free[i + 1];
            if (a.first + a.count === b.first) {
                a.count += b.count;
                this._free.splice(i + 1, 1);
            }
        }
        const last = this._free.at(-1);
        if (last && last.first + last.count === this.extent) {
            this.extent = last.first;
            this._free.pop();
        }
    }
    private texture(format: GPUTextureFormat, height: number): GPUTexture {
        return this._world.gpu.device.createTexture({
            size: [PaletteWidth, height],
            format,
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
        });
    }
    private growPalette(height: number): void {
        if (height <= this._paletteHeight) return;
        height = Math.min(8192, Math.max(height, this._paletteHeight * 2));
        const data = new Float32Array(height * PaletteWidth * 4);
        data.set(this.matrices);
        this.matrices = data;
        this.jointPalette?.destroy();
        this.jointPalette = this.texture("rgba32float", height);
        this.bindings.jointPalette = this._paletteTarget.texture = this.jointPalette;
        this._paletteHeight = height;
    }
    upload(): void {
        this.growPalette(1);
        const column = this._world.fieldStorage(SkinnedMesh, "firstJoint").column;
        const height = Math.max(1, Math.ceil(column.length / PaletteWidth));
        if (height > 8192)
            throw new Error("SkinnedMesh entity map exceeds WebGPU's default texture dimensions");
        if (height > this._offsetHeight) {
            this._offsetHeight = this._offsetSize.height = height;
            this._offsets = new Uint32Array(height * PaletteWidth);
            this.skinOffsets?.destroy();
            this.skinOffsets = this.texture("r32uint", height);
            this.bindings.skinOffsets = this._offsetTarget.texture = this.skinOffsets;
        }
        this._offsets.set(column);
        this._paletteSize.height = Math.max(1, Math.ceil(this.extent / JointsPerRow));
        const queue = this._world.gpu.device.queue;
        queue.writeTexture(
            this._paletteTarget,
            this.matrices,
            this._paletteLayout,
            this._paletteSize,
        );
        queue.writeTexture(this._offsetTarget, this._offsets, this._offsetLayout, this._offsetSize);
    }
}

const SkinOutput = d.struct({ world: d.vec4f, worldNormal: d.vec3f });
type SkinLayout = Pick<
    ReturnType<typeof materialLayout<typeof skinningBindings, d.WgslStruct<Record<never, never>>>>,
    "$"
>;
/** Returns a vertex function taking eid, family-absolute vidx, local position, local normal and
 * placement Xform, returning world position and normal. Compose with any material declaring
 * skinningBindings. Uses the last StandardMaterial vertex storage binding and two sampled textures. Normals use the blended matrix's inverse transpose;
 * tangents are not deformed because vertex functions cannot patch mesh streams. */
export function skinningVertex(layout: SkinLayout) {
    const jointMatrix = tgpu.fn(
        [d.u32],
        d.mat4x4f,
    )((joint) => {
        "use gpu";
        const x = d.i32((joint % JointsPerRow) * 4);
        const y = d.i32(joint / JointsPerRow);
        return d.mat4x4f(
            std.textureLoad(layout.$.jointPalette, d.vec2i(x, y), 0),
            std.textureLoad(layout.$.jointPalette, d.vec2i(x + 1, y), 0),
            std.textureLoad(layout.$.jointPalette, d.vec2i(x + 2, y), 0),
            std.textureLoad(layout.$.jointPalette, d.vec2i(x + 3, y), 0),
        );
    });
    return tgpu.fn(
        [d.u32, d.u32, d.vec3f, d.vec3f, Xform],
        SkinOutput,
    )((eid, vidx, position, sourceNormal, xform) => {
        "use gpu";
        const first = std.textureLoad(
            layout.$.skinOffsets,
            d.vec2i(d.i32(eid % PaletteWidth), d.i32(eid / PaletteWidth)),
            0,
        ).x;
        const influence = layout.$.skin[vidx];
        const j = influence.joints;
        const w = influence.weights;
        const m = std.add(
            std.add(
                std.mul(w.x, jointMatrix(first + (j.x & 65535))),
                std.mul(w.y, jointMatrix(first + (j.x >>> 16))),
            ),
            std.add(
                std.mul(w.z, jointMatrix(first + (j.y & 65535))),
                std.mul(w.w, jointMatrix(first + (j.y >>> 16))),
            ),
        );
        const c0 = m.columns[0].xyz;
        const c1 = m.columns[1].xyz;
        const c2 = m.columns[2].xyz;
        const x = std.cross(c1, c2);
        const y = std.cross(c2, c0);
        const z = std.cross(c0, c1);
        const n = sourceNormal;
        const det = std.dot(c2, z);
        const normal = std.normalize(
            std.add(std.add(std.mul(n.x / det, x), std.mul(n.y / det, y)), std.mul(n.z / det, z)),
        );
        return {
            world: d.vec4f(xformPoint(xform, std.mul(m, d.vec4f(position, 1)).xyz), 1),
            worldNormal: xformNormal(xform, normal),
        };
    });
}
const layout = extensionLayout(Materials, d.struct({ skinPad: d.u32 }), skinningBindings);
const vertex = skinningVertex(layout);
/** StandardMaterial with linear blend skinning. A game's replacement material draws the bind pose
 * unless it declares skinningBindings and calls skinningVertex, using its last vertex-stage storage
 * buffer and two sampled textures. Tangents are not deformed. */
export const SkinnedMaterial = extendedMaterial(Materials, {
    name: "SkinnedMaterial",
    layout,
    defaults: { skinPad: 0 },
    bindings: (world) => world.resource(SkinPalette).bindings,
    vertex: (input) => {
        "use gpu";
        const posed = vertex(input.eid, input.vidx, input.localPos, input.localNormal, input.xform);
        return { world: posed.world, worldNormal: posed.worldNormal };
    },
    fragment: (ctx) => {
        "use gpu";
        return applyPbrLighting(pbrInputFromStandardMaterial(ctx));
    },
});
const UploadSkinPaletteSystem: System = {
    name: "upload-skin-palette",
    group: "draw",
    before: [MeshPreprocessSystem],
    update(world) {
        world.resource(SkinPalette).upload();
    },
};
/** glTF vertex skinning and entity-local palette lifetime. Replacement materials draw bind pose
 * unless they declare skinningBindings and call skinningVertex (one storage buffer and two sampled
 * textures). Tangents are not deformed. */
export const GltfPlugin: Plugin = {
    name: "Gltf",
    gpu: {},
    components: [SkinnedMesh],
    dependencies: [MeshPlugin, MaterialPlugin(SkinnedMaterial)],
    systems: [UploadSkinPaletteSystem],
    initialize(world) {
        const palette = world.resource(SkinPalette);
        palette.enabled = true;
        world.onDispose(
            world.observeMembership(SkinnedMesh, (eid, present) => {
                const storage = world.storage(SkinnedMesh);
                const count = storage.jointCount.get(eid);
                if (present) storage.firstJoint.set(eid, palette.allocate(count));
                else palette.release(storage.firstJoint.get(eid), count);
            }),
        );
        palette.upload();
    },
};
