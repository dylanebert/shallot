// Sprite is a mesh material producer. Its image, tint, fill, billboard and blend route live in one row of
// the Sprite material type's table; MeshInstance preprocessing supplies the same stepped draw and shadow
// path as every other mesh.

import * as d from "typegpu/data";
import { type MeshHandle, MeshInstance, MeshPlugin, registerMesh } from "../../core/mesh";
import { BeginFrameSystem, imageArray, PrepassSystem, RenderingPlugin } from "../../core/rendering";
import { GlobalTransform } from "../../core/transform";
import { type Plugin, Registry, type System, type World } from "../../engine";
import { packColor } from "../../engine/utils";
import {
    MaterialPlugin,
    MeshMaterial,
    MeshPreprocessSystem,
    MeshRenderPlugin,
    materialTypeId,
    StandardRenderingPlugin,
} from "../../standard/rendering";
import { SpriteAlphaMaterialType, SpriteMaterialType } from "./material";
import { packSpriteFill, Sprite, SpriteBlend } from "./pack";

export { SpriteAlphaMaterialType, SpriteMaterialInput, SpriteMaterialType } from "./material";
export { Sprite, SpriteBillboard, SpriteBlend, SpriteFill } from "./pack";

export const Images = { create: () => new Registry<{ name: string; source: string | Blob }>() };

const PIXEL_PNG =
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAADUlEQVR42mNgYGD4DwABBAEAX+XBlwAAAABJRU5ErkJggg==";

function transparentPixel(): Blob {
    const bytes = Uint8Array.from(atob(PIXEL_PNG), (c) => c.charCodeAt(0));
    return new Blob([bytes], { type: "image/png" });
}

/** Register a source image; its returned id is the Sprite material row's texture-array layer. */
export function registerImage(world: World, source: string | Blob, name?: string): number {
    const images = world.resource(Images);
    const key = name ?? (typeof source === "string" ? source : `image${images.size}`);
    return images.register({ name: key, source });
}

// prettier-ignore
const QUAD_VERTS = new Float32Array([
    0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0,
]);
const QUAD_INDICES = new Uint32Array([0, 1, 2, 0, 2, 3]);

interface SpriteGpuState {
    atlas: GPUTexture | null;
    sampler: GPUSampler | null;
    quad: MeshHandle | null;
    instances: Map<number, SpriteInstance>;
}
interface SpriteInstance {
    type: typeof SpriteMaterialType | typeof SpriteAlphaMaterialType;
    signature: string;
}
const spriteGpuKey = {
    create: (): SpriteGpuState => ({
        atlas: null,
        sampler: null,
        quad: null,
        instances: new Map(),
    }),
};

function values(world: World, eid: number) {
    const sprite = world.storage(Sprite);
    const width = sprite.size.x.get(eid);
    const height = sprite.size.y.get(eid);
    const anchorX = sprite.anchor.x.get(eid);
    const anchorY = sprite.anchor.y.get(eid);
    const image = sprite.image.get(eid);
    const color = sprite.color.get(eid);
    const opacity = sprite.opacity.get(eid);
    const billboard = Math.min(sprite.billboard.get(eid), 2);
    const blend = Math.min(sprite.blend.get(eid), 1);
    const fill = sprite.fill.get(eid);
    const fillMode = sprite.fillMode.get(eid);
    const params = {
        offset: d.vec2f(-width * anchorX, -height * anchorY),
        size: d.vec2f(width, height),
        layer: image,
        color: packColor(color, opacity),
        fill: packSpriteFill(fill, fillMode),
        billboard,
    };
    return {
        params,
        type: blend === SpriteBlend.Alpha ? SpriteAlphaMaterialType : SpriteMaterialType,
        signature: [
            image,
            width,
            height,
            anchorX,
            anchorY,
            color,
            opacity,
            billboard,
            blend,
            fill,
            fillMode,
        ].join("/"),
    };
}

function releaseInstance(world: World, eid: number): void {
    if (world.has(eid, MeshMaterial)) world.remove(eid, MeshMaterial);
    if (world.has(eid, MeshInstance)) world.remove(eid, MeshInstance);
}

const SpriteSystem: System = {
    name: "sprite",
    group: "draw",
    after: [BeginFrameSystem],
    before: [MeshPreprocessSystem, PrepassSystem],
    update(world) {
        const state = world.resource(spriteGpuKey);
        if (state.quad === null) return;
        const live = new Set<number>();
        const sprite = world.storage(Sprite);
        for (const eid of world.query([Sprite])) {
            live.add(eid);
            const current = state.instances.get(eid);
            if (!world.has(eid, GlobalTransform)) continue;
            if (!sprite.visible.get(eid)) {
                if (current) {
                    releaseInstance(world, eid);
                    state.instances.delete(eid);
                }
                continue;
            }
            const next = values(world, eid);
            const type = next.type;
            const assets = world.resource(type);
            if (!current) {
                if (world.has(eid, MeshInstance) || world.has(eid, MeshMaterial)) {
                    throw new Error(
                        `Sprite ${eid} cannot share its entity with another MeshInstance or MeshMaterial`,
                    );
                }
                assets.setAt(eid, next.params);
                world.add(eid, MeshInstance, { mesh: state.quad });
                world.add(eid, MeshMaterial, {
                    type: materialTypeId(world, type),
                    material: eid,
                });
                state.instances.set(eid, { type, signature: next.signature });
                continue;
            }
            if (current.type !== type || current.signature !== next.signature) {
                assets.setAt(eid, next.params);
                current.type = type;
                current.signature = next.signature;
                world.storage(MeshMaterial).type.set(eid, materialTypeId(world, type));
                world.storage(MeshMaterial).material.set(eid, eid);
            }
        }
        for (const eid of state.instances.keys()) {
            if (live.has(eid)) continue;
            releaseInstance(world, eid);
            state.instances.delete(eid);
        }
    },
};

const SpriteMaterialPlugin = MaterialPlugin(SpriteMaterialType);
const SpriteAlphaMaterialPlugin = MaterialPlugin(SpriteAlphaMaterialType);

/** Retained textured quads; clip sprites cast shadows through StandardRenderingPlugin. */
export const SpritePlugin: Plugin = {
    gpu: {},
    name: "Sprite",
    components: [Sprite],
    systems: [SpriteSystem],
    dependencies: [
        MeshPlugin,
        RenderingPlugin,
        StandardRenderingPlugin,
        MeshRenderPlugin,
        SpriteMaterialPlugin,
        SpriteAlphaMaterialPlugin,
    ],

    initialize(world) {
        const state = world.resource(spriteGpuKey);
        state.atlas = null;
        state.sampler = null;
        state.instances.clear();
        if (world.gpu.device) {
            state.quad = registerMesh(world, {
                name: "spriteQuad",
                vertices: QUAD_VERTS,
                indices: QUAD_INDICES,
            });
        }
    },

    async warm(world) {
        const images = world.resource(Images);
        const state = world.resource(spriteGpuKey);
        if (!world.gpu.device) return;
        const device = world.gpu.device;
        if (images.size > 0 && typeof createImageBitmap !== "undefined") {
            const blobs = await Promise.all(
                Array.from({ length: images.size }, async (_, id) => {
                    const source = world
                        .resource(Images)
                        .get(world.resource(Images).name(id)!)!.source;
                    if (typeof source !== "string") return source;
                    try {
                        const response = await fetch(source);
                        if (!response.ok) throw new Error(`${response.status}`);
                        return await response.blob();
                    } catch (error) {
                        console.warn(`[Sprite] image ${id} (${source}) failed to load:`, error);
                        return transparentPixel();
                    }
                }),
            );
            state.atlas = await imageArray(world, device, blobs);
        } else {
            state.atlas = device.createTexture({
                label: "sprite-atlas-fallback",
                size: { width: 1, height: 1, depthOrArrayLayers: 1 },
                format: "rgba8unorm-srgb",
                usage: GPUTextureUsage.TEXTURE_BINDING,
            });
        }
        world.gpu.textures.set("spriteAtlas", state.atlas);
        state.sampler = device.createSampler({
            label: "sprite",
            magFilter: "linear",
            minFilter: "linear",
        });
        world.gpu.samplers.set("spriteSamp", state.sampler);
    },

    dispose(world) {
        const state = world.resource(spriteGpuKey);
        state.atlas?.destroy();
        state.atlas = null;
        state.sampler = null;
        state.quad = null;
        state.instances.clear();
    },
};
