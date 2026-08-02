// Sprite — world-space iconography. A retained `Sprite` component (registered image, world size,
// anchor, tint, billboard mode) draws textured unit quads instanced from one shared buffer, one
// indirect draw per (billboard, blend) variant. Images register into one `texture_2d_array` (the
// glTF binding model — per-instance layer index, one bind group, one draw), so adding an icon never
// adds a draw. Sprite rides the `eids`+`transforms` instancing convention: a slot-major eids buffer
// (packSprites' bucket-contiguous ranges) publishes each slot's owning entity, and the engine
// resolves the per-instance transform from the global entity-transform firehose — so moving a
// sprite flows through the Transform slab and triggers no rebuild, only the buffer rebuilds when a
// layout-affecting field changes, gated by a per-frame signature (the text producer's shape). The
// per-instance sprite data itself is eid-indexed, not slot-indexed (see surface.ts) — the shadow
// atlas re-gather preserves only `eid`, so both the vs and fs look it up by `VsIn.eid`/`ctx.eid`.
// The packing substance is pack.ts, the surface variants surface.ts, the billboard math spec
// billboard.ts; this file is the surface + producer around them.

import { Compute, formatHex, type Plugin, Registry, type State, type System } from "../../engine";
import { mesh, RenderPlugin } from "../../standard/render";
import { BeginFrameSystem, Draws, imageArray, Meshes, Surfaces } from "../../standard/render/core";
import { PrepassSystem, registerSurface } from "../../standard/sear/core";
import { Transform, TransformsPlugin } from "../../standard/transforms";
import {
    BUCKETS,
    INITIAL,
    packSprites,
    resetPack,
    SPRITE_BYTES,
    Sprite,
    SpriteBillboard,
    SpriteBlend,
    SpriteFill,
    signature,
} from "./pack";
import { spriteSurface, surfaceName } from "./surface";

export { Sprite, SpriteBillboard, SpriteBlend, SpriteFill } from "./pack";

const Images = new Registry<{ name: string; source: string | Blob }>();

// a 1×1 transparent png — the placeholder a failed fetch decodes into, keeping the failed image's
// layer index aligned with its id
const PIXEL_PNG =
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNgYGD4DwABBAEAX+XBlwAAAABJRU5ErkJggg==";

function transparentPixel(): Blob {
    const bytes = Uint8Array.from(atob(PIXEL_PNG), (c) => c.charCodeAt(0));
    return new Blob([bytes], { type: "image/png" });
}

/**
 * register a sprite image, returning the id stored in {@link Sprite.image}. `source` is a url or a
 * `Blob` (a procedurally-drawn `OffscreenCanvas.convertToBlob` works); `name` is the handle a
 * scene's `image:` attribute resolves (defaults to the url). Register any time up to a plugin's
 * `initialize` (`SpritePlugin` builds the `texture_2d_array` at `warm`, after every initialize), so a
 * plugin can register its own images (no pre-`build` call needed); all images share one array, layer-per-image
 *
 * @example
 * ```
 * image("/icons/house.png", "house");
 * ```
 */
export function image(source: string | Blob, name?: string): number {
    const key = name ?? (typeof source === "string" ? source : `image${Images.size}`);
    return Images.register({ name: key, source });
}

// the unit quad sear instances per sprite: posU.xyz = (corner.x, corner.y, 0); normal +Z so the
// world variant's worldNormal is meaningful
// prettier-ignore
const QUAD_VERTS = new Float32Array([
    0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0,
]);
const QUAD_INDICES = new Uint32Array([0, 1, 2, 0, 2, 3]);

let _atlas: GPUTexture | null = null;
let _sampler: GPUSampler | null = null;
let _spriteBuf: GPUBuffer | null = null;
// the slot-major eids buffer (the instancing convention's `eids` binding; `_spriteBuf` is
// eid-indexed instead, sized independently — see pack.ts) — overridden per-draw via `spriteQuad`'s
// `mesh.bindings` since it's sprite's own packed instance list, not the global entity-transform
// firehose (`transforms` stays unoverridden, resolving to that global)
let _eidsBuf: GPUBuffer | null = null;
let _argBuf: GPUBuffer | null = null;
let _quadBase = 0;
let _sig = -1;
const _args = new Uint32Array(5);

function rebuild(state: State, device: GPUDevice): void {
    const { ranges, count, dataCap, f32, eids } = packSprites(state);

    // `dataCap` is eid-indexed (sized off `maxEid + 1`), not slot count — a sprite's instance data
    // lands at `eid * SPRITE_BYTES`, so the whole eid-addressable range must upload, not just the
    // first `count` records
    if (dataCap * SPRITE_BYTES > _spriteBuf!.size) {
        const stale = _spriteBuf!;
        _spriteBuf = device.createBuffer({
            label: "kitchen-sprites",
            size: dataCap * SPRITE_BYTES,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        Compute.buffers.set("spriteData", _spriteBuf);
        device.queue.onSubmittedWorkDone().then(() => stale.destroy());
    }
    // eid-indexed, so the whole capacity uploads every rebuild — a dead eid's stale slot is never
    // read (only an eid appearing in `eids` this rebuild is), so uploading it costs bandwidth, not
    // correctness
    device.queue.writeBuffer(_spriteBuf!, 0, f32, 0, dataCap * (SPRITE_BYTES / 4));

    if (eids.length * 4 > _eidsBuf!.size) {
        const stale = _eidsBuf!;
        _eidsBuf = device.createBuffer({
            label: "kitchen-sprite-eids",
            size: eids.length * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        const quad = Meshes.get("spriteQuad");
        if (quad) quad.bindings = { ...quad.bindings, eids: _eidsBuf };
        device.queue.onSubmittedWorkDone().then(() => stale.destroy());
    }
    if (count > 0) device.queue.writeBuffer(_eidsBuf!, 0, eids, 0, count * 4);

    for (let b = 0; b < BUCKETS; b++) {
        _args[0] = 6;
        _args[1] = ranges[b].count;
        _args[2] = _quadBase;
        _args[3] = 0;
        _args[4] = ranges[b].start;
        device.queue.writeBuffer(_argBuf!, b * 20, _args);
    }
}

const SpriteSystem: System = {
    name: "sprite",
    group: "draw",
    annotations: { mode: "always" },
    after: [BeginFrameSystem],
    before: [PrepassSystem],
    setup() {
        _quadBase = Meshes.get("spriteQuad")?.indexBase ?? 0;
        // all six draws, unconditionally — an empty bucket packs instanceCount 0 and no-ops
        for (let b = 0; b < BUCKETS; b++) {
            Draws.register({
                name: surfaceName(b),
                surface: surfaceName(b),
                mesh: "spriteQuad",
                args: { indirect: _argBuf!, offset: b * 20 },
            });
        }
    },
    update(state) {
        if (!Compute.device || !_spriteBuf || !_eidsBuf || !_argBuf) return;
        const sig = signature(state);
        if (sig === _sig) return;
        _sig = sig;
        rebuild(state, Compute.device);
    },
};

/**
 * the sprite producer: the retained {@link Sprite} component drawn as instanced textured quads,
 * world-space icons and markers. Register images with {@link image}; they upload into one
 * `texture_2d_array`, so every sprite draws in one indirect draw per (billboard, blend) variant.
 * Default `clip` blend writes depth and casts holed shadows; billboard modes are compile-time
 * surface variants. Depends on {@link RenderPlugin}; a Sear camera renders it
 */
export const SpritePlugin: Plugin = {
    name: "Sprite",
    components: { Sprite },
    systems: [SpriteSystem],
    dependencies: [RenderPlugin, TransformsPlugin],
    traits: {
        Sprite: {
            requires: [Transform],
            defaults: () => ({
                image: 0,
                size: [1, 1],
                anchor: [0.5, 0.5],
                color: 0xffffff,
                opacity: 1,
                visible: 1,
                billboard: SpriteBillboard.Screen,
                blend: SpriteBlend.Clip,
                fill: 1,
                fillMode: SpriteFill.None,
            }),
            parse: {
                image: (name: string) => Images.id(name) ?? 0,
            },
            format: {
                color: formatHex,
                image: (id: number) => Images.name(id) ?? "",
            },
            enums: { billboard: SpriteBillboard, blend: SpriteBlend, fillMode: SpriteFill },
        },
    },

    initialize(state) {
        _atlas = null;
        _sampler = null;
        _spriteBuf = null;
        _eidsBuf = null;
        _argBuf = null;
        _sig = -1;

        for (let b = 0; b < BUCKETS; b++) {
            // the string-registry shell (bindings only — no code): keeps `sprite-*` in the surface-id
            // space every string-era consumer still enumerates. The code is the typed registration
            // below, which `record()` consults first (4a-ii-d dual-accept)
            Surfaces.register({
                name: surfaceName(b),
                blend: b & 1 ? ("alpha" as const) : ("clip" as const),
                bindings: {
                    spriteData: { type: "storage" as const, element: "SpriteData" },
                    eids: { type: "storage" as const, element: "u32" },
                    transforms: { type: "storage" as const, element: "Xform" },
                    spriteAtlas: { type: "texture-2d-array" as const },
                    spriteSamp: { type: "sampler" as const },
                },
            });
            registerSurface(state, spriteSurface(b));
        }

        if (!Compute.device) return;
        mesh({ name: "spriteQuad", vertices: QUAD_VERTS, indices: QUAD_INDICES });
    },

    // the atlas builds in warm, not initialize: warm runs after EVERY plugin's initialize, so any plugin
    // (e.g. orrstead's gauges) can register images in its own initialize with no pre-run call. Draws bind
    // the atlas at frame 1 (after warm), so the timing is safe.
    async warm() {
        if (!Compute.device) return;
        const device = Compute.device;

        // fetch every registered source to a Blob (a failed fetch becomes the transparent-pixel
        // placeholder so layer indices stay aligned with image ids), then build the array. Zero
        // images — or a decode-less host (bun test has a device but no createImageBitmap) — gets
        // a 1×1 fallback array instead, so the draws always bind cleanly
        if (Images.size > 0 && typeof createImageBitmap !== "undefined") {
            const blobs = await Promise.all(
                Array.from({ length: Images.size }, async (_, id) => {
                    const source = Images.get(Images.name(id)!)!.source;
                    if (typeof source !== "string") return source;
                    try {
                        const res = await fetch(source);
                        if (!res.ok) throw new Error(`${res.status}`);
                        return await res.blob();
                    } catch (e) {
                        console.warn(`[Sprite] image ${id} (${source}) failed to load:`, e);
                        return transparentPixel();
                    }
                }),
            );
            _atlas = await imageArray(device, blobs);
        } else {
            _atlas = device.createTexture({
                label: "sprite-atlas-fallback",
                size: { width: 1, height: 1, depthOrArrayLayers: 1 },
                format: "rgba8unorm-srgb",
                usage: GPUTextureUsage.TEXTURE_BINDING,
            });
        }
        Compute.textures.set("spriteAtlas", _atlas);
        // clamp-to-edge (the default), not repeat — a wrapping icon bleeds its opposite edge
        _sampler = device.createSampler({
            label: "sprite",
            magFilter: "linear",
            minFilter: "linear",
            mipmapFilter: "linear",
        });
        Compute.samplers.set("spriteSamp", _sampler);

        resetPack();
        _sig = -1;
        _spriteBuf = device.createBuffer({
            label: "kitchen-sprites",
            size: INITIAL * SPRITE_BYTES,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        Compute.buffers.set("spriteData", _spriteBuf);
        _eidsBuf = device.createBuffer({
            label: "kitchen-sprite-eids",
            size: INITIAL * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        const quad = Meshes.get("spriteQuad");
        if (quad) quad.bindings = { ...quad.bindings, eids: _eidsBuf };
        _argBuf = device.createBuffer({
            label: "kitchen-sprite-args",
            // one DrawIndexedIndirect record per bucket; COPY_SRC so a gym Mirror can read back instanceCount
            size: BUCKETS * 20,
            usage: GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
        });
    },

    dispose() {
        _spriteBuf?.destroy();
        _eidsBuf?.destroy();
        _argBuf?.destroy();
        _atlas?.destroy();
        _spriteBuf = null;
        _eidsBuf = null;
        _argBuf = null;
        _atlas = null;
        _sampler = null;
    },
};
