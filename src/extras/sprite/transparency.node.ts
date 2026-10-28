import { expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import * as d from "typegpu/data";
import { gpuScene } from "../../../scripts/gpu.fixture";
import { Meshes, MeshInstance } from "../../core/mesh";
import {
    Anchor,
    attachTexture,
    Billboard,
    Camera,
    CameraMode,
    captureTexture,
    Tonemapping,
    TonemappingMethod,
    TransparentPhases,
} from "../../core/rendering";
import { ImageBindings } from "../../core/rendering/image";
import { Transform } from "../../core/transform";
import type { Plugin, World } from "../../engine";
import {
    AlphaMode,
    MaterialPlugin,
    MeshMaterial,
    materialType,
    StandardRenderer,
} from "../../standard/rendering";
import { Sprite, SpriteAlphaMode, SpritePlugin } from "./index";

setDefaultTimeout(TIMEOUT.node);
const BlueInput = d.struct({ color: d.vec4f });
const Blue = materialType({
    name: "SpriteSortBlue",
    parameters: BlueInput,
    defaults: { color: d.vec4f(0, 0, 1, 0.5) },
    fragment: (ctx) => {
        "use gpu";
        return d.vec4f(ctx.parameters.color);
    },
});
const Proof: Plugin = {
    name: "SpriteSortProof",
    dependencies: [SpritePlugin, MaterialPlugin(Blue)],
};
const config = { defaults: false, plugins: [Proof] };

function camera(world: World) {
    const eid = world.create();
    world.add(eid, Transform, { translation: [0, 0, 5] });
    world.add(eid, Camera, { mode: CameraMode.Orthographic, size: 2, clearColor: 0, antialias: 0 });
    world.add(eid, StandardRenderer);
    world.add(eid, Tonemapping, { method: TonemappingMethod.None });
    attachTexture(world, eid, { width: 32, height: 32 });
    const atlas = world.gpu.device.createTexture({
        size: [1, 1, 1],
        format: "rgba8unorm-srgb",
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });
    world.own(atlas);
    world.gpu.device.queue.writeTexture(
        { texture: atlas },
        new Uint8Array([255, 255, 255, 255]),
        { bytesPerRow: 4 },
        [1, 1, 1],
    );
    return { view: eid, atlas };
}

function encode(linear: number): number {
    return Math.round(
        255 * (linear <= 0.0031308 ? linear * 12.92 : 1.055 * linear ** (1 / 2.4) - 0.055),
    );
}

const rotated = gpuScene(import.meta.path, config, (world) => {
    const { view, atlas } = camera(world);
    const sprite = world.create();
    world.add(sprite, Transform, { rotation: [0, Math.SQRT1_2, 0, Math.SQRT1_2] });
    world.add(sprite, Sprite, {
        size: [2, 2],
        billboard: Billboard.Screen,
        color: 0xff0000,
        opacity: 0.5,
        alphaMode: SpriteAlphaMode.Blend,
    });
    const mesh = world.resource(Meshes).add({
        name: "sprite-sort-pane",
        vertices: new Float32Array([
            -1, -1, 0, 0, 0, 0, 1, 0, 1, -1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, -1, 1, 0, 0,
            1, 0, 1, 0,
        ]),
        indices: new Uint32Array([0, 1, 2, 0, 2, 3]),
    });
    const pane = world.create();
    world.add(pane, Transform, { translation: [0, 0, -0.25] });
    world.add(pane, MeshInstance, { mesh });
    world.add(
        pane,
        MeshMaterial,
        world.resource(Blue).add({ color: d.vec4f(0, 0, 1, 0.5) }, { alphaMode: AlphaMode.Blend }),
    );
    return { world, view, atlas, pane, sprite };
});
test("a rotated Blend sprite sorts at its origin-centered sphere and composites over the farther pane", async () => {
    const {
        scene: { world, view, atlas, pane, sprite },
    } = rotated();
    world.resource(ImageBindings).images = atlas;
    world.step(0);
    const { rgba } = await captureTexture(world, view);
    const offset = (16 * 32 + 16) * 4;
    const alpha = 128 / 255;
    const expected = [encode(alpha), 0, encode(0.5 * (1 - alpha))];
    for (let channel = 0; channel < 3; channel++)
        expect(Math.abs(rgba[offset + channel]! - expected[channel]!)).toBeLessThanOrEqual(2);
    const phase = world.resource(TransparentPhases).get(view)!;
    expect([phase.item(0).stableId, phase.item(1).stableId]).toEqual([pane, sprite]);
});

const sharing = gpuScene(import.meta.path, config, (world) => {
    const { view, atlas } = camera(world);
    const owner = world.create();
    world.add(owner, Transform, { translation: [100, 0, 0] });
    world.add(owner, Sprite, { size: [10, 10], alphaMode: SpriteAlphaMode.Blend });
    return { world, view, atlas, owner };
});
test("a consumer sharing a large Sprite row stays visible at the frustum edge without its own Sprite", async () => {
    const {
        scene: { world, view, atlas, owner },
    } = sharing();
    world.resource(ImageBindings).images = atlas;
    world.step(0);
    const shared = world.create();
    world.add(shared, Transform, { translation: [4, 0, 0] });
    world.add(shared, MeshInstance, { mesh: world.storage(MeshInstance).mesh.get(owner) });
    world.add(shared, MeshMaterial, {
        type: world.storage(MeshMaterial).type.get(owner),
        material: world.storage(MeshMaterial).material.get(owner),
    });
    world.step(0);
    const phase = world.resource(TransparentPhases).get(view)!;
    expect(phase.length).toBe(1);
    expect(phase.item(0).stableId).toBe(shared);
    const { rgba } = await captureTexture(world, view);
    expect(rgba[(16 * 32 + 16) * 4]).toBeGreaterThan(0);
    world.storage(Sprite).size.set(owner, 1, 1);
    world.step(0);
    expect(world.resource(TransparentPhases).get(view)!.length).toBe(0);
    world.storage(Sprite).size.set(owner, 4, 4);
    world.add(owner, Anchor, { offset: Anchor.BOTTOM_RIGHT });
    world.step(0);
    expect(world.resource(TransparentPhases).get(view)!.length).toBe(1);
});

const offscreen = gpuScene(import.meta.path, config, (world) => {
    const { view, atlas } = camera(world);
    const sprite = world.create();
    world.add(sprite, Transform, { translation: [100, 0, 0] });
    world.add(sprite, Sprite, { alphaMode: SpriteAlphaMode.Blend });
    return { world, view, atlas, sprite };
});
test("an offscreen Blend sprite contributes no shared transparent phase item", () => {
    const {
        scene: { world, view, atlas, sprite },
    } = offscreen();
    world.resource(ImageBindings).images = atlas;
    world.step(0);
    expect(world.resource(TransparentPhases).get(view)!.length).toBe(0);
    world.storage(Transform).translation.x.set(sprite, 3.8);
    world.storage(Sprite).size.set(sprite, 2, 2);
    world.add(sprite, Anchor, { offset: Anchor.BOTTOM_RIGHT });
    world.step(0);
    expect(world.resource(TransparentPhases).get(view)!.length).toBe(1);
    world.remove(sprite, Anchor);
    world.step(0);
    expect(world.resource(TransparentPhases).get(view)!.length).toBe(0);
});
