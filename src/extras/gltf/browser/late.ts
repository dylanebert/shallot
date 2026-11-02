import {
    attachTexture,
    Camera,
    captureTexture,
    DirectionalLight,
    imageTextures,
    prepareImages,
    registerImage,
} from "../../../core/rendering";
import { Transform } from "../../../core/transform";
import { type App, createApp } from "../../../engine/app";
import type { World } from "../../../engine/ecs";
import {
    Materials,
    MeshMaterial,
    MeshRenderPlugin,
    StandardMaterial,
    StandardRenderer,
    StandardRenderingPlugin,
} from "../../../standard/rendering";
import { type Gltf, loadGltfScene, spawnGltfScene } from "../scene";

const source = new URL("../fixtures/BoxTextured/glTF/BoxTextured.gltf", import.meta.url);
const apps: App[] = [];
async function subject(atWarm: boolean) {
    let scene: Gltf | undefined;
    let camera = 0;
    const app = await createApp({
        defaults: false,
        plugins: [
            MeshRenderPlugin,
            StandardRenderingPlugin,
            {
                name: "LateTextures",
                dependencies: [MeshRenderPlugin, StandardRenderingPlugin],
                async initialize(world) {
                    camera = world.create();
                    world.add(camera, Transform, { translation: [0, 0, 4] });
                    world.add(camera, Camera, { antialias: 0 });
                    world.add(camera, StandardRenderer);
                    attachTexture(world, camera, { width: 64, height: 64 });
                    const sun = world.create();
                    world.add(sun, Transform);
                    world.add(sun, DirectionalLight, { illuminance: 10000 });
                    if (atWarm) scene = await loadGltfScene(world, source);
                },
            },
        ],
    });
    apps.push(app);
    return { world: app.world, camera, scene };
}
async function capture(world: World, camera: number) {
    world.step(0);
    world.step(0);
    return (await captureTexture(world, camera)).rgba;
}
function equal(a: Uint8ClampedArray, b: Uint8ClampedArray) {
    return a.every((value, i) => value === b[i]);
}
try {
    const warm = await subject(true);
    spawnGltfScene(warm.world, warm.scene!);
    const expected = await capture(warm.world, warm.camera);
    const late = await subject(false);
    const scene = await loadGltfScene(late.world, source);
    spawnGltfScene(late.world, scene);
    const actual = await capture(late.world, late.camera);
    // Keep the fixture's material and sampler; replace only its image with the role's 1x1 fallback.
    const textures = late.world.resource(imageTextures);
    const image = late.world
        .resource(Materials)
        .get(scene.primitives[0].material)!.baseColorTexture;
    const native = textures.textures[0][image];
    textures.textures[0][image] = textures.fallback[0];
    const blank = await capture(late.world, late.camera);
    textures.textures[0][image] = native;

    // Two stable batches: only the left one's image identity will change after decode.
    const world = warm.world;
    for (const eid of [...world.query([MeshMaterial])]) world.destroy(eid);
    const fixed = spawnGltfScene(world, warm.scene!, {
        translation: [0.8, 0, 0],
        scale: [0.6, 0.6, 0.6],
    });
    const pending = spawnGltfScene(world, warm.scene!, {
        translation: [-0.8, 0, 0],
        scale: [0.6, 0.6, 0.6],
    });
    const decode = globalThis.createImageBitmap;
    let release!: () => void;
    const held = new Promise<void>((resolve) => {
        release = resolve;
    });
    let entered!: () => void;
    const decoding = new Promise<void>((resolve) => {
        entered = resolve;
    });
    globalThis.createImageBitmap = (async (...args: Parameters<typeof decode>) => {
        entered();
        await held;
        return decode(...args);
    }) as typeof decode;
    let before: Uint8ClampedArray;
    try {
        const image = registerImage(
            world,
            await (
                await fetch(
                    new URL("../fixtures/BoxTextured/glTF/CesiumLogoFlat.png", import.meta.url),
                )
            ).blob(),
        );
        const material = world
            .resource(Materials)
            .add(StandardMaterial({ baseColorTexture: image }));
        for (const eid of pending.entities) world.add(eid, MeshMaterial, material);
        await decoding;
        before = await capture(world, warm.camera);
        release();
        await prepareImages(world);
    } finally {
        release();
        globalThis.createImageBitmap = decode;
    }
    const device = world.gpu.device;
    const create = device.createBindGroup.bind(device);
    let groups = 0;
    device.createBindGroup = (...args) => {
        groups++;
        return create(...args);
    };
    const after = await capture(world, warm.camera);
    const rebuilt = groups;
    await capture(world, warm.camera);
    const steady = groups - rebuilt;
    device.createBindGroup = create;
    let changed = false;
    let unchanged = true;
    for (let y = 0; y < 64; y++)
        for (let x = 0; x < 64; x++)
            for (let c = 0; c < 4; c++) {
                const i = (y * 64 + x) * 4 + c;
                if (x < 32 && before![i] !== after[i]) changed = true;
                if (x >= 32 && before![i] !== after[i]) unchanged = false;
            }
    if (!fixed.entities.length) throw new Error("missing fixed material control");
    document.body.textContent = JSON.stringify({
        equal: equal(actual, expected),
        textured: !equal(actual, blank),
        changed,
        unchanged,
        groups: rebuilt,
        steady,
    });
} finally {
    for (const app of apps) app.dispose();
}
