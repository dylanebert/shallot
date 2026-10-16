import {
    AlphaMode,
    Camera,
    DirectionalLight,
    GlobalAmbientLight,
    lookAtRotation,
    Materials,
    Meshes,
    MeshInstance,
    MeshMaterial,
    MeshPlugin,
    type Plugin,
    type Resource,
    registerImage,
    StandardMaterial,
    StandardRenderer,
    sphere,
    Transform,
    type World,
} from "@dylanebert/shallot";
import {
    internText,
    Justify,
    Orbit,
    OrbitPlugin,
    Text,
    TextPlugin,
} from "@dylanebert/shallot/extras";
import { Billboard } from "@dylanebert/shallot/rendering";
import {
    BackgroundContext,
    backgroundLayout,
    CameraBackground,
    type MaterialHandle,
    registerBackground,
} from "@dylanebert/shallot/standard/rendering";
import tgpu from "typegpu";
import * as d from "typegpu/data";
import { CheckerMaterial, CheckerPlugin, WaveMaterial, WavePlugin, waveVector } from "./materials";

export const presets = [
    {
        name: "Plastic",
        baseColor: [0.72, 0.4, 0.05, 1],
        metallic: 0,
        perceptualRoughness: 0.38,
    },
    {
        name: "Rough metal",
        baseColor: [0.2, 0.19, 0.17, 1],
        metallic: 1,
        perceptualRoughness: 0.65,
    },
    {
        name: "Polished metal",
        baseColor: [0.2, 0.19, 0.17, 1],
        metallic: 1,
        perceptualRoughness: 0.12,
    },
    { name: "Glass", baseColor: [0.04, 0.43, 0.53, 0.28], metallic: 0, perceptualRoughness: 0.08 },
] as const;
export const Gallery: Resource<{
    camera: number;
    backing: number;
    spheres: number[];
    names: number[];
    checker?: MaterialHandle;
    wave?: MaterialHandle;
    time: number;
}> = {
    create: () => ({ camera: 0, backing: 0, spheres: [], names: [], time: 0 }),
};
function scene(world: World) {
    const gallery = world.resource(Gallery);
    const ambient = world.resource(GlobalAmbientLight);
    ambient.color = 0xd8e4ff;
    ambient.brightness = 520;
    const sun = world.create();
    world.add(sun, Transform, { rotation: lookAtRotation(0, 0, 0, -0.5, -1, -0.6) });
    world.add(sun, DirectionalLight, { color: 0xffe8cc, illuminance: 3200 });
    world.storage(DirectionalLight).shadowMapsEnabled.set(sun, 1);
    const camera = world.create();
    gallery.camera = camera;
    world.add(camera, Transform);
    world.add(camera, Camera, { clearColor: 0x191816, fov: 20 });
    world.add(camera, StandardRenderer);
    world.add(camera, Orbit, { distance: 32, yaw: 0.04, pitch: 0.2, pan: [0, 0.6, 0] });
}
function backdrop(world: World) {
    const background = registerBackground(world, {
        name: "Gallery",
        layout: backgroundLayout({}),
        fs: tgpu.fn(
            [BackgroundContext],
            d.vec3f,
        )(() => {
            "use gpu";
            return d.vec3f(0.0097212, 0.0091341, 0.0080232);
        }),
    });
    world.add(world.resource(Gallery).camera, CameraBackground, { name: background });
}
function surfaces(world: World) {
    const gallery = world.resource(Gallery);
    const materials = world.resource(Materials);
    const wood = registerImage(world, "/wood.jpg");
    const handles = presets.map((preset) =>
        materials.add(StandardMaterial(preset), {
            alphaMode: preset.name === "Glass" ? AlphaMode.Blend : AlphaMode.Opaque,
        }),
    );
    handles.push(
        materials.add(StandardMaterial({ baseColorTexture: wood, perceptualRoughness: 0.72 })),
    );
    handles.push(gallery.checker!, gallery.wave!);
    const geometry = sphere(0.65, 64, 32);
    const ball = world.resource(Meshes).add({ name: "Gallery sphere", ...geometry });
    const phases = new Float32Array(geometry.vertices.length / 8);
    for (let i = 0; i < phases.length; i++)
        phases[i] =
            geometry.vertices[i * 8]! * waveVector.x +
            geometry.vertices[i * 8 + 1]! * waveVector.y +
            geometry.vertices[i * 8 + 2]! * waveVector.z;
    const waveBall = world.resource(Meshes).add({
        name: "Wave sphere",
        ...geometry,
        attributes: { phase: { element: d.f32, data: phases } },
    });
    const stone = materials.add(
        StandardMaterial({
            baseColor: [0.085, 0.08, 0.072, 1],
            perceptualRoughness: 0.82,
        }),
    );
    handles.forEach((handle, i) => {
        const x = (i - 3) * 2.35;
        const eid = world.create();
        gallery.spheres.push(eid);
        world.add(eid, Transform, { translation: [x, 1.05, 0] });
        world.add(eid, MeshInstance, { mesh: i === 6 ? waveBall : ball });
        world.add(eid, MeshMaterial, handle);
        const label = world.create();
        gallery.names.push(label);
        world.add(label, Transform, { translation: [x, 2.15, 0] });
        world.add(label, Text, {
            content: internText(
                world,
                [
                    ...presets.map((preset) => preset.name),
                    "Wood",
                    "Scrolling checker",
                    "Vertex wave",
                ][i]!,
            ),
            billboard: Billboard.Screen,
            fontSize: 0.17,
            maxWidth: 1.1,
            justify: Justify.Center,
            anchor: [0.5, 0],
            color: 0xeee9df,
        });
        const plinth = world.create();
        world.add(plinth, Transform, { translation: [x, 0.1, 0], scale: [1.45, 0.6, 1.45] });
        world.add(plinth, MeshInstance);
        world.add(plinth, MeshMaterial, stone);
    });
    const backing = world.create();
    gallery.backing = backing;
    world.add(backing, Transform, { translation: [0, 1.05, -0.85], scale: [0.65, 0.8, 0.15] });
    world.add(backing, MeshInstance);
    world.add(
        backing,
        MeshMaterial,
        materials.add(
            StandardMaterial({
                baseColor: [0.65, 0.075, 0.09, 1],
                perceptualRoughness: 0.7,
            }),
        ),
    );
    const ground = world.create();
    world.add(ground, Transform, { translation: [0, -0.26, 0], scale: [21, 0.1, 8] });
    world.add(ground, MeshInstance);
    world.add(ground, MeshMaterial, stone);
}
export const AnimationPlugin: Plugin = {
    name: "GalleryAnimation",
    dependencies: [CheckerPlugin, WavePlugin],
    initialize(world) {
        const gallery = world.resource(Gallery);
        gallery.checker = world.resource(CheckerMaterial).add({
            darkColor: d.vec3f(0.042, 0.039, 0.035),
            lightColor: d.vec3f(0.65, 0.075, 0.09),
        });
        gallery.wave = world.resource(WaveMaterial).add({ time: 0 });
    },
    systems: [
        {
            name: "Gallery/Animate",
            update(world) {
                const gallery = world.resource(Gallery);
                gallery.time += world.time.deltaTime;
                world.resource(WaveMaterial).update(gallery.wave!, { time: gallery.time });
            },
        },
    ],
};
export const GalleryPlugin: Plugin = {
    name: "Gallery",
    dependencies: [MeshPlugin, OrbitPlugin, TextPlugin, AnimationPlugin],
    initialize(world) {
        scene(world);
        surfaces(world);
        backdrop(world);
    },
};
