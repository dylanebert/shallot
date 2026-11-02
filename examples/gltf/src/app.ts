import {
    Camera,
    DirectionalLight,
    GlobalAmbientLight,
    lookAtRotation,
    MeshPlugin,
    type Plugin,
    type Resource,
    StandardRenderer,
    Transform,
    type World,
} from "@dylanebert/shallot";
import { DiagnosticsOverlayPlugin, Orbit, OrbitPlugin } from "@dylanebert/shallot/extras";
import {
    type Gltf,
    GltfPlugin,
    type GltfSceneInstance,
    loadGltfScene,
    poseGltfScene,
    spawnGltfScene,
} from "@dylanebert/shallot/gltf";
import { fitBounds, sceneAngles, sceneBounds } from "./framing";
import { initialScene, modelPath, type Scene, scenes, truckPlacement } from "./scenes";

export const Viewer: Resource<{
    models: Map<Scene, Gltf>;
    loads: Map<Scene, Promise<void>>;
    selection: number;
    entities: number[];
    camera: number;
    sun: number;
    bounds: Gltf["bounds"] | null;
    active: Scene | null;
    clocks: Map<Scene, number>;
    instance: GltfSceneInstance | null;
}> = {
    create: () => ({
        models: new Map(),
        loads: new Map(),
        selection: 0,
        entities: [],
        camera: 0,
        sun: 0,
        bounds: null,
        active: null,
        clocks: new Map(),
        instance: null,
    }),
};
export function frameScene(world: World, aspect: number) {
    const viewer = world.resource(Viewer);
    if (viewer.bounds) fitBounds(world, viewer.camera, viewer.bounds, aspect);
}

export async function selectScene(world: World, scene: Scene, aspect = 1) {
    const viewer = world.resource(Viewer);
    const selection = ++viewer.selection;
    if (!viewer.models.has(scene)) await (viewer.loads.get(scene) ?? loadScene(world, scene));
    if (viewer.selection !== selection) return;
    for (const eid of viewer.entities) world.destroy(eid);
    for (const eid of viewer.instance?.lightEntities ?? []) world.destroy(eid);
    const lit = viewer.models.get(scene)!.lights.length > 0;
    world.resource(GlobalAmbientLight).brightness = lit ? 0 : 380;
    if (viewer.sun) world.storage(DirectionalLight).illuminance.set(viewer.sun, lit ? 0 : 6000);
    viewer.active = scene;
    viewer.instance = spawnGltfScene(
        world,
        viewer.models.get(scene)!,
        scene === "CesiumMilkTruck" ? truckPlacement : {},
    );
    viewer.entities = viewer.instance.entities;
    animateScene(world, 0);
    viewer.bounds = sceneBounds(viewer.models.get(scene)!, scene);
    const view = sceneAngles(scene);
    const orbit = world.storage(Orbit);
    orbit.yaw.set(viewer.camera, view.yaw);
    orbit.pitch.set(viewer.camera, view.pitch);
    frameScene(world, aspect);
}
export function animateScene(world: World, delta: number): void {
    const viewer = world.resource(Viewer);
    if (!viewer.active || !viewer.instance) return;
    const animation = viewer.models.get(viewer.active)!.animations[0];
    if (!animation) return;
    let duration = 0;
    for (const sampler of animation.samplers)
        duration = Math.max(duration, sampler.input[sampler.input.length - 1]);
    const time = duration > 0 ? ((viewer.clocks.get(viewer.active) ?? 0) + delta) % duration : 0;
    viewer.clocks.set(viewer.active, time);
    poseGltfScene(world, viewer.instance, animation, time);
}

function loadScene(
    world: World,
    scene: Scene,
    onProgress?: (progress: number) => void,
): Promise<void> {
    const viewer = world.resource(Viewer);
    const load = loadGltfScene(world, new URL(modelPath(scene), document.baseURI), onProgress).then(
        (model) => {
            viewer.models.set(scene, model);
        },
    );
    viewer.loads.set(scene, load);
    return load;
}

export function loadRemainingScenes(world: World): void {
    const viewer = world.resource(Viewer);
    for (const scene of Object.keys(scenes) as Scene[]) {
        if (!viewer.loads.has(scene)) void loadScene(world, scene);
    }
}

export const AppPlugin: Plugin = {
    name: "GltfApp",
    dependencies: [OrbitPlugin, MeshPlugin, GltfPlugin, DiagnosticsOverlayPlugin],
    systems: [
        {
            name: "Gltf/Animate",
            group: "fixed",
            update(world) {
                animateScene(world, world.time.deltaTime);
            },
        },
    ],
    async initialize(world, progress) {
        const viewer = world.resource(Viewer);
        const ambient = world.resource(GlobalAmbientLight);
        ambient.color = 0xd8e4ff;
        ambient.brightness = 380;
        const sun = world.create();
        viewer.sun = sun;
        world.add(sun, Transform, { rotation: lookAtRotation(0, 0, 0, 1, -2, -0.35) });
        world.add(sun, DirectionalLight, { color: 0xffe8cc, illuminance: 6000 });
        world.storage(DirectionalLight).shadowMapsEnabled.set(sun, 1);
        viewer.camera = world.create();
        world.add(viewer.camera, Camera, { clearColor: 0x191816 });
        world.add(viewer.camera, StandardRenderer);
        world.add(viewer.camera, Transform);
        world.add(viewer.camera, Orbit);
        const requested = new URL(location.href).searchParams.get("scene");
        const scene =
            requested && Object.hasOwn(scenes, requested) ? (requested as Scene) : initialScene;
        await loadScene(world, scene, progress);
        const canvas = document.querySelector("canvas")!;
        await selectScene(world, scene, canvas.clientWidth / canvas.clientHeight);
    },
};
