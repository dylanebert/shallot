import {
    Camera,
    DirectionalLight,
    GlobalAmbientLight,
    lookAtRotation,
    Materials,
    MeshInstance,
    MeshMaterial,
    type Plugin,
    type Resource,
    StandardMaterial,
    StandardRenderer,
    Transform,
    type World,
} from "@dylanebert/shallot";
import {
    ChromaticAberration,
    EffectStackPlugin,
    LensDistortion,
    Orbit,
    OrbitPlugin,
    Vignette,
} from "@dylanebert/shallot/extras";

export const effects = [
    { name: "Lens distortion", component: LensDistortion },
    { name: "Chromatic aberration", component: ChromaticAberration },
    { name: "Vignette", component: Vignette },
] as const;

export const Scene: Resource<{ camera: number }> = {
    create: () => ({ camera: -1 }),
};

export function setEffect(world: World, index: number, enabled: boolean): void {
    const camera = world.resource(Scene).camera;
    const effect = effects[index].component;
    if (enabled) {
        if (effect === Vignette)
            world.add(camera, Vignette, { intensity: 0.65, radius: 0.9, smoothness: 2 });
        else world.add(camera, effect);
    } else world.remove(camera, effect);
}

function makeScene(world: World): void {
    const ambient = world.resource(GlobalAmbientLight);
    ambient.color = 0xd8e4ff;
    ambient.brightness = 380;

    const sun = world.create();
    world.add(sun, Transform, { rotation: lookAtRotation(0, 0, 0, 1, -2, -0.35) });
    world.add(sun, DirectionalLight, { color: 0xffe8cc, illuminance: 6000 });
    world.storage(DirectionalLight).shadowMapsEnabled.set(sun, 1);

    const camera = world.create();
    world.resource(Scene).camera = camera;
    world.add(camera, Camera, { clearColor: 0x191816 });
    world.add(camera, StandardRenderer);
    world.add(camera, Orbit, {
        pan: [0, 1.3, 0],
        distance: Math.hypot(2.1, 10),
        yaw: 0,
        pitch: Math.atan2(2.1, 10),
    });
    world.add(camera, Transform);

    const materials = world.resource(Materials);
    const carbon = materials.add(
        StandardMaterial({
            baseColor: [0.085, 0.08, 0.072, 1],
            perceptualRoughness: 0.82,
        }),
    );
    const stone = materials.add(
        StandardMaterial({
            baseColor: [0.15, 0.14, 0.125, 1],
            perceptualRoughness: 0.65,
        }),
    );
    const amber = materials.add(
        StandardMaterial({
            baseColor: [0.72, 0.4, 0.05, 1],
            perceptualRoughness: 0.38,
        }),
    );
    type Block = {
        position: [number, number, number];
        scale: [number, number, number];
        material: typeof carbon;
    };
    function block({ position, scale, material }: Block): number {
        const eid = world.create();
        world.add(eid, MeshInstance);
        world.add(eid, MeshMaterial, material);
        world.add(eid, Transform, { translation: position, scale });
        return eid;
    }
    block({ position: [0, -0.2, 0], scale: [20, 0.4, 18], material: carbon });
    // Receding rectangular frames put straight, contrasting edges across the view.
    for (const z of [-3, 0, 3]) {
        for (const x of [-5, 5])
            block({ position: [x, 2.4, z], scale: [0.22, 4.8, 0.22], material: stone });
        block({ position: [0, 4.8, z], scale: [10.22, 0.22, 0.22], material: stone });
    }
    const subjects: Block[] = [
        { position: [0, 1.1, 0], scale: [1.6, 2.2, 1.6], material: amber },
        { position: [-3.4, 0.65, 1.8], scale: [1.1, 1.3, 1.1], material: amber },
        { position: [3.4, 0.65, 1.8], scale: [1.1, 1.3, 1.1], material: amber },
    ];
    for (const subject of subjects) block(subject);
    for (let x = -6; x <= 6; x++)
        block({ position: [x, 0.01, 0], scale: [0.025, 0.02, 12], material: stone });
    for (let z = -6; z <= 6; z++)
        block({ position: [0, 0.01, z], scale: [12, 0.02, 0.025], material: stone });
}

export const ScenePlugin: Plugin = {
    name: "PostProcessingScene",
    dependencies: [EffectStackPlugin, OrbitPlugin],
    initialize: makeScene,
};
