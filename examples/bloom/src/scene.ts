import {
    Camera,
    lookAtRotation,
    Materials,
    Meshes,
    MeshInstance,
    MeshMaterial,
    type Plugin,
    type Resource,
    StandardMaterial,
    StandardRenderer,
    sphere,
    Tonemapping,
    TonemappingMethod,
    Transform,
    type World,
} from "@dylanebert/shallot";
import { Bloom, BloomPlugin } from "@dylanebert/shallot/bloom";

export const Scene: Resource<{ camera: number; spheres: number[] }> = {
    create: () => ({ camera: -1, spheres: [] }),
};

export function setBloom(world: World, enabled: boolean): void {
    const camera = world.resource(Scene).camera;
    if (enabled) world.add(camera, Bloom);
    else world.remove(camera, Bloom);
}

function makeScene(world: World): void {
    const camera = world.create();
    world.resource(Scene).camera = camera;
    world.add(camera, Camera, { clearColor: 0x000000 });
    world.add(camera, StandardRenderer);
    world.add(camera, Transform, {
        translation: [-8, 10, 20],
        rotation: lookAtRotation(-8, 10, 20, 0, 0, 0),
    });
    world.add(camera, Tonemapping, { method: TonemappingMethod.TonyMcMapface });
    setBloom(world, true);

    const mesh = world.resource(Meshes).add({ name: "Bloom sphere", ...sphere(0.4, 48, 24) });
    const materials = world.resource(Materials);
    const colors: [number, number, number][] = [
        [0, 0, 150],
        [1000, 1000, 1000],
        [50, 0, 0],
        [0, 0, 0],
    ];
    const surfaces = colors.map((emissive) =>
        materials.add(StandardMaterial({ baseColor: [0, 0, 0, 1], emissive })),
    );
    type Sphere = { x: number; z: number; kind: number };
    function spawn({ x, z, kind }: Sphere): number {
        const eid = world.create();
        const scale = [0.5, 0.1, 1, 1.5][kind];
        world.add(eid, MeshInstance, { mesh });
        world.add(eid, MeshMaterial, surfaces[kind]);
        world.add(eid, Transform, {
            translation: [x * 2, Math.sin(x * 2 + z * 2), z * 2],
            scale: [scale, scale, scale],
        });
        return eid;
    }
    for (let x = -5; x < 5; x++) {
        for (let z = -5; z < 5; z++) {
            // A coordinate hash keeps the six-way colour distribution repeatable.
            const hash = (Math.imul(x + 5, 73856093) ^ Math.imul(z + 5, 19349663)) >>> 0;
            world.resource(Scene).spheres.push(spawn({ x, z, kind: Math.min(hash % 6, 3) }));
        }
    }
}

export const ScenePlugin: Plugin = {
    name: "BloomScene",
    dependencies: [BloomPlugin],
    initialize: makeScene,
    systems: [
        {
            name: "bounceSpheres",
            group: "fixed",
            update(world) {
                const translation = world.storage(Transform).translation;
                for (const eid of world.resource(Scene).spheres)
                    translation.y.set(
                        eid,
                        Math.sin(
                            translation.x.get(eid) + translation.z.get(eid) + world.time.elapsed,
                        ),
                    );
            },
        },
    ],
};
