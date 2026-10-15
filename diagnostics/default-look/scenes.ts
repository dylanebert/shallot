import {
    Camera,
    DirectionalLight,
    GlobalAmbientLight,
    lookAtRotation,
    Materials,
    MeshInstance,
    MeshMaterial,
    type Plugin,
    PointLight,
    SpotLight,
    StandardMaterial,
    StandardRenderer,
    SunDisk,
    Transform,
    type World,
} from "@dylanebert/shallot";
import { internText, Sky, SkyPlugin, Text, TextPlugin } from "@dylanebert/shallot/extras";
import {
    capsule,
    cube,
    Meshes,
    type MeshHandle,
    MeshPlugin,
    sphere,
} from "@dylanebert/shallot/mesh";
import {
    Backgrounds,
    CameraBackground,
    type MaterialHandle,
} from "@dylanebert/shallot/standard/rendering";
import type { LookPalette } from "./palette";

export interface SceneDefinition {
    id: string;
    title: string;
    description: string;
    build(world: World, palette: LookPalette): void;
    plugins?: readonly Plugin[];
    ui?: boolean;
}

type Vec3 = readonly [number, number, number];
type MaterialRole = keyof LookPalette["materials"];

interface SceneBuilder {
    box(role: MaterialRole, position: Vec3, scale: Vec3, roughness?: number): number;
    primitive(
        role: MaterialRole,
        mesh: MeshHandle,
        position: Vec3,
        scale: Vec3,
        roughness?: number,
    ): number;
}

function builder(world: World, palette: LookPalette): SceneBuilder {
    const materials = world.resource(Materials);
    const handles = new Map<string, MaterialHandle>();
    function material(role: MaterialRole, roughness: number): MaterialHandle {
        const key = `${role}/${roughness}`;
        let handle = handles.get(key);
        if (handle === undefined) {
            handle = materials.add(
                StandardMaterial({
                    baseColor: palette.materials[role] ?? palette.materials.greyboxTower,
                    perceptualRoughness: roughness,
                }),
            );
            handles.set(key, handle);
        }
        return handle;
    }
    function draw(
        role: MaterialRole,
        mesh: MeshHandle | undefined,
        position: Vec3,
        scale: Vec3,
        roughness: number,
    ): number {
        const eid = world.create();
        world.add(eid, MeshInstance, mesh ? { mesh } : {});
        world.add(eid, MeshMaterial, material(role, roughness));
        world.add(eid, Transform, {
            translation: [position[0], position[1], position[2], 0],
            scale: [scale[0], scale[1], scale[2], 0],
        });
        return eid;
    }
    return {
        box(role, position, scale, roughness = 0.82) {
            return draw(role, undefined, position, scale, roughness);
        },
        primitive(role, mesh, position, scale, roughness = 0.38) {
            return draw(role, mesh, position, scale, roughness);
        },
    };
}

function camera(
    world: World,
    palette: LookPalette,
    position: Vec3,
    target: Vec3,
    fov = 48,
    background?: string,
): void {
    const eid = world.create();
    const rotation = lookAtRotation(...position, ...target);
    world.add(eid, Camera, { clearColor: palette.clearColor, fov });
    world.add(eid, StandardRenderer);
    if (background) {
        world.add(eid, CameraBackground, { name: world.resource(Backgrounds).id(background) });
    }
    world.add(eid, Transform, {
        translation: [position[0], position[1], position[2], 0],
        rotation,
    });
}

function exampleLighting(
    world: World,
    shadows = false,
    sunDisk = false,
    travel: Vec3 = [-0.5, -1, -0.6],
): void {
    const ambient = world.resource(GlobalAmbientLight);
    ambient.color = 0xd8e4ff;
    ambient.brightness = 520;

    const sun = world.create();
    world.add(sun, Transform, { rotation: lookAtRotation(0, 0, 0, ...travel) });
    world.add(sun, DirectionalLight, {
        color: 0xffe8cc,
        illuminance: 3200,
        shadowMapsEnabled: shadows ? 1 : 0,
        numCascades: 1,
        maximumDistance: 40,
    });
    if (sunDisk) world.add(sun, SunDisk);
}

function primitivesAndGreybox(world: World, palette: LookPalette): void {
    exampleLighting(world, true);
    camera(world, palette, [5, 4.5, 7.5], [0, 0.7, 0]);
    const scene = builder(world, palette);
    const cubeMesh = world.resource(Meshes).add({ name: "default-look/cube", ...cube() });
    const sphereMesh = world.resource(Meshes).add({ name: "default-look/sphere", ...sphere() });
    const capsuleMesh = world.resource(Meshes).add({
        name: "default-look/capsule",
        ...capsule(0.42, 1),
    });

    scene.box("greyboxGround", [0, -0.38, 0], [7, 0.25, 7]);
    for (let index = 0; index < 4; index++) {
        const x = -4.3 + index * 0.9;
        const height = 0.3 + index * 0.42;
        scene.box(index === 3 ? "greyboxLift" : "greyboxStep", [x, height, -2.5], [0.8, 0.25, 0.8]);
    }
    scene.box("greyboxTower", [2.5, 0.65, -2.2], [0.75, 0.9, 0.8]);
    scene.box("trim", [2.5, 1.65, -2.2], [0.75, 0.1, 0.8]);
    scene.box("greyboxPerch", [2.5, 2.25, -2.2], [0.55, 0.5, 0.55]);

    scene.box("greyboxStep", [-3, 0.12, 2.6], [0.95, 0.18, 0.95]);
    scene.box("greyboxStep", [0, 0.12, 2.6], [0.95, 0.18, 0.95]);
    scene.box("greyboxStep", [3, 0.12, 2.6], [0.95, 0.18, 0.95]);
    scene.primitive("blue", cubeMesh, [-3, 0.95, 2.6], [1, 1, 1]);
    scene.primitive("orange", sphereMesh, [0, 0.95, 2.6], [1.05, 1.05, 1.05]);
    scene.primitive("greyboxLift", capsuleMesh, [3, 1.05, 2.6], [1, 1.25, 1]);
}

function sunlitExterior(world: World, palette: LookPalette): void {
    exampleLighting(world, true, true, [0.4, -0.25, 0.8]);
    camera(world, palette, [6, 3.5, 8], [0, 1.5, -0.5], 46, "sky");
    const sky = world.create();
    world.add(sky, Sky, {
        zenith: 0x89b6e9,
        horizon: 0xc4cdda,
        band: 0.08,
        cloudCoverage: 0,
        cloudDensity: 0,
        hazeDensity: 0.003,
    });
    const scene = builder(world, palette);
    const sphereMesh = world
        .resource(Meshes)
        .add({ name: "default-look/exterior-sphere", ...sphere() });

    scene.box("blueGround", [0, -0.35, 0], [8, 0.3, 8]);
    scene.primitive("greyboxPerch", sphereMesh, [-4, 0.15, -3], [3, 1.8, 2.4]);
    scene.primitive("greyboxTower", sphereMesh, [3.8, 0.1, -3.2], [3.4, 2.2, 2.5]);
    scene.box("trim", [-2.8, 0.65, 1.2], [0.35, 1.1, 0.35]);
    scene.primitive("orange", sphereMesh, [-2.8, 2.1, 1.2], [1.4, 1.4, 1.4]);
    scene.box("trim", [2.6, 0.65, -0.7], [0.35, 1.1, 0.35]);
    scene.primitive("blue", sphereMesh, [2.6, 2.1, -0.7], [1.5, 1.5, 1.5]);
    scene.box("greyboxLift", [0, 0.3, 2.3], [1.2, 0.28, 0.8]);
    scene.box("orange", [0, 0.84, 2.3], [0.42, 0.28, 0.42]);
}

function pointLitInterior(world: World, palette: LookPalette): void {
    const ambient = world.resource(GlobalAmbientLight);
    ambient.color = 0xd8e4ff;
    ambient.brightness = 40;
    camera(world, palette, [0, 1.6, 3.4], [0, 0.3, -0.7], 54);
    const scene = builder(world, palette);
    scene.box("greyboxGround", [0, -1.1, 0], [6.6, 0.2, 7]);
    scene.box("greyboxTower", [0, 1.1, -3.3], [6.6, 4.2, 0.2]);
    scene.box("greyboxPerch", [-3.2, 1.1, 0], [0.2, 4.2, 6.6]);
    scene.box("greyboxPerch", [3.2, 1.1, 0], [0.2, 4.2, 6.6]);
    scene.box("greyboxTower", [0, 1.1, 3.3], [6.6, 4.2, 0.2]);
    scene.box("greyboxTower", [0, 3.3, 0], [6.6, 0.2, 6.6]);

    scene.box("blueGround", [-1.8, -0.55, -1.3], [0.7, 0.9, 0.7]);
    scene.box("greyboxLift", [0, -0.5, -1.45], [0.9, 1, 0.8], 0.48);
    scene.box("greyboxStep", [1.25, -0.55, -0.9], [0.65, 0.9, 0.7], 0.58);

    const pointPosition = [-1.4, 1.8, 0.3] as const;
    const point = world.create();
    world.add(point, Transform, { translation: [...pointPosition, 0] });
    world.add(point, PointLight, {
        color: 0xffe8cc,
        intensity: 315225.92,
        range: 9,
        radius: 0.18,
        shadowMapsEnabled: 1,
    });

    const spotPosition = [2.5, 1.2, 1.6] as const;
    const spotTarget = [1.25, -0.55, -0.9] as const;
    const spot = world.create();
    world.add(spot, Transform, {
        translation: [...spotPosition, 0],
        rotation: lookAtRotation(...spotPosition, ...spotTarget),
    });
    world.add(spot, SpotLight, {
        color: 0xffe8cc,
        intensity: 315225.92,
        range: 8,
        innerAngle: 18,
        outerAngle: 34,
        shadowMapsEnabled: 1,
    });
}

function textAndUi(world: World, palette: LookPalette): void {
    exampleLighting(world);
    camera(world, palette, [0, 1.25, 6.5], [0, 1.25, 0], 43);
    const scene = builder(world, palette);
    scene.box("blueGround", [0, -0.95, -0.4], [5.8, 0.28, 3.2]);
    scene.box("greyboxTower", [0, 1.25, -1.4], [5.1, 1.7, 0.22]);
    scene.box("orange", [-2.2, 0.65, 0.1], [0.8, 0.52, 0.8], 0.44);
    scene.box("greyboxStep", [0, 0.65, 0.1], [0.8, 0.52, 0.8], 0.66);
    scene.box("blue", [2.2, 0.65, 0.1], [0.8, 0.52, 0.8], 0.38);

    const title = world.create();
    world.add(title, Transform, { translation: [-2.65, 2.1, -0.95, 0] });
    world.add(title, Text, {
        content: internText(world, "SHALLOT"),
        fontSize: 0.9,
        color: palette.textColor,
    });
    const subtitle = world.create();
    world.add(subtitle, Transform, { translation: [-2.6, 1.45, -0.94, 0] });
    world.add(subtitle, Text, {
        content: internText(world, "LIGHT / SHAPE / PLAY"),
        fontSize: 0.25,
        color: palette.textColor,
    });
}

export const SCENES: readonly SceneDefinition[] = [
    {
        id: "primitives-greybox",
        title: "Primitives and greybox",
        description: "Built-in shapes, a small blockout, and a warm directional light.",
        build: primitivesAndGreybox,
    },
    {
        id: "sunlit-exterior",
        title: "Sunlit exterior",
        description: "Simple outdoor forms under the shared warm sun and cool ambient fill.",
        build: sunlitExterior,
        plugins: [SkyPlugin],
    },
    {
        id: "point-lit-interior",
        title: "Point-lit interior",
        description: "A closed room with warm point- and spot-lit shadow edges.",
        build: pointLitInterior,
    },
    {
        id: "text-ui",
        title: "Text and UI",
        description: "World-space type and a canvas-bounded interface over the scene.",
        build: textAndUi,
        ui: true,
    },
];

export const DIAGNOSTIC_PLUGINS = [MeshPlugin, TextPlugin] as const;
