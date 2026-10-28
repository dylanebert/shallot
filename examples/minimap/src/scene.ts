import {
    Body,
    BodyType,
    Camera,
    CameraMode,
    Character,
    DirectionalLight,
    GlobalAmbientLight,
    lookAtRotation,
    Materials,
    MeshInstance,
    MeshMaterial,
    NotShadowCaster,
    type Plugin,
    type Resource,
    Shape,
    ShapeKind,
    StandardMaterial,
    StandardPhysicsPlugin,
    StandardRenderer,
    Transform,
    Viewports,
    type World,
} from "@dylanebert/shallot";
import { CameraViewport, RenderLayers } from "@dylanebert/shallot/rendering";
import { LocalPlayer, Player, PlayerPlugin, UpdatePlayerControlSystem } from "./player";

export const Scene: Resource<{
    camera: number;
    minimap: number;
    player: number;
    marker: number;
    orangeCube: number;
}> = {
    create: () => ({ camera: 0, minimap: 0, player: 0, marker: 0, orangeCube: 0 }),
};
// Copied dev-orange linear albedos: primary, secondary, lift, embedded, ground and dark.
const ORANGE = [0.621407, 0.267918, 0.117568, 1] as const;
const BLUE = [0.152701, 0.366286, 0.704245, 1] as const;
const YELLOW = [0.416966, 0.351066, 0.085339, 1] as const;
const RED = [0.669826, 0.235983, 0.211473, 1] as const;
const GROUND = [0.384987, 0.337404, 0.267916, 1] as const;
const DARK = [0.038351, 0.037506, 0.036164, 1] as const;
function block(
    w: World,
    position: readonly [number, number, number],
    size: readonly [number, number, number],
    color: readonly [number, number, number, number],
    solid = true,
) {
    const e = w.create();
    w.add(e, Transform, { translation: position, scale: size });
    w.add(e, MeshInstance);
    w.add(e, MeshMaterial, w.resource(Materials).add(StandardMaterial({ baseColor: color })));
    if (solid) {
        w.add(e, Body, { type: BodyType.Static });
        w.add(e, Shape);
    }
    return e;
}
export const MinimapSystem = {
    name: "minimap",
    group: "simulation" as const,
    after: [UpdatePlayerControlSystem],
    update(w: World) {
        const s = w.resource(Scene),
            t = w.storage(Transform);
        t.translation.set(
            s.marker,
            t.translation.x.get(s.camera),
            4,
            t.translation.z.get(s.camera),
        );
        const viewport = w.resource(Viewports).get(0);
        if (!viewport) return;
        const width = Math.floor(viewport.cssWidth),
            height = Math.floor(viewport.cssHeight);
        const margin = Math.min(16, Math.floor(Math.min(width, height) / 8));
        const size = Math.max(
            0,
            Math.min(
                224,
                Math.floor(Math.min(width, height) * 0.4),
                width - 2 * margin,
                height - 2 * margin,
            ),
        );
        const v = w.storage(CameraViewport);
        v.x.set(s.minimap, width - margin - size);
        v.y.set(s.minimap, margin);
        v.width.set(s.minimap, size);
        v.height.set(s.minimap, size);
    },
};
export const ScenePlugin: Plugin = {
    name: "Minimap",
    dependencies: [StandardPhysicsPlugin, PlayerPlugin],
    systems: [MinimapSystem],
    initialize(w) {
        const s = w.resource(Scene);
        w.resource(GlobalAmbientLight).color = 0xd8e4ff;
        w.resource(GlobalAmbientLight).brightness = 380;
        const sun = w.create();
        w.add(sun, Transform, { rotation: lookAtRotation(0, 0, 0, 1, -2, -1) });
        w.add(sun, DirectionalLight, { color: 0xffe8cc, illuminance: 6000, shadowMapsEnabled: 1 });
        w.add(sun, RenderLayers, { mask: 3 });
        s.camera = w.create();
        w.add(s.camera, Camera, { fov: 90, clearColor: 0x1c1917 });
        w.add(s.camera, StandardRenderer);
        w.add(s.camera, Transform, { translation: [0, 1.6, 7] });
        s.minimap = w.create();
        w.add(s.minimap, Camera, {
            mode: CameraMode.Orthographic,
            size: 11,
            order: 1,
            near: 0.1,
            far: 50,
        });
        w.add(s.minimap, StandardRenderer);
        w.add(s.minimap, CameraViewport, { x: 400, y: 16, width: 224, height: 224 });
        w.add(s.minimap, Transform, {
            translation: [0, 25, 0],
            rotation: lookAtRotation(0, 25, 0, 0, 0, 0, 0, 0, -1),
        });
        w.add(s.minimap, RenderLayers, { mask: 3 });
        s.player = w.create();
        w.add(s.player, Transform, { translation: [0, 0.9, 7] });
        w.add(s.player, Body, { type: BodyType.Kinematic });
        w.add(s.player, Shape, {
            kind: ShapeKind.Capsule,
            capsuleCenter1: [0, -0.55, 0],
            capsuleCenter2: [0, 0.55, 0],
            radius: 0.3,
        });
        w.add(s.player, Character);
        w.add(s.player, Player, { camera: s.camera });
        w.add(s.player, LocalPlayer);
        block(w, [0, -0.15, 0], [20, 0.3, 20], GROUND);
        const walls = [
            { p: [0, 0.6, -10], s: [20, 1.2, 0.3] },
            { p: [0, 0.6, 10], s: [20, 1.2, 0.3] },
            { p: [-10, 0.6, 0], s: [0.3, 1.2, 20] },
            { p: [10, 0.6, 0], s: [0.3, 1.2, 20] },
        ] as const;
        for (const wall of walls) block(w, wall.p, wall.s, DARK);
        s.orangeCube = block(w, [-4, 1, -3], [2, 2, 2], ORANGE);
        const cubes = [
            { p: [3, 1.5, -4], s: [3, 3, 3], c: BLUE },
            { p: [-5, 0.5, 3], s: [1, 1, 1], c: YELLOW },
            { p: [5, 0.75, 2], s: [1.5, 1.5, 1.5], c: RED },
            { p: [0, 0.5, -7], s: [1, 1, 1], c: DARK },
        ] as const;
        for (const cube of cubes) block(w, cube.p, cube.s, cube.c);
        s.marker = block(w, [0, 4, 7], [0.65, 0.2, 0.65], [1, 1, 1, 1], false);
        w.add(s.marker, RenderLayers, { mask: 2 });
        w.add(s.marker, NotShadowCaster);
    },
};
