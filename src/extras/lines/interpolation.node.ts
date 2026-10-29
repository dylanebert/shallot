import { expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { gpuScene } from "../../../scripts/gpu.fixture";
import { MeshInstance } from "../../core/mesh";
import {
    attachTexture,
    Camera,
    captureTexture,
    Tonemapping,
    TonemappingMethod,
} from "../../core/rendering";
import { Transform } from "../../core/transform";
import type { System } from "../../engine";
import {
    Materials,
    MeshMaterial,
    MeshRenderPlugin,
    StandardMaterial,
    StandardRenderer,
    StandardRenderingPlugin,
} from "../../standard/rendering";
import { drawLine, Gizmo, GizmoAsset, LinesPlugin, registerGizmo } from "./index";

setDefaultTimeout(TIMEOUT.node);
const SIZE = 256;
const ROW = SIZE / 2;
const moving = { body: 0 };
const Mover: System = {
    group: "fixed",
    name: "gizmo-interpolation-mover",
    update(world) {
        const x = world.storage(Transform).translation.x;
        if (moving.body) x.set(moving.body, x.get(moving.body) + 0.25);
    },
};
const subject = gpuScene(
    import.meta.path,
    {
        defaults: false,
        plugins: [
            StandardRenderingPlugin,
            MeshRenderPlugin,
            LinesPlugin,
            { name: "GizmoInterpolationMover", systems: [Mover] },
        ],
    },
    (world) => {
        const camera = world.create();
        world.add(camera, Transform, { translation: [0, 0, 6, 0] });
        world.add(camera, Camera, { antialias: 0 });
        world.add(camera, StandardRenderer);
        world.add(camera, Tonemapping, { method: TonemappingMethod.None });
        attachTexture(world, camera, { width: SIZE, height: SIZE });
        const material = world
            .resource(Materials)
            .add(StandardMaterial({ baseColor: [1, 0, 0, 1], unlit: true }));
        const body = world.create();
        // A thin slab shows no side faces, so its silhouette center stays its center off axis.
        world.add(body, Transform, { translation: [-1, 0, 0, 0], scale: [1, 1, 0.01, 0] });
        world.add(body, MeshInstance);
        world.add(body, MeshMaterial, material);
        const asset = new GizmoAsset();
        drawLine(asset, [0, -0.4, 0.5], [0, 0.4, 0.5], 0x00ff00);
        world.add(body, Gizmo, { handle: registerGizmo(world, asset), width: 2, depthBias: -1 });
        return { camera, body };
    },
);

function center(rgba: Uint8ClampedArray, pick: (r: number, g: number) => boolean): number {
    let min = Infinity;
    let max = -Infinity;
    for (let x = 0; x < SIZE; x++) {
        const at = (ROW * SIZE + x) * 4;
        if (!pick(rgba[at], rgba[at + 1])) continue;
        min = Math.min(min, x);
        max = Math.max(max, x);
    }
    expect(max).toBeGreaterThanOrEqual(min);
    return (min + max) / 2;
}

test("a retained gizmo draws at its mesh's interpolated placement", async () => {
    const { world, scene } = subject();
    world.step(0);
    world.step(0);
    const rest = (await captureTexture(world, scene.camera)).rgba;
    // The line is drawn on the mesh's center; the silhouette is the mesh with the line over it.
    const restOffset = center(rest, (r, g) => g > r) - center(rest, (r, g) => r > 128 || g > 128);
    moving.body = scene.body;
    try {
        const alphas = new Set<number>();
        let travel = 0;
        for (let frame = 0; frame < 6; frame++) {
            world.step(1 / 120);
            alphas.add(world.time.fixedAlpha);
            const rgba = (await captureTexture(world, scene.camera)).rgba;
            const mesh = center(rgba, (r, g) => r > 128 || g > 128);
            const offset = center(rgba, (r, g) => g > r) - mesh - restOffset;
            travel = Math.max(travel, Math.abs(mesh - center(rest, (r, g) => r > 128 || g > 128)));
            // Edge centers quantize to half pixels; one fixed tick moves the body about 10 px.
            expect(Math.abs(offset)).toBeLessThanOrEqual(2);
        }
        expect([...alphas].sort()).toEqual([0, 0.5]);
        expect(travel).toBeGreaterThan(15);
    } finally {
        moving.body = 0;
    }
});
