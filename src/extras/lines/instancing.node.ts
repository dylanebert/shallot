import { expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { gpuScene } from "../../../scripts/gpu.fixture";
import { attachTexture, Camera, TransparentPhases } from "../../core/rendering";
import { Transform } from "../../core/transform";
import { drawLine, Gizmo, GizmoAsset, LinesPlugin, registerGizmo } from "./index";

setDefaultTimeout(TIMEOUT.node);
const subject = gpuScene(import.meta.path, { defaults: false, plugins: [LinesPlugin] }, (world) => {
    const cameras = [];
    for (let i = 0; i < 2; i++) {
        const camera = world.create();
        world.add(camera, Transform, { translation: [0, 0, 5] });
        world.add(camera, Camera, { antialias: 0 });
        attachTexture(world, camera, { width: 32, height: 32 });
        cameras.push(camera);
    }
    return cameras;
});

test("many gizmos draw once per asset per view regardless of depth interleaving", () => {
    const { world, scene: cameras } = subject();
    const handles = [];
    for (const color of [0xff3030, 0x30ff30]) {
        const asset = new GizmoAsset();
        drawLine(asset, [-1, 0, 0], [1, 0, 0], color);
        handles.push(registerGizmo(world, asset));
    }
    for (let i = 0; i < 100; i++) {
        const eid = world.create();
        world.add(eid, Transform, { translation: [0, 0, -i / 100] });
        world.add(eid, Gizmo, { handle: handles[0] });
    }
    world.step(1 / 60);
    for (const camera of cameras) {
        const phase = world.resource(TransparentPhases).get(camera)!;
        expect(phase.metrics.draws).toBe(1);
    }
    const eid = world.create();
    world.add(eid, Transform, { translation: [0, 0, -0.5] });
    world.add(eid, Gizmo, { handle: handles[1] });
    world.step(1 / 60);
    for (const camera of cameras) {
        const phase = world.resource(TransparentPhases).get(camera)!;
        expect(phase.metrics.draws).toBe(2);
        expect(phase.metrics.items).toBe(2);
    }
});
