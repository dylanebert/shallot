import { expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { gpuScene } from "../../../scripts/gpu.fixture";
import { attachTexture, Camera, RenderLayers, TransparentPhases } from "../../core/rendering";
import { Transform } from "../../core/transform";
import {
    drawArrow,
    drawLine,
    drawWireBox,
    Gizmo,
    GizmoAsset,
    LinesPlugin,
    registerGizmo,
} from "./index";

setDefaultTimeout(TIMEOUT.node);
const subject = gpuScene(import.meta.path, { defaults: false, plugins: [LinesPlugin] }, (world) => {
    const camera = world.create();
    world.add(camera, Transform, { translation: [0, 0, 5] });
    world.add(camera, Camera, { antialias: 0 });
    attachTexture(world, camera, { width: 32, height: 32 });
    return camera;
});

for (const kind of ["entity", "line", "box", "arrow"] as const)
    for (const [itemMask, cameraMask] of [
        [2, 1],
        [1, 2],
    ])
        test(`${kind} mask ${itemMask} is not drawn by camera mask ${cameraMask}`, () => {
            const { world, scene: camera } = subject();
            if (cameraMask !== 1) world.add(camera, RenderLayers, { mask: cameraMask });
            const entity = world.create();
            if (kind === "entity") {
                const asset = new GizmoAsset();
                drawLine(asset, [0, 0, 0], [1, 0, 0], 0xffffff);
                world.add(entity, Transform);
                world.add(entity, Gizmo, { handle: registerGizmo(world, asset) });
                if (itemMask !== 1) world.add(entity, RenderLayers, { mask: itemMask });
            }
            const frame = () => {
                const options = itemMask === 1 ? undefined : { renderLayers: itemMask };
                if (kind === "line") drawLine(world, [0, 0, 0], [1, 0, 0], 0xffffff, options);
                if (kind === "box") drawWireBox(world, [0, 0, 0], [1, 1, 1], 0xffffff, options);
                if (kind === "arrow") drawArrow(world, [0, 0, 0], [1, 0, 0], 0xffffff, options);
                world.step(1 / 60);
                return world.resource(TransparentPhases).get(camera)!.metrics.draws;
            };
            try {
                expect(frame()).toBe(0);
                if (!world.has(camera, RenderLayers)) world.add(camera, RenderLayers);
                world.storage(RenderLayers).mask.set(camera, 3);
                expect(frame()).toBe(kind === "box" ? 12 : kind === "arrow" ? 2 : 1);
            } finally {
                world.destroy(entity);
                world.remove(camera, RenderLayers);
            }
        });
