import { expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { gpuApps } from "../../../scripts/gpu.fixture";
import { Transform } from "../transform";
import { attachTexture, Camera, CorePipelinePlugin, RenderPhases } from ".";

setDefaultTimeout(TIMEOUT.gpu);
const subjects = gpuApps(import.meta.path, [{ defaults: false, plugins: [CorePipelinePlugin] }]);

test("every opaque record precedes every afterOpaque record and the first transparent draw in the same core-owned pass", () => {
    const { world } = subjects()[0]!;
    const camera = world.create();
    world.add(camera, Transform);
    world.add(camera, Camera, { antialias: 0 });
    attachTexture(world, camera, { width: 8, height: 8 });
    const order: string[] = [];
    let mainPass!: GPURenderPassEncoder;
    for (let i = 0; i < 2; i++)
        world.resource(RenderPhases).push({
            opaque(_world, _eid, _view, pass) {
                mainPass ??= pass;
                expect(pass).toBe(mainPass);
                order.push(`opaque:${i}`);
            },
            afterOpaque(_world, _eid, _view, pass) {
                expect(pass).toBe(mainPass);
                order.push(`afterOpaque:${i}`);
            },
            transparent(_world, _eid, _view, phase) {
                phase.add(i, 0, 0, 0, -i - 1, i);
            },
            drawTransparent(_world, _eid, _view, _phase, _start, _end, pass) {
                expect(pass).toBe(mainPass);
                order.push(`transparent:${i}`);
            },
        });
    world.step(0);
    expect(order).toEqual([
        "opaque:0",
        "opaque:1",
        "afterOpaque:0",
        "afterOpaque:1",
        "transparent:1",
        "transparent:0",
    ]);
});
