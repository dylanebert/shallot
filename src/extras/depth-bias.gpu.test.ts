import { expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import tgpu from "typegpu";
import * as d from "typegpu/data";
import { gpuApps } from "../../scripts/gpu.fixture";
import {
    Anchor,
    attachTexture,
    BeginFrameSystem,
    Camera,
    CameraMode,
    captureTexture,
    DEPTH_FORMAT,
    RenderContext,
    RenderPhases,
} from "../core/rendering";
import { Transform } from "../core/transform";
import type { Plugin } from "../engine";
import { drawLine, Gizmo, GizmoAsset, LinesPlugin, registerGizmo } from "./lines";
import { internText, Text, TextPlugin } from "./text";

setDefaultTimeout(TIMEOUT.gpu);
const a = [-1.5, 0, -3];
const b = [1.5, 0, -3];
// Past the far plane at z = -6, which the line from a crosses a third of the way along.
const beyondFar = [1.5, 0, -12];
// Just inside the far plane: a depth exactly on it rounds to either side.
const insideFar = [-0.5 - 0.0001 / 3, 0, -5.9999];
const immediate = { create: () => ({ enabled: false, depthBias: 0, width: 6, end: b }) };
const Surface: Plugin = {
    name: "OpaqueDepthSurface",
    dependencies: [LinesPlugin, TextPlugin],
    systems: [
        {
            name: "depth-line",
            group: "draw",
            after: [BeginFrameSystem],
            before: LinesPlugin.systems,
            update(world) {
                const options = world.resource(immediate);
                if (options.enabled) drawLine(world, a, options.end, 0xffffff, options);
            },
        },
    ],
    warm(world) {
        const pipeline = world.gpu.root.createRenderPipeline({
            vertex: tgpu.vertexFn({
                in: { index: d.builtin.vertexIndex },
                out: { position: d.builtin.position },
            })(({ index }) => {
                "use gpu";
                const corners = d.arrayOf(
                    d.vec2f,
                    3,
                )([d.vec2f(-1, -1), d.vec2f(3, -1), d.vec2f(-1, 3)]);
                return { position: d.vec4f(corners[index], 0.5, 1) };
            }),
            fragment: tgpu.fragmentFn({ out: d.vec4f })(() => {
                "use gpu";
                return d.vec4f(0, 0, 0, 1);
            }),
            targets: { format: world.resource(RenderContext).format },
            depthStencil: {
                format: DEPTH_FORMAT,
                depthCompare: "greater-equal",
                depthWriteEnabled: true,
            },
        });
        world.resource(RenderPhases).push({
            opaque(_world, _eid, _view, pass) {
                pass.setPipeline(world.gpu.root.unwrap(pipeline));
                pass.draw(3);
            },
        });
    },
};
const subjects = gpuApps(import.meta.path, [{ defaults: false, plugins: [Surface] }]);

for (const kind of ["immediate line", "retained line", "text"] as const) {
    test(`${kind} is hidden behind an opaque surface by default; depthBias moves its visibility cutoff`, async () => {
        const { world } = subjects()[0];
        const camera = world.create();
        world.add(camera, Transform, { translation: [0, 0, 5] });
        world.add(camera, Camera, {
            mode: CameraMode.Orthographic,
            size: 2,
            near: 1,
            far: 11,
            antialias: 0,
        });
        attachTexture(world, camera, { width: 64, height: 64 });
        const frame = async () => {
            world.step(1 / 60);
            return (await captureTexture(world, camera)).rgba;
        };
        const baseline = await frame();
        const entity = world.create();
        world.add(entity, Transform, { translation: [-1.5, -0.25, -3] });
        if (kind === "text") {
            world.add(entity, Anchor, { offset: Anchor.BOTTOM_LEFT });
            world.add(entity, Text, { content: internText(world, "depth"), fontSize: 0.8 });
        } else if (kind === "retained line") {
            const asset = new GizmoAsset();
            drawLine(asset, [0, 0.25, 0], [3, 0.25, 0], 0xffffff);
            world.add(entity, Gizmo, { handle: registerGizmo(world, asset), width: 6 });
        } else world.resource(immediate).enabled = true;
        const bias = (value: number) => {
            if (kind === "text") world.storage(Text).depthBias.set(entity, value);
            else if (kind === "retained line") world.storage(Gizmo).depthBias.set(entity, value);
            else world.resource(immediate).depthBias = value;
        };
        expect(await frame()).toEqual(baseline);
        bias(-0.25);
        expect(await frame()).toEqual(baseline);
        bias(-0.4);
        expect(await frame()).toEqual(baseline);
        bias(-0.5);
        const biased = await frame();
        expect(biased).not.toEqual(baseline);
        bias(-1);
        expect(await frame()).toEqual(biased);
        bias(-2);
        expect(await frame()).toEqual(biased);
        bias(0);
        expect(await frame()).toEqual(baseline);
        world.resource(immediate).enabled = false;
        world.resource(immediate).depthBias = 0;
        world.destroy(entity);
        world.destroy(camera);
    });
}

function largestDifference(a: ArrayLike<number>, b: ArrayLike<number>): number {
    let largest = 0;
    for (let i = 0; i < a.length; i++) largest = Math.max(largest, Math.abs(a[i]! - b[i]!));
    return largest;
}

for (const kind of ["immediate line", "retained line"] as const) {
    test(`${kind} crossing an orthographic far plane keeps its in-range part under negative depthBias`, async () => {
        const { world } = subjects()[0];
        const camera = world.create();
        world.add(camera, Transform, { translation: [0, 0, 5] });
        world.add(camera, Camera, {
            mode: CameraMode.Orthographic,
            size: 2,
            near: 1,
            far: 11,
            antialias: 0,
        });
        attachTexture(world, camera, { width: 64, height: 64 });
        const frame = async () => {
            world.step(1 / 60);
            return (await captureTexture(world, camera)).rgba;
        };
        const baseline = await frame();
        const options = world.resource(immediate);
        const entity = world.create();
        world.add(entity, Transform, { translation: [a[0]!, a[1]!, a[2]!] });
        const draw = (end: number[], depthBias: number) => {
            if (kind === "immediate line") {
                options.enabled = true;
                options.end = end;
                options.depthBias = depthBias;
                return;
            }
            const asset = new GizmoAsset();
            drawLine(asset, [0, 0, 0], [end[0]! - a[0]!, 0, end[2]! - a[2]!], 0xffffff);
            world.remove(entity, Gizmo);
            world.add(entity, Gizmo, { handle: registerGizmo(world, asset), width: 6, depthBias });
        };
        for (const depthBias of [-0.5, -1]) {
            draw(insideFar, depthBias);
            const inRange = await frame();
            expect(inRange).not.toEqual(baseline);
            draw(beyondFar, depthBias);
            // The clipped end may move a covered edge pixel by a unit of coverage.
            expect(largestDifference(await frame(), inRange)).toBeLessThanOrEqual(2);
        }
        draw(beyondFar, 0);
        expect(await frame()).toEqual(baseline);
        options.enabled = false;
        options.end = b;
        options.depthBias = 0;
        world.destroy(entity);
        world.destroy(camera);
    });
}
