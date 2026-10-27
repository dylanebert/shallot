import { expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { gpuScene } from "../../../scripts/gpu.fixture";
import { attachTexture, Camera, TransparentPhases } from "../../core/rendering";
import { Transform } from "../../core/transform";
import { probeBuffer } from "../../engine/runtime";
import { drawLine, Gizmo, GizmoAsset, LinesPlugin, registerGizmo } from "../lines";
import { isolationFont } from "./font.fixture";
import { GLYPH_AT, GLYPH_BYTES, GLYPH_FLOATS } from "./glyph";
import { internText, registerFont, Text, TextPlugin, textStateKey } from "./index";

setDefaultTimeout(TIMEOUT.node);
let font = 0;
const subject = gpuScene(
    import.meta.path,
    {
        defaults: false,
        plugins: [TextPlugin, LinesPlugin],
        setup(world) {
            font = registerFont(
                world,
                `data:font/ttf;base64,${Buffer.from(isolationFont()).toString("base64")}`,
            );
        },
    },
    (world) => {
        const camera = world.create();
        world.add(camera, Transform, { translation: [0, 0, 5] });
        world.add(camera, Camera, { antialias: 0 });
        attachTexture(world, camera, { width: 64, height: 64 });
        return camera;
    },
);

test("sorted labels of one font draw one contiguous run; another renderer splits it", async () => {
    const { world, scene: camera } = subject();
    // Author near first: retained layout order is the reverse of transparent order.
    const near = world.create();
    world.add(near, Transform, { translation: [0, 0, 1] });
    world.add(near, Text, { content: internText(world, "ii"), font });
    const far = world.create();
    world.add(far, Transform, { translation: [0, 0, -1] });
    world.add(far, Text, { content: internText(world, "i"), font });
    const queue = world.gpu.device.queue;
    const writeBuffer = queue.writeBuffer.bind(queue);
    const calls: [GPUBuffer, number | undefined][] = [];
    queue.writeBuffer = (buffer, offset, data, dataOffset, size) => {
        calls.push([buffer, size]);
        writeBuffer(buffer, offset, data, dataOffset, size);
    };
    try {
        world.step(0);
    } finally {
        queue.writeBuffer = writeBuffer;
    }
    const instances = world.resource(textStateKey).instances.values().next().value!;
    const orderBuffer = world.gpu.root.unwrap(instances.buffer);
    expect(calls.filter((call) => call[0] === orderBuffer)).toHaveLength(1);
    expect(calls.find((call) => call[0] === orderBuffer)![1]).toBe(3 * 4);
    const phase = world.resource(TransparentPhases).get(camera)!;
    expect(phase.length).toBe(2);
    expect(phase.item(0).stableId).toBe(far);
    expect(phase.item(1).stableId).toBe(near);
    expect(phase.metrics.draws).toBe(1);
    const packed = async () => {
        const order = await probeBuffer(world, orderBuffer, { size: 3 * 4 });
        const indices = new Uint32Array(order.bytes);
        expect(Array.from(indices)).toEqual([2, 0, 1]);
        const snapshot = await probeBuffer(
            world,
            world.gpu.root.unwrap(world.resource(textStateKey).glyphBuf!),
            { size: 3 * GLYPH_BYTES },
        );
        const rows = new Uint32Array(snapshot.bytes);
        expect(Array.from(indices, (i) => rows[i * GLYPH_FLOATS + GLYPH_AT.eid])).toEqual([
            far,
            near,
            near,
        ]);
    };
    await packed();
    const line = world.create();
    world.add(line, Transform);
    const asset = new GizmoAsset();
    drawLine(asset, [-1, 0, 0], [1, 0, 0], 0xffffff);
    world.add(line, Gizmo, { handle: registerGizmo(world, asset) });
    world.step(0);
    expect(phase.length).toBe(3);
    expect([0, 1, 2].map((i) => phase.item(i).stableId)).toEqual([far, line, near]);
    expect(phase.metrics.draws).toBe(3);
    await packed();
});
