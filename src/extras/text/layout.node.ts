import { expect, setDefaultTimeout, spyOn, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { gpuApps } from "../../../scripts/gpu.fixture";
import { Anchor, Billboard } from "../../core/rendering";
import { Transform } from "../../core/transform";
import { probeBuffer } from "../../engine/runtime";
import * as atlas from "./atlas";
import { isolationFont } from "./font.fixture";
import { GLYPH_AT, GLYPH_BYTES, GLYPH_FLOATS } from "./glyph";
import { internText, Justify, registerFont, Text, TextPlugin } from "./index";

setDefaultTimeout(TIMEOUT.node);
let font = 0;
const subjects = gpuApps(import.meta.path, [
    {
        defaults: false,
        plugins: [TextPlugin],
        setup(world) {
            font = registerFont(
                world,
                `data:font/ttf;base64,${Buffer.from(isolationFont()).toString("base64")}`,
            );
        },
    },
]);

test("one changed label among 1,000 lays out once; unchanged and placement-only frames lay out none", async () => {
    const { world } = subjects()[0];
    const labels: number[] = [];
    const content = internText(world, "i");
    for (let i = 0; i < 1000; i++) {
        const eid = world.create();
        world.add(eid, Transform);
        world.add(eid, Text, { content, font });
        labels.push(eid);
    }
    const layout = spyOn(atlas, "layoutText");
    const queue = world.gpu.device.queue;
    const writeBuffer = queue.writeBuffer.bind(queue);
    const writes: [GPUBuffer, number, number | undefined][] = [];
    queue.writeBuffer = (buffer, offset, data, dataOffset, size) => {
        writes.push([buffer, offset, size]);
        writeBuffer(buffer, offset, data, dataOffset, size);
    };
    const buffer = () => world.gpu.buffers.get("textGlyphs")!;
    const glyphWrites = () => writes.filter((call) => call[0] === buffer());
    try {
        world.step(0);
        expect(layout).toHaveBeenCalledTimes(1000);
        layout.mockClear();
        writes.length = 0;
        world.step(0);
        expect(layout).toHaveBeenCalledTimes(0);
        expect(glyphWrites()).toHaveLength(0);
        const changed = labels[500]!;
        world.storage(Text).content.set(changed, internText(world, "ii"));
        world.step(0);
        expect(layout).toHaveBeenCalledTimes(1);
        expect(layout.mock.calls[0]![0]).toBe("ii");
        layout.mockClear();
        const packed = async (counts: number[]) => {
            const size = counts.reduce((sum, count) => sum + count, 0) * GLYPH_BYTES;
            const snapshot = await probeBuffer(world, buffer(), { size });
            const rows = new Uint32Array(snapshot.bytes);
            let offset = 0;
            for (let i = 0; i < counts.length; i++)
                for (let j = 0; j < counts[i]!; j++)
                    expect(rows[offset++ * GLYPH_FLOATS + GLYPH_AT.eid]).toBe(labels[i]!);
        };
        const counts: number[] = labels.map((_, i) => (i === 500 ? 2 : 1));
        await packed(counts);
        // Equal-sized edits write only the changed label's range.
        writes.length = 0;
        world.storage(Text).color.set(changed, 0xffaa55);
        world.step(0);
        expect(layout).toHaveBeenCalledTimes(1);
        expect(glyphWrites()).toHaveLength(1);
        expect(glyphWrites()[0]![1]).toBe(500 * GLYPH_BYTES);
        expect(glyphWrites()[0]![2]).toBe(2 * GLYPH_BYTES);
        layout.mockClear();
        for (const update of [
            () => world.storage(Text).fontSize.set(changed, 2),
            () => world.storage(Text).maxWidth.set(changed, 3),
            () => world.storage(Text).justify.set(changed, Justify.Right),
            () => world.storage(Text).opacity.set(changed, 0.5),
            () => world.storage(Text).font.set(changed, 0),
            () => world.add(changed, Anchor, { offset: Anchor.CENTER }),
            () => world.storage(Anchor).offset.set(changed, ...Anchor.TOP_RIGHT),
            () => world.remove(changed, Anchor),
        ]) {
            update();
            world.step(0);
            expect(layout).toHaveBeenCalledTimes(1);
            layout.mockClear();
            world.step(0);
            expect(layout).toHaveBeenCalledTimes(0);
        }
        world.storage(Transform).translation.set(changed, 1, 2, 3);
        world.storage(Text).billboard.set(changed, Billboard.Screen);
        world.step(0);
        expect(layout).toHaveBeenCalledTimes(0);
        world.remove(changed, Text);
        world.step(0);
        expect(layout).toHaveBeenCalledTimes(0);
        counts[500] = 0;
        await packed(counts);
        world.add(changed, Text, { content, font });
        world.step(0);
        expect(layout).toHaveBeenCalledTimes(1);
        layout.mockClear();
        labels.splice(500, 1);
        labels.push(changed);
        counts.splice(500, 1);
        counts.push(1);
        await packed(counts);
        world.storage(Text).content.set(changed, internText(world, "i".repeat(5000)));
        world.step(0);
        counts[counts.length - 1] = 5000;
        await packed(counts);
        const capacity = buffer().size;
        expect(capacity).toBeGreaterThanOrEqual(5999 * GLYPH_BYTES);
        expect(capacity).toBeLessThan(2 * 5999 * GLYPH_BYTES);
        for (let i = 0; i < 20; i++) {
            world.storage(Text).content.set(changed, internText(world, i % 2 ? "i" : "iiii"));
            world.step(0);
        }
        expect(buffer().size).toBe(capacity);
        world.storage(Text).content.set(changed, internText(world, ""));
        world.step(0);
        layout.mockClear();
        world.step(0);
        expect(layout).toHaveBeenCalledTimes(0);
    } finally {
        layout.mockRestore();
        queue.writeBuffer = writeBuffer;
    }
});
