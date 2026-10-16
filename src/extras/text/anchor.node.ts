import { expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { readFromArrayBuffer } from "typegpu";
import { gpuApps } from "../../../scripts/gpu.fixture";
import { Anchor } from "../../core/rendering";
import { Transform } from "../../core/transform";
import { probeBuffer } from "../../engine/runtime";
import { Sprite, SpriteMaterialInput, SpriteMaterialType, SpritePlugin } from "../sprite";
import { spriteOffset } from "../sprite/material";
import { isolationFont } from "./font.fixture";
import { GLYPH_AT, GLYPH_BYTES } from "./glyph";
import { internText, registerFont, Text, TextPlugin } from "./index";

setDefaultTimeout(TIMEOUT.node);
let font = 0;
const subjects = gpuApps(import.meta.path, [
    {
        defaults: false,
        plugins: [SpritePlugin, TextPlugin],
        setup(world) {
            font = registerFont(
                world,
                `data:font/ttf;base64,${Buffer.from(isolationFont()).toString("base64")}`,
            );
        },
    },
]);

const anchors = [
    Anchor.BOTTOM_LEFT,
    Anchor.BOTTOM_CENTER,
    Anchor.BOTTOM_RIGHT,
    Anchor.CENTER_LEFT,
    Anchor.CENTER,
    Anchor.CENTER_RIGHT,
    Anchor.TOP_LEFT,
    Anchor.TOP_CENTER,
    Anchor.TOP_RIGHT,
];

test("sprite and label pivots follow all nine named anchors, updates, removal and extra defaults", async () => {
    const { world } = subjects()[0];
    const sprite = world.create();
    world.add(sprite, Transform);
    world.add(sprite, Sprite, { size: [2, 4] });
    const label = world.create();
    world.add(label, Transform);
    world.add(label, Text, { content: internText(world, "i"), font });
    const glyphPosition = async () => {
        const snapshot = await probeBuffer(world, world.gpu.buffers.get("textGlyphs")!, {
            size: GLYPH_BYTES,
        });
        return new Float32Array(snapshot.bytes).slice(GLYPH_AT.pos, GLYPH_AT.pos + 2);
    };
    const spritePosition = () => {
        const table = world.resource(SpriteMaterialType.source!).table;
        const stride = table.rowBytes;
        const row = table.rowIndex(sprite);
        const parameters = readFromArrayBuffer(
            table.bytes.slice(row * stride, (row + 1) * stride).buffer,
            SpriteMaterialInput,
        );
        const offset = spriteOffset(parameters);
        return [offset.x, offset.y];
    };
    world.step(0);
    expect([...spritePosition()]).toEqual([-1, -2]);
    const initial = await glyphPosition();
    // The fixture advances 600/1000 em; its 800/1000-em rectangle has 0.1-em SDF padding on each edge.
    expect(initial[0]).toBeCloseTo(-0.1);
    expect(initial[1]).toBeCloseTo(-0.1);
    world.add(sprite, Anchor);
    world.add(label, Anchor);
    for (let i = 0; i < anchors.length; i++) {
        world.storage(Anchor).offset.set(sprite, anchors[i]![0], anchors[i]![1]);
        world.storage(Anchor).offset.set(label, anchors[i]![0], anchors[i]![1]);
        world.step(0);
        const column = i % 3;
        const row = Math.floor(i / 3);
        expect([...spritePosition()]).toEqual([-column, -2 * row]);
        const glyph = await glyphPosition();
        expect(glyph[0]).toBeCloseTo(-0.1 - column * 0.3);
        expect(glyph[1]).toBeCloseTo(-0.1 - row * 0.5);
    }
    world.storage(Anchor).offset.set(sprite, ...Anchor.CENTER);
    world.storage(Anchor).offset.set(label, ...Anchor.CENTER);
    world.step(0);
    expect([...spritePosition()]).toEqual([-1, -2]);
    expect((await glyphPosition())[0]).toBeCloseTo(-0.4);
    world.remove(sprite, Anchor);
    world.remove(label, Anchor);
    world.step(0);
    expect([...spritePosition()]).toEqual([-1, -2]);
    expect(await glyphPosition()).toEqual(initial);
});
