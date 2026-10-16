import { expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { gpuApps } from "../../../scripts/gpu.fixture";
import { Anchor, attachTexture, Camera, captureTexture } from "../../core/rendering";
import { Transform } from "../../core/transform";
import { isolationFont } from "./font.fixture";
import { internText, Justify, registerFont, Text, TextPlugin } from "./index";

setDefaultTimeout(TIMEOUT.gpu);
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

test("a wrapped centred label covers two separated rows in a captured frame", async () => {
    const { world } = subjects()[0];
    const camera = world.create();
    world.add(camera, Transform, { translation: [0, 0, 5] });
    world.add(camera, Camera);
    attachTexture(world, camera, { width: 64, height: 64 });
    world.step(1 / 60);
    world.step(1 / 60);
    const baseline = (await captureTexture(world, camera)).rgba;
    const label = world.create();
    world.add(label, Transform);
    world.add(label, Anchor, { offset: Anchor.CENTER });
    world.add(label, Text, {
        content: internText(world, "ii i"),
        font,
        fontSize: 1,
        maxWidth: 1.3,
        justify: Justify.Center,
        color: 0xffffff,
    });
    world.step(1 / 60);
    world.step(1 / 60);
    const { rgba } = await captureTexture(world, camera);
    const rows: number[][] = [];
    for (let y = 0; y < 64; y++) {
        const xs: number[] = [];
        for (let x = 0; x < 64; x++)
            if (rgba[(y * 64 + x) * 4]! > baseline[(y * 64 + x) * 4]! + 30) xs.push(x);
        if (xs.length) rows.push([y, xs[0]!, xs.at(-1)!]);
    }
    const bands = rows.filter((row, i) => i === 0 || row[0]! > rows[i - 1]![0]! + 1);
    expect(bands).toHaveLength(2);
    for (const row of rows) expect((row[1]! + row[2]!) / 2).toBeCloseTo(31.5, 0);
});
