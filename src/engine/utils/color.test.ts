import { expect, test } from "bun:test";
import { unpackColor, unpackColorInto } from "./color";

test("unpackColorInto writes the same linear channels as unpackColor at its offset", () => {
    for (const packed of [0x000000, 0x804020, 0xffffff]) {
        const color = unpackColor(packed);
        const out = new Float32Array(5).fill(-1);
        unpackColorInto(packed, out, 1);

        expect([...out]).toEqual([
            -1,
            Math.fround(color.r),
            Math.fround(color.g),
            Math.fround(color.b),
            -1,
        ]);
        expect(unpackColor(packed)).toEqual(color);
    }
});
