import { expect, test } from "bun:test";
import * as d from "typegpu/data";
import { StandardMaterial, StandardMaterialInput } from "./material-data";

test("five identity texture transforms occupy 100 scalar bytes, reusing the old row's tail padding", () => {
    expect(d.sizeOf(StandardMaterialInput)).toBe(208);
    const row = StandardMaterial();
    for (const slot of [
        "baseColor",
        "metallicRoughness",
        "normalMap",
        "occlusion",
        "emissive",
    ] as const)
        expect(row[`${slot}Transform`]).toEqual({
            offsetX: 0,
            offsetY: 0,
            scaleX: 1,
            scaleY: 1,
            rotation: 0,
        });
});
