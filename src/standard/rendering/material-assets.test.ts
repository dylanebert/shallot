import { expect, test } from "bun:test";
import { createApp } from "../../engine/app";
import { AlphaMode, Materials } from "./material";
import { StandardMaterial } from "./material-data";

test("standard material rows are authored without a renderer or device", async () => {
    const app = await createApp({ defaults: false, plugins: [] });
    try {
        const materials = app.world.resource(Materials);
        const handle = materials.add(
            StandardMaterial({ baseColor: [0.2, 0.4, 0.6, 1], perceptualRoughness: 0.3 }),
        );
        materials.update(handle, { metallic: 0.7 });
        expect(handle).toEqual({ type: 0, material: 1, alphaMode: 0, alphaCutoff: 0.5 });
        const glass = materials.add(StandardMaterial({ baseColor: [1, 1, 1, 0.4] }), {
            alphaMode: AlphaMode.Blend,
        });
        materials.update(glass, {}, { alphaMode: AlphaMode.Mask(0.3) });
        expect(glass).toEqual({ type: 0, material: 2, alphaMode: 1, alphaCutoff: 0.3 });
    } finally {
        app.dispose();
    }
});
