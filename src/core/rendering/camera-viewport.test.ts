import { expect, test } from "bun:test";
import { World } from "../../engine";
import { CameraViewport } from "./camera";
import { cameraRect } from "./camera-viewport";

for (const [width, height, expected] of [
    [48, 20, { x: 40, y: 4, width: 8, height: 16 }],
    [24, 2, { x: 23, y: 1, width: 1, height: 1 }],
    [0, 20, { x: 0, y: 4, width: 0, height: 16 }],
    [48, 0, { x: 40, y: 0, width: 8, height: 0 }],
] as const) {
    test(`camera rectangle clamps to ${width} × ${height} as Bevy Viewport::clamp_to_size`, () => {
        const world = new World();
        world.registry.register(CameraViewport);
        const camera = world.create();
        world.add(camera, CameraViewport, { x: 40, y: 4, width: 16, height: 32 });
        const rect = { x: 0, y: 0, width: 0, height: 0 };
        try {
            cameraRect(world, camera, width, height, rect);
            expect(rect).toEqual(expected);
        } finally {
            world.dispose();
        }
    });
}
