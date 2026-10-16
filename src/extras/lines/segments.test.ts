import { expect, test } from "bun:test";
import { World } from "../../engine";
import {
    drawArrow,
    drawLine,
    drawWireBox,
    GizmoAsset,
    initializeSegmentState,
    lineStableIds,
    lineStableParts,
    push,
    segmentState,
} from "./segments";

test("line stable identities survive segment staging growth", () => {
    const world = new World();
    try {
        initializeSegmentState(world);
        const count = 17_000;
        for (let part = 0; part < count; part++)
            push(segmentState(world), 0, 0, 0, 1, 0, 0, 1, 0xffffffff, 123, part);
        const stableIds = lineStableIds(world);
        const stableParts = lineStableParts(world);
        expect(stableIds.length).toBeGreaterThan(count);
        expect(stableIds[0]).toBe(123);
        expect(stableIds[16_383]).toBe(123);
        expect(stableIds[count - 1]).toBe(123);
        expect(stableParts[0]).toBe(0);
        expect(stableParts[16_383]).toBe(16_383);
        expect(stableParts[count - 1]).toBe(count - 1);
    } finally {
        world.dispose();
    }
});

test("an asset repacks once per geometry change, with no authored width", () => {
    const asset = new GizmoAsset();
    drawLine(asset, [0, 0, 0], [1, 0, 0], 0xffffff);
    expect(asset.rebuilds).toBe(0);
    const first = asset.geometry();
    expect(first.f32[3]).toBe(0);
    expect(asset.rebuilds).toBe(1);
    expect(asset.geometry()).toBe(first);
    expect(asset.rebuilds).toBe(1);
    drawWireBox(asset, [0, 0, 0], [1, 1, 1], 0xffffff);
    expect(asset.geometry().count).toBe(13);
    expect(asset.rebuilds).toBe(2);
});

function assetWidthTypeControls(world: World, asset: GizmoAsset): void {
    drawLine(world, [0, 0, 0], [1, 0, 0], 0xffffff, 3);
    // @ts-expect-error Retained width belongs to Gizmo, not its asset.
    drawLine(asset, [0, 0, 0], [1, 0, 0], 0xffffff, 3);
    // @ts-expect-error Retained width belongs to Gizmo, not its asset.
    drawWireBox(asset, [0, 0, 0], [1, 1, 1], 0xffffff, 3);
    drawArrow(asset, [0, 0, 0], [1, 0, 0], 0xffffff, undefined, 0.3, [0, 1, 0]);
    // @ts-expect-error Retained arrows cannot author a width, even without a tipLength.
    drawArrow(asset, [0, 0, 0], [1, 0, 0], 0xffffff, 3);
    // @ts-expect-error Retained arrows skip width before tipLength and normal.
    drawArrow(asset, [0, 0, 0], [1, 0, 0], 0xffffff, 3, 0.3, [0, 1, 0]);
}
void assetWidthTypeControls;
