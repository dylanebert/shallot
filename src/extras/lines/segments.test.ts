import { expect, test } from "bun:test";
import { World } from "../../engine";
import { SEGMENT_FLOATS } from "./pipeline";
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

test("an asset appends each drawn segment with no authored width", () => {
    const asset = new GizmoAsset();
    drawLine(asset, [0, 0, 0], [1, 0, 0], 0xffffff);
    expect(asset.segments.count).toBe(1);
    expect(asset.segments.f32[3]).toBe(0);
    drawWireBox(asset, [0, 0, 0], [1, 1, 1], 0xffffff);
    expect(asset.segments.count).toBe(13);
    for (let i = 0; i < 13; i++) expect(asset.segments.f32[i * SEGMENT_FLOATS + 3]).toBe(0);
});

test("immediate line, box and arrow options stage depth bias on every primitive", () => {
    const world = new World();
    try {
        const options = { width: 3, depthBias: -0.25 };
        drawLine(world, [0, 0, 0], [1, 0, 0], 0xffffff, options);
        drawWireBox(world, [0, 0, 0], [1, 1, 1], 0xffffff, options);
        drawArrow(world, [0, 0, 0], [1, 0, 0], 0xffffff, options);
        const state = segmentState(world);
        expect(state.count).toBe(15);
        for (let i = 0; i < state.count; i++) {
            expect(state.f32[i * SEGMENT_FLOATS + 3]).toBe(3);
            expect(state.f32[i * SEGMENT_FLOATS + 12]).toBe(-0.25);
        }
        drawLine(world, [0, 0, 0], [1, 0, 0], 0xffffff);
        expect(state.f32[15 * SEGMENT_FLOATS + 12]).toBe(0);
    } finally {
        world.dispose();
    }
});

function assetWidthTypeControls(world: World, asset: GizmoAsset): void {
    drawLine(world, [0, 0, 0], [1, 0, 0], 0xffffff, { width: 3 });
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
