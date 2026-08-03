import { expect, test } from "bun:test";
import type { StorageFlag, TgpuBuffer } from "typegpu";
import * as d from "typegpu/data";
import { surfaceLayout } from "../render/core";
import { validateMeshBindingOverrides } from "./forward";

const layout = surfaceLayout({ values: { type: "storage", element: d.vec4f } });

function typed<T extends d.AnyData>(
    dataType: T,
    storage: boolean,
): TgpuBuffer<T> | (TgpuBuffer<T> & StorageFlag) {
    return {
        resourceType: "buffer",
        dataType,
        usableAsStorage: storage,
    } as TgpuBuffer<T> | (TgpuBuffer<T> & StorageFlag);
}

test("mesh overrides reject a wrong typed schema before bind-group creation", () => {
    const wrong = typed(d.arrayOf(d.u32, 4), true);
    expect(() => validateMeshBindingOverrides(layout, { values: wrong })).toThrow(/values.*schema/);
});

test("mesh overrides reject a typed buffer missing the layout usage", () => {
    const missing = typed(d.arrayOf(d.vec4f, 4), false);
    expect(() => validateMeshBindingOverrides(layout, { values: missing })).toThrow(
        /values.*storage usage/,
    );
});

test("mesh overrides ignore typed bindings owned by another surface", () => {
    const values = typed(d.arrayOf(d.vec4f, 4), true);
    const otherSurface = typed(d.arrayOf(d.u32, 4), true);
    expect(() => validateMeshBindingOverrides(layout, { values, otherSurface })).not.toThrow();
});
