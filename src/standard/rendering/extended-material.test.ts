import { expect, test } from "bun:test";
import * as rendering from "@dylanebert/shallot/standard/rendering";

const { extendedMaterial, Materials, StandardMaterialInput } = rendering;
const extensionLayout = rendering.extensionLayout;

import * as d from "typegpu/data";

const Extension = d.struct({ steps: d.u32 });

test("extension layout preserves base binding numbers and starts its bindings at 100 in both variants", () => {
    const layout = extensionLayout(Materials, Extension, {
        tint: { type: "uniform", struct: d.struct({ color: d.vec4f }) },
    });
    for (const variant of [layout, layout.depthVariant]) {
        const entries = Object.entries(variant.entries);
        expect(entries.slice(0, 8).map(([name]) => name)).toEqual(
            Object.keys(Materials.layout.entries),
        );
        expect(entries[100]![0]).toBe("tint");
        expect(entries.slice(8, 100).every(([, entry]) => entry === null)).toBe(true);
    }
    expect(layout.parameters.propTypes).toEqual({
        ...StandardMaterialInput.propTypes,
        steps: d.u32,
    });
});

test("extension fields cannot replace base fields", () => {
    expect(() => extensionLayout(Materials, d.struct({ metallic: d.f32 }), {})).toThrow(
        'extension parameter "metallic" collides with a base parameter',
    );
});

test("extension bindings cannot replace base or renderer bindings", () => {
    for (const name of Object.keys(Materials.layout.entries)) {
        expect(() =>
            extensionLayout(Materials, Extension, {
                [name]: { type: "uniform", struct: Extension },
            }),
        ).toThrow(`extension binding "${name}" collides with a base or reserved binding`);
    }
});

test("extension layout checks merged WebGPU stage budgets", () => {
    const bindings = (count: number, binding: object) =>
        Object.fromEntries(Array.from({ length: count }, (_, i) => [`extra${i}`, binding]));
    for (const type of ["storage", "attribute"] as const) {
        expect(() =>
            extensionLayout(Materials, Extension, bindings(2, { type, element: d.vec4f }) as never),
        ).toThrow("material layout uses 9 vertex storage buffers; WebGPU's limit is 8");
    }
    expect(() =>
        extensionLayout(Materials, Extension, bindings(10, { type: "texture" }) as never),
    ).toThrow("material layout uses 17 fragment sampled textures; WebGPU's limit is 16");
    expect(() =>
        extensionLayout(
            Materials,
            Extension,
            bindings(5, {
                type: "storage-texture",
                format: "rgba8unorm",
                access: "write",
            }) as never,
        ),
    ).toThrow("storage textures");
    expect(() =>
        extensionLayout(Materials, Extension, {
            extra: {
                type: "storage-texture",
                format: "rgba8unorm",
                access: "write",
                visibility: ["vertex"],
            },
        } as never),
    ).toThrow("fragment stage only");
});

test("extended materials refuse empty names and prebuilt vertex functions", () => {
    expect(() => extendedMaterial(Materials, { name: "", fragment: () => d.vec4f(1) })).toThrow(
        "material type name must not be empty",
    );
    expect(() =>
        extendedMaterial(Materials, {
            name: "PrebuiltVertex",
            vertex: Materials.fragment as never,
            fragment: () => d.vec4f(1),
        }),
    ).toThrow('vertex shader must be a plain "use gpu" function');
});

test("extended materials refuse prebuilt TypeGPU fragments", () => {
    expect(() =>
        extendedMaterial(Materials, {
            name: "PrebuiltExtension",
            // @ts-expect-error A prebuilt TypeGPU function is not a plain shader.
            fragment: Materials.fragment,
        }),
    ).toThrow(
        'material type "PrebuiltExtension" fragment shader must be a plain "use gpu" function, not a prebuilt TypeGPU function',
    );
});
