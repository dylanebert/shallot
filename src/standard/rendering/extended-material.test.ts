import { expect, test } from "bun:test";
import { extendedMaterial, Materials } from "@dylanebert/shallot/standard/rendering";

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
