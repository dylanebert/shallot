import { expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import * as d from "typegpu/data";
import { gpuApps } from "../../../scripts/gpu.fixture";
import { Meshes, MeshInstance } from "../../core/mesh";
import {
    attachTexture,
    Camera,
    captureTexture,
    Tonemapping,
    TonemappingMethod,
} from "../../core/rendering";
import { Transform } from "../../core/transform";
import { DEFAULT_PLUGINS } from "../../standard";
import { MeshMaterial, materialLayout, materialType, StandardRenderer } from "./index";

setDefaultTimeout(TIMEOUT.node);
const subjects = gpuApps(import.meta.path, [
    {
        defaults: false,
        plugins: [
            ...DEFAULT_PLUGINS,
            {
                name: "OwnedMaterial",
                initialize(world) {
                    world.resource(material);
                },
            },
        ],
    },
]);
const Color = d.struct({ value: d.vec3f });
const OwnedColor = {
    create: (world: import("../../engine").World) => ({
        color: world.gpu.root.createBuffer(Color, { value: d.vec3f(1, 0, 0) }).$usage("uniform"),
    }),
};
const layout = materialLayout(d.struct({ unused: d.f32 }), {
    color: { type: "uniform", struct: Color },
});
const material = materialType({
    name: "OwnedColor",
    layout,
    defaults: { unused: 0 },
    bindings: (world) => world.resource(OwnedColor),
    fragment: () => {
        "use gpu";
        return d.vec4f(layout.$.color.value, 1);
    },
});

test("material owned uniforms resolve from live bindings and rebind after replacement", async () => {
    const { world } = subjects()[0];
    const mesh = world.resource(Meshes).add({
        name: "owned-color-quad",
        vertices: new Float32Array([
            -2, -2, 0, 0, 0, 0, 1, 0, 2, -2, 0, 1, 0, 0, 1, 0, 2, 2, 0, 1, 0, 0, 1, 1, -2, 2, 0, 0,
            0, 0, 1, 1,
        ]),
        indices: new Uint32Array([0, 1, 2, 0, 2, 3]),
    });
    const eid = world.create();
    world.add(eid, Transform);
    world.add(eid, MeshInstance, { mesh });
    world.add(eid, MeshMaterial, world.resource(material).add({ unused: 0 }));
    const camera = world.create();
    world.add(camera, Transform, { translation: [0, 0, 5, 0] });
    world.add(camera, Camera);
    world.storage(Camera).antialias.set(camera, 0);
    world.add(camera, StandardRenderer);
    world.add(camera, Tonemapping, { method: TonemappingMethod.None });
    attachTexture(world, camera, { width: 8, height: 8 });
    world.step(0);
    world.step(0);
    const first = (await captureTexture(world, camera)).rgba;
    const center = (4 * 8 + 4) * 4;
    expect(Array.from(first.subarray(center, center + 4))).toEqual([255, 0, 0, 255]);
    world.resource(OwnedColor).color = world.gpu.root
        .createBuffer(Color, { value: d.vec3f(0, 1, 0) })
        .$usage("uniform");
    world.step(0);
    const replaced = (await captureTexture(world, camera)).rgba;
    expect(Array.from(replaced.subarray(center, center + 4))).toEqual([0, 255, 0, 255]);
});
