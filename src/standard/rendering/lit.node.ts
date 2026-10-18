import { expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import * as d from "typegpu/data";
import { gpuApps } from "../../../scripts/gpu.fixture";
import { MeshInstance } from "../../core/mesh";
import {
    AmbientLight,
    attachTexture,
    Camera,
    captureTexture,
    PointLight,
    Tonemapping,
    TonemappingMethod,
} from "../../core/rendering";
import { Transform } from "../../core/transform";
import {
    lit,
    MaterialPlugin,
    Materials,
    MeshMaterial,
    MeshRenderPlugin,
    materialType,
    StandardMaterial,
    StandardRenderer,
    StandardRenderingPlugin,
} from "./index";

setDefaultTimeout(TIMEOUT.node);

const LitMaterial = materialType({
    name: "LitWhite",
    parameters: d.struct({ unused: d.f32 }),
    defaults: { unused: 0 },
    fragment: (ctx) => {
        "use gpu";
        return d.vec4f(lit(d.vec3f(1), ctx.worldNormal), 1);
    },
});
const subjects = gpuApps(import.meta.path, [
    {
        defaults: false,
        plugins: [StandardRenderingPlugin, MeshRenderPlugin, MaterialPlugin(LitMaterial)],
    },
]);

test("lit equals StandardMaterial at metallic 0 within a centimetre of a point light", async () => {
    const { world } = subjects()[0];
    const camera = world.create();
    world.add(camera, Transform, { translation: [0, 0, 5, 0] });
    world.add(camera, Camera);
    world.add(camera, AmbientLight, { brightness: 0 });
    world.add(camera, StandardRenderer);
    world.add(camera, Tonemapping, { method: TonemappingMethod.None });
    // An odd size puts the centre pixel's ray on the axis, through the face point under the light.
    attachTexture(world, camera, { width: 63, height: 31 });
    // The box's front face is at z = 0.25, so the light is 5 mm in front of the centre pixel's fragment.
    const light = world.create();
    world.add(light, Transform, { translation: [0, 0, 0.255, 0] });
    world.add(light, PointLight, { intensity: 0.3 });
    const box = world.create();
    world.add(box, Transform, { scale: [0.5, 0.5, 0.5, 0] });
    world.add(box, MeshInstance);
    world.add(box, MeshMaterial, world.resource(LitMaterial).add({ unused: 0 }));
    const centre = async () => {
        world.step(0);
        world.step(0);
        const { rgba } = await captureTexture(world, camera);
        return Array.from(rgba.subarray((15 * 63 + 31) * 4, (15 * 63 + 31) * 4 + 4));
    };
    const litPixel = await centre();
    const standard = world
        .resource(Materials)
        .add(StandardMaterial({ baseColor: [1, 1, 1, 1], perceptualRoughness: 1 }));
    world.storage(MeshMaterial).type.set(box, standard.type);
    world.storage(MeshMaterial).material.set(box, standard.material);
    const standardPixel = await centre();
    expect(standardPixel[0]).toBeGreaterThan(64);
    expect(standardPixel[0]).toBeLessThan(250);
    expect(litPixel).toEqual(standardPixel);
});
