import { expect, setDefaultTimeout, test } from "bun:test";
import { gpuApps } from "../../../scripts/gpu.fixture";
import { CEILING } from "../../../scripts/test-tiers";
import { MeshInstance } from "../../core/mesh";
import { attachTexture, Camera, Views } from "../../core/rendering";
import { Transform } from "../../core/transform";
import { DEFAULT_PLUGINS } from "../../standard";
import { StandardRenderer } from "../../standard/rendering";
import { Outline, OutlinePlugin } from ".";

setDefaultTimeout(CEILING.gpu);

const subjects = gpuApps(import.meta.path, [
    { defaults: false, plugins: [...DEFAULT_PLUGINS, OutlinePlugin] },
]);

test("occluded outlines request and read the 4× depth lane without DepthPrepass", async () => {
    const { world } = subjects()[0]!;
    const camera = world.create();
    world.add(camera, Transform, { translation: [0, 0, 5, 0] });
    world.add(camera, Camera);
    world.add(camera, StandardRenderer);
    attachTexture(world, camera, { width: 48, height: 48 });

    const outlined = world.create();
    world.add(outlined, Transform);
    world.add(outlined, MeshInstance);
    world.add(outlined, Outline, { occlude: 1 });
    const occluder = world.create();
    world.add(occluder, Transform, { translation: [0.5, 0, 1.5, 0] });
    world.add(occluder, MeshInstance);

    const created: { label: string | undefined; sampleCount: number }[] = [];
    const device = world.gpu.device;
    const createTexture = device.createTexture.bind(device);
    device.createTexture = (descriptor) => {
        created.push({ label: descriptor.label, sampleCount: descriptor.sampleCount ?? 1 });
        return createTexture(descriptor);
    };
    device.pushErrorScope("validation");
    world.step(0);
    world.step(0);

    const view = world.resource(Views).get(camera)!;
    expect(view.depth).not.toBeNull();
    expect(created.find((target) => target.label === `standard-depth-${camera}`)?.sampleCount).toBe(
        4,
    );
    expect(await device.popErrorScope()).toBeNull();
});
