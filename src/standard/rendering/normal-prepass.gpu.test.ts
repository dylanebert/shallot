import { expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { gpuApps } from "../../../scripts/gpu.fixture";
import { MeshInstance } from "../../core/mesh";
import { attachTexture, Camera, DepthPrepass, NormalPrepass, Views } from "../../core/rendering";
import { viewTargetsKey } from "../../core/rendering/targets";
import { Transform } from "../../core/transform";
import { lookAtRotation, probeTexture, type World } from "../../engine";
import { StandardRenderer } from "./forward";
import { StandardRenderingPlugin } from "./index";
import { Materials, MeshMaterial, StandardMaterial } from "./material";
import { MeshRenderPlugin } from "./mesh-render";

setDefaultTimeout(TIMEOUT.gpu);
const angle = 0.37;
const cosine = Math.cos(angle);
const sine = Math.sin(angle);
const normals = [
    [cosine, sine, 0],
    [-cosine, -sine, 0],
    [-sine, cosine, 0],
    [sine, -cosine, 0],
    [0, 0, 1],
    [0, 0, -1],
    [0, 1, 0],
] as const;
const cameras: number[] = [];
let absent = 0;
let multisampled = 0;
function scene(world: World) {
    const material = world.resource(Materials).add(StandardMaterial());
    for (const ground of [false, true]) {
        const eid = world.create();
        world.add(eid, Transform, {
            translation: ground ? [6, -1, 0, 0] : [0, 0, 0, 0],
            scale: ground ? [4, 0.1, 4, 0] : [1, 2, 0.75, 0],
            rotation: ground ? [0, 0, 0, 1] : [0, 0, Math.sin(angle / 2), Math.cos(angle / 2)],
        });
        world.add(eid, MeshInstance);
        world.add(eid, MeshMaterial, material);
    }
    for (let i = 0; i < normals.length; i++) {
        const n = normals[i]!;
        const center: [number, number, number] = i === 6 ? [6, -0.95, 0] : [0, 0, 0];
        const eye: [number, number, number] = [
            center[0] + n[0] * 4,
            center[1] + n[1] * 4,
            center[2] + n[2] * 4,
        ];
        const camera = world.create();
        world.add(camera, Transform, {
            translation: [...eye, 0],
            rotation: lookAtRotation(...eye, ...center, 0, n[1] ? 0 : 1, n[1] ? 1 : 0),
        });
        world.add(camera, Camera, { antialias: 0 });
        world.add(camera, StandardRenderer);
        world.add(camera, NormalPrepass);
        attachTexture(world, camera, { width: 33, height: 33 });
        cameras.push(camera);
    }
    multisampled = world.create();
    world.add(multisampled, Transform, { translation: [0, 0, 4, 0] });
    world.add(multisampled, Camera);
    world.add(multisampled, StandardRenderer);
    world.add(multisampled, NormalPrepass);
    attachTexture(world, multisampled, { width: 33, height: 33 });
    absent = world.create();
    world.add(absent, Transform, { translation: [0, 0, 4, 0] });
    world.add(absent, Camera, { antialias: 0 });
    world.add(absent, StandardRenderer);
    world.add(absent, DepthPrepass);
    attachTexture(world, absent, { width: 33, height: 33 });
}
const subjects = gpuApps(import.meta.path, [
    {
        defaults: false,
        plugins: [
            StandardRenderingPlugin,
            MeshRenderPlugin,
            {
                name: "NormalScene",
                dependencies: [StandardRenderingPlugin, MeshRenderPlugin],
                initialize: scene,
            },
        ],
    },
]);
test("NormalPrepass writes each box face and the ground's world normal at its center", async () => {
    const { world } = subjects()[0];
    world.step(1 / 60);
    for (let i = 0; i < cameras.length; i++) {
        const target = world.resource(viewTargetsKey).normal.get(cameras[i]!)!;
        const result = await probeTexture(world, target.texture);
        const bytes = new DataView(result.bytes);
        for (let c = 0; c < 3; c++) {
            const packed = bytes.getUint32((16 * 33 + 16) * 4, true);
            const normal = (((packed >>> (c * 10)) & 1023) / 1023) * 2 - 1;
            expect(Math.abs(normal - normals[i]![c]!)).toBeLessThan(1e-2);
        }
    }
});
test("normal targets match the multisampled depth lane", () => {
    const { world } = subjects()[0];
    world.step(1 / 60);
    const targets = world.resource(viewTargetsKey);
    expect(targets.normal.get(multisampled)!.texture.sampleCount).toBe(4);
    expect(targets.normal.get(multisampled)!.sampleCount).toBe(
        targets.depth.get(multisampled)!.sampleCount,
    );
});
test("removing NormalPrepass retires its target", () => {
    const { world } = subjects()[0];
    world.remove(multisampled, NormalPrepass);
    world.step(1 / 60);
    expect(world.resource(viewTargetsKey).normal.has(multisampled)).toBe(false);
    expect(world.resource(Views).get(multisampled)!.normal).toBeNull();
});
test("a depth-only camera allocates no normal target", () => {
    const { world } = subjects()[0];
    world.step(1 / 60);
    expect(world.resource(viewTargetsKey).depth.has(absent)).toBe(true);
    expect(world.resource(viewTargetsKey).normal.has(absent)).toBe(false);
    expect(world.resource(Views).get(absent)!.normal).toBeNull();
});
test("a camera keeping its depth prepass drops and regains its normal attachment as NormalPrepass toggles", async () => {
    const { world } = subjects()[0];
    const device = world.gpu.device;
    const view = world.resource(Views).get(absent)!;
    const prepass = () => world.resource(viewTargetsKey).depth.get(absent)!.pass;
    const frame = async () => {
        device.pushErrorScope("validation");
        world.step(1 / 60);
        world.step(1 / 60);
        return (await device.popErrorScope())?.message;
    };
    world.add(absent, NormalPrepass);
    expect(await frame()).toBeUndefined();
    const first = view.normal;
    expect(first).not.toBeNull();
    expect(Array.from(prepass().colorAttachments, (a) => a?.view as GPUTextureView | null)).toEqual(
        [first],
    );

    world.remove(absent, NormalPrepass);
    expect(await frame()).toBeUndefined();
    expect(view.normal).toBeNull();
    expect(view.depth).not.toBeNull();
    expect(Array.from(prepass().colorAttachments)).toEqual([]);

    world.add(absent, NormalPrepass);
    expect(await frame()).toBeUndefined();
    expect(view.normal).not.toBeNull();
    expect(view.normal).not.toBe(first);
    expect(Array.from(prepass().colorAttachments, (a) => a?.view as GPUTextureView | null)).toEqual(
        [view.normal],
    );
    world.remove(absent, NormalPrepass);
    world.step(1 / 60);
});
