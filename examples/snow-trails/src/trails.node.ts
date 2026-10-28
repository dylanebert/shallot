import { afterAll, beforeAll, expect, setDefaultTimeout, test } from "bun:test";
import {
    Camera,
    CameraMode,
    GlobalTransform,
    lookAtRotation,
    pointerMove,
    precompile,
    pressButton,
    probeBuffer,
    probeTexture,
    releaseButton,
    resetInterpolation,
    resizeViewport,
    StandardRenderingPlugin,
    Transform,
    touchPoint,
} from "@dylanebert/shallot";
import { Orbit, OrbitMode } from "@dylanebert/shallot/extras";
import {
    attachTexture,
    captureTexture,
    DepthPrepass,
    Views,
    worldToViewport,
} from "@dylanebert/shallot/rendering";
import { setLinearVelocity, setTransform } from "@dylanebert/shallot/standard/physics";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { openGpuApp } from "@dylanebert/shallot/testing/gpu";
import { DemoPlugin, RADIUS, SnowTrails } from "./demo";
import { Grab, pick } from "./grab";
import { DEPTH, FIELD, RESOLUTION, Trail } from "./trail";

setDefaultTimeout(TIMEOUT.node);
const { setupGlobals } = await import("@dylanebert/shallot/webgpu");
await setupGlobals();
const DepthProbe = {
    create: () => ({
        pipeline: null as GPUComputePipeline | null,
        output: null as GPUBuffer | null,
        points: null as GPUBuffer | null,
    }),
};
const config: Parameters<typeof openGpuApp>[0] = {
    plugins: [
        StandardRenderingPlugin,
        DemoPlugin,
        {
            name: "TestView",
            dependencies: [DemoPlugin],
            initialize(world) {
                const camera = world.resource(SnowTrails).camera;
                world.storage(Camera).antialias.set(camera, 0);
                attachTexture(world, camera, { width: 320, height: 240 });
                world.add(camera, DepthPrepass);
                const s = world.resource(DepthProbe),
                    device = world.gpu.device;
                s.output = device.createBuffer({
                    size: 8,
                    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
                });
                s.points = device.createBuffer({
                    size: 16,
                    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
                });
                world.own(s.output);
                world.own(s.points);
                const module = device.createShaderModule({
                    code: `
            @group(0) @binding(0) var depth: texture_depth_2d;
            @group(0) @binding(1) var<uniform> points: vec4i;
            @group(0) @binding(2) var<storage,read_write> result: array<f32>;
            @compute @workgroup_size(1) fn main(){result[0]=textureLoad(depth,points.xy,0);result[1]=textureLoad(depth,points.zw,0);}
        `,
                });
                s.pipeline = device.createComputePipeline({
                    label: "snow:test-depth",
                    layout: device.createPipelineLayout({
                        bindGroupLayouts: [
                            device.createBindGroupLayout({
                                entries: [
                                    {
                                        binding: 0,
                                        visibility: GPUShaderStage.COMPUTE,
                                        texture: { sampleType: "depth" },
                                    },
                                    {
                                        binding: 1,
                                        visibility: GPUShaderStage.COMPUTE,
                                        buffer: { type: "uniform" },
                                    },
                                    {
                                        binding: 2,
                                        visibility: GPUShaderStage.COMPUTE,
                                        buffer: { type: "storage" },
                                    },
                                ],
                            }),
                        ],
                    }),
                    compute: { module },
                });
                precompile(world, "snow depth probe", () => [s.pipeline!]);
            },
        },
    ],
};
const subjects: Awaited<ReturnType<typeof openGpuApp>>[] = [];
let device: GPUDevice;
beforeAll(async () => {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error("Snow trails needs a WebGPU adapter");
    device = await adapter.requestDevice({ requiredFeatures: ["indirect-first-instance"] });
    for (let i = 0; i < 5; i++) subjects.push(await openGpuApp({ ...config, device, adapter }));
}, TIMEOUT.startup);
afterAll(() => {
    for (const app of subjects) app.dispose();
    device?.destroy();
});
const apps = () => subjects;
function pixel(data: Float32Array, x: number, z: number) {
    return data[
        Math.floor((0.5 - z / FIELD) * RESOLUTION) * RESOLUTION +
            Math.floor((x / FIELD + 0.5) * RESOLUTION)
    ];
}
test("a rolling boulder leaves a deep persistent trail under every passed bottom and none off the path", async () => {
    const world = apps()[0].world;
    const { boulder } = world.resource(SnowTrails);
    world.tick();
    setTransform(world, boulder, [-6, RADIUS, 0], [0, 0, 0, 1]);
    setLinearVelocity(world, boulder, { x: 3, y: 0, z: 0 });
    const positions: [number, number][] = [];
    for (let i = 0; i < 120; i++) {
        world.step(1 / 60);
        const p = world.storage(GlobalTransform).translation;
        positions.push([p.x.get(boulder), p.z.get(boulder)]);
    }
    world.step(0);
    world.step(0);
    const data = new Float32Array((await probeTexture(world, world.resource(Trail).image!)).bytes);
    for (const [x, z] of positions) expect(pixel(data, x, z)).toBeGreaterThan(0.85);
    expect(pixel(data, -4, 3)).toBe(0);
    expect(pixel(data, 4, -3)).toBe(0);
});

test("a grab holding the bottom halfway into snow records half the snow depth", async () => {
    const world = apps()[1].world;
    world.step(1 / 60);
    const { boulder, camera } = world.resource(SnowTrails);
    resizeViewport(world, 0, 320, 240, 1);
    setTransform(world, boulder, [0, RADIUS + DEPTH / 2, 4], [0, 0, 0, 1]);
    setLinearVelocity(world, boulder, { x: 0, y: 0, z: 0 });
    const [x, y] = worldToViewport(world, camera, [0, RADIUS + DEPTH / 2, 4])!;
    pointerMove(world, x, y);
    pressButton(world, "left");
    world.step(0);
    const grab = world.resource(Grab);
    expect(grab.body).toBe(boulder);
    // Hold at its centre so the spring does not swing an off-centre anchor beneath the rock.
    const p0 = world.storage(GlobalTransform).translation;
    const centre = { x: p0.x.get(boulder), y: p0.y.get(boulder), z: p0.z.get(boulder) };
    const t = grab.translation,
        o = grab.origin;
    grab.hit.point = centre;
    grab.hit.fraction =
        ((centre.x - o.x) * t.x + (centre.y - o.y) * t.y + (centre.z - o.z) * t.z) /
        (t.x * t.x + t.y * t.y + t.z * t.z);
    grab.hit.body = boulder;
    pick(world, true);
    for (let i = 0; i < 90; i++) world.step(1 / 60);
    world.step(0);
    world.step(0);
    const p = world.storage(GlobalTransform).translation;
    const bottom = p.y.get(boulder) - RADIUS;
    expect(bottom).toBeGreaterThan(DEPTH * 0.4);
    expect(bottom).toBeLessThan(DEPTH * 0.6);
    const data = new Float32Array((await probeTexture(world, world.resource(Trail).image!)).bytes);
    expect(pixel(data, p.x.get(boulder), p.z.get(boulder))).toBeGreaterThan(0.4);
    expect(pixel(data, p.x.get(boulder), p.z.get(boulder))).toBeLessThan(0.6);
    releaseButton(world, "left");
    world.step(0);
});
for (const touch of [false, true])
    test(`${touch ? "touch" : "pointer"} drags the boulder with orbit held, but snow presses orbit`, () => {
        const world = apps()[touch ? 3 : 2].world;
        resizeViewport(world, 0, 320, 240, 1);
        world.step(1 / 60);
        const { boulder, camera } = world.resource(SnowTrails);
        setTransform(world, boulder, [0, RADIUS, 0], [0, 0, 0, 1]);
        setLinearVelocity(world, boulder, { x: 0, y: 0, z: 0 });
        world.step(1 / 60);
        const [x, y] = worldToViewport(world, camera, [0, RADIUS, 0])!;
        const down = (x: number, y: number) => {
            if (touch) touchPoint(world, 1, x, y);
            else {
                pointerMove(world, x, y);
                pressButton(world, "left");
            }
        };
        const move = (x: number, y: number) => {
            if (touch) touchPoint(world, 1, x, y);
            else pointerMove(world, x, y, 40, 0);
        };
        const up = () => {
            if (touch) touchPoint(world, 1, x + 50, y, false);
            else releaseButton(world, "left");
        };
        down(x, y);
        world.step(0);
        expect(world.resource(Grab).body).toBe(boulder);
        expect(world.storage(Orbit).mode.get(camera)).toBe(OrbitMode.Locked);
        const yaw = world.storage(Orbit).yaw.get(camera);
        move(x + 50, y);
        for (let i = 0; i < 45; i++) world.step(1 / 60);
        expect(world.storage(Orbit).yaw.get(camera)).toBe(yaw);
        expect(world.storage(GlobalTransform).translation.x.get(boulder)).toBeGreaterThan(1);
        up();
        world.step(0);
        expect(world.storage(Orbit).mode.get(camera)).toBe(OrbitMode.Free);
        const snow = worldToViewport(world, camera, [-8, DEPTH, -5])!;
        down(...snow);
        world.step(0);
        expect(world.resource(Grab).body).toBe(0);
        const before = world.storage(Orbit).yaw.get(camera);
        move(snow[0] + 40, snow[1]);
        world.step(1 / 60);
        expect(world.storage(Orbit).yaw.get(camera)).not.toBe(before);
        up();
        world.step(0);
    });

test("the stepped snow frame is lower and differently shaded on the trail than beside it", async () => {
    const world = apps()[4].world;
    for (let i = 0; i < 120; i++) world.step(1 / 60);
    world.step(0);
    world.step(0);
    const { camera, boulder } = world.resource(SnowTrails);
    setTransform(world, boulder, [8, RADIUS, 8], [0, 0, 0, 1]);
    world.remove(camera, Orbit);
    world.storage(Camera).mode.set(camera, CameraMode.Orthographic);
    world.storage(Camera).size.set(camera, 12);
    world.storage(Camera).near.set(camera, 0);
    world.storage(Camera).far.set(camera, 30);
    world.storage(Transform).translation.set(camera, 0, 20, 0);
    world.storage(Transform).rotation.set(camera, ...lookAtRotation(0, 20, 0, 0, 0, 0, 0, 0, -1));
    world.tick();
    resetInterpolation(world, camera);
    world.step(0);
    const on = worldToViewport(world, camera, [-4, 0, 0])!;
    const off = worldToViewport(world, camera, [-4, DEPTH, 2])!;
    const s = world.resource(DepthProbe);
    world.gpu.device.queue.writeBuffer(
        s.points!,
        0,
        new Int32Array([...on.map(Math.floor), ...off.map(Math.floor)]),
    );
    const group = world.gpu.device.createBindGroup({
        layout: s.pipeline!.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: world.resource(Views).get(camera)!.depth! },
            { binding: 1, resource: { buffer: s.points! } },
            { binding: 2, resource: { buffer: s.output! } },
        ],
    });
    const depth = new Float32Array(
        (
            await probeBuffer(world, s.output!, {
                encode(encoder) {
                    const pass = encoder.beginComputePass();
                    pass.setPipeline(s.pipeline!);
                    pass.setBindGroup(0, group);
                    pass.dispatchWorkgroups(1);
                    pass.end();
                },
            })
        ).bytes,
    );
    expect(depth[1] - depth[0]).toBeGreaterThan((DEPTH / 30) * 0.8);
    const frame = await captureTexture(world, camera);
    const color = (p: number[]) => {
        const i = (Math.floor(p[1]) * frame.width + Math.floor(p[0])) * 4;
        return frame.rgba.slice(i, i + 3);
    };
    const a = color(on),
        b = color(off);
    expect(Math.abs(a[0] - b[0]) + Math.abs(a[1] - b[1]) + Math.abs(a[2] - b[2])).toBeGreaterThan(
        40,
    );
});
