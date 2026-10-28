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
    type World,
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
import { FIELD, groundAt } from "./hills";
import { DEPTH, RESOLUTION, Trail } from "./trail";

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
const POINTS = 1024;
const config = (width = 320, height = 240): Parameters<typeof openGpuApp>[0] => ({
    plugins: [
        StandardRenderingPlugin,
        DemoPlugin,
        {
            name: "TestView",
            dependencies: [DemoPlugin],
            initialize(world) {
                const camera = world.resource(SnowTrails).camera;
                world.storage(Camera).antialias.set(camera, 0);
                attachTexture(world, camera, { width, height });
                world.add(camera, DepthPrepass);
                const s = world.resource(DepthProbe),
                    device = world.gpu.device;
                s.output = device.createBuffer({
                    size: POINTS * 4,
                    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
                });
                s.points = device.createBuffer({
                    size: POINTS * 8,
                    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
                });
                world.own(s.output);
                world.own(s.points);
                const module = device.createShaderModule({
                    code: `
            @group(0) @binding(0) var depth: texture_depth_2d;
            @group(0) @binding(1) var<storage,read> points: array<vec2i>;
            @group(0) @binding(2) var<storage,read_write> result: array<f32>;
            @compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id: vec3u){result[id.x]=textureLoad(depth,points[id.x],0);}
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
                                        buffer: { type: "read-only-storage" },
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
});
const subjects: Awaited<ReturnType<typeof openGpuApp>>[] = [];
let device: GPUDevice;
beforeAll(async () => {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error("Snow trails needs a WebGPU adapter");
    device = await adapter.requestDevice({ requiredFeatures: ["indirect-first-instance"] });
    for (let i = 0; i < 8; i++) subjects.push(await openGpuApp({ ...config(), device, adapter }));
    for (let i = 0; i < 2; i++)
        subjects.push(await openGpuApp({ ...config(1024, 512), device, adapter }));
}, TIMEOUT.startup);
afterAll(() => {
    for (const app of subjects) app.dispose();
    device?.destroy();
});
const apps = () => subjects;
test("a resting hillside boulder presses through the local snow to ground", async () => {
    const world = apps()[5].world;
    const { boulder } = world.resource(SnowTrails);
    world.tick();
    setTransform(world, boulder, [5, groundAt(5, 3) + RADIUS, 3], [0, 0, 0, 1]);
    setLinearVelocity(world, boulder, { x: 0, y: 0, z: 0 });
    for (let i = 0; i < 90; i++) world.tick();
    world.step(0);
    world.step(0);
    world.step(0);
    const p = world.storage(GlobalTransform).translation;
    const x = p.x.get(boulder),
        z = p.z.get(boulder);
    expect(Math.abs(p.y.get(boulder) - RADIUS - groundAt(x, z))).toBeLessThan(0.08);
    const data = new Float32Array((await probeTexture(world, world.resource(Trail).image!)).bytes);
    expect(pixel(data, x, z)).toBeGreaterThan(0.85);
});
test("the opening fall and roll depart from a straight line by more than the boulder radius", () => {
    const world = apps()[6].world;
    const { boulder } = world.resource(SnowTrails);
    const positions: [number, number][] = [];
    for (let i = 0; i < 360; i++) {
        world.tick();
        const p = world.storage(GlobalTransform).translation;
        positions.push([p.x.get(boulder), p.z.get(boulder)]);
    }
    const [x0, z0] = positions[0],
        [x1, z1] = positions.at(-1)!;
    const length = Math.hypot(x1 - x0, z1 - z0);
    expect(
        Math.max(
            ...positions.map(
                ([x, z]) =>
                    Math.abs((x - x0) * (z1 - z0) - (z - z0) * (x1 - x0)) / Math.max(length, 1e-6),
            ),
        ),
    ).toBeGreaterThan(RADIUS);
});
test("a boulder dragged hard toward each side and released stays on the field", () => {
    const world = apps()[7].world;
    resizeViewport(world, 0, 320, 240, 1);
    world.step(1 / 60);
    const { boulder, camera } = world.resource(SnowTrails);
    const grab = world.resource(Grab);
    const p = world.storage(GlobalTransform).translation;
    for (const [dx, dz] of [
        [1, 0],
        [-1, 0],
        [0, 1],
        [0, -1],
    ]) {
        setTransform(world, boulder, [0, groundAt(0, 0) + RADIUS, 0], [0, 0, 0, 1]);
        setLinearVelocity(world, boulder, { x: 0, y: 0, z: 0 });
        world.step(1 / 60);
        pointerMove(world, ...worldToViewport(world, camera, [0, groundAt(0, 0) + RADIUS, 0])!);
        pressButton(world, "left");
        world.step(0);
        expect(grab.body).toBe(boulder);
        // Pull the held point at 30 m/s from the field's center to 6 m past the side, then let go.
        grab.fraction = 0;
        for (let i = 0; i < 36; i++) {
            const d = Math.min(30 * (i + 1) * (1 / 60), FIELD / 2 + 6);
            grab.origin.x = dx * d;
            grab.origin.y = groundAt(0, 0) + RADIUS + 3;
            grab.origin.z = dz * d;
            world.tick();
        }
        releaseButton(world, "left");
        world.step(0);
        expect(grab.body).toBe(0);
        for (let i = 0; i < 300; i++) world.tick();
        const x = p.x.get(boulder),
            z = p.z.get(boulder);
        expect(Math.max(Math.abs(x), Math.abs(z))).toBeLessThan(FIELD / 2);
        expect(p.y.get(boulder) - RADIUS).toBeGreaterThan(groundAt(x, z) - 0.1);
    }
});
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
    setTransform(world, boulder, [-6, groundAt(-6, 0) + RADIUS, 0], [0, 0, 0, 1]);
    setLinearVelocity(world, boulder, { x: 3, y: 0, z: 0 });
    const positions: [number, number][] = [];
    for (let i = 0; i < 45; i++) {
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
    setTransform(world, boulder, [0, groundAt(0, 4) + RADIUS + DEPTH / 2, 4], [0, 0, 0, 1]);
    setLinearVelocity(world, boulder, { x: 0, y: 0, z: 0 });
    const [x, y] = worldToViewport(world, camera, [0, groundAt(0, 4) + RADIUS + DEPTH / 2, 4])!;
    pointerMove(world, x, y);
    pressButton(world, "left");
    world.step(0);
    const grab = world.resource(Grab);
    expect(grab.body).toBe(boulder);
    // Hold at its center so the spring does not swing an off-center anchor beneath the rock.
    const p0 = world.storage(GlobalTransform).translation;
    const center = { x: p0.x.get(boulder), y: p0.y.get(boulder), z: p0.z.get(boulder) };
    const t = grab.translation,
        o = grab.origin;
    grab.hit.point = center;
    grab.hit.fraction =
        ((center.x - o.x) * t.x + (center.y - o.y) * t.y + (center.z - o.z) * t.z) /
        (t.x * t.x + t.y * t.y + t.z * t.z);
    grab.hit.body = boulder;
    pick(world, true);
    for (let i = 0; i < 90; i++) world.tick();
    world.step(0);
    world.step(0);
    const p = world.storage(GlobalTransform).translation;
    const bottom = p.y.get(boulder) - RADIUS - groundAt(p.x.get(boulder), p.z.get(boulder));
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
        setTransform(world, boulder, [0, groundAt(0, 0) + RADIUS, 0], [0, 0, 0, 1]);
        setLinearVelocity(world, boulder, { x: 0, y: 0, z: 0 });
        world.step(1 / 60);
        const [x, y] = worldToViewport(world, camera, [0, groundAt(0, 0) + RADIUS, 0])!;
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
        world.step(1 / 60);
        for (let i = 0; i < 44; i++) world.tick();
        world.step(0);
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

/** Looks straight down from 20 m over (x, z), orthographic with half-height `size`, with the boulder parked aside. */
function lookDown(world: World, x: number, z: number, size: number) {
    const { camera, boulder } = world.resource(SnowTrails);
    setTransform(world, boulder, [8, groundAt(8, 8) + RADIUS, 8], [0, 0, 0, 1]);
    setLinearVelocity(world, boulder, { x: 0, y: 0, z: 0 });
    world.remove(camera, Orbit);
    world.storage(Camera).mode.set(camera, CameraMode.Orthographic);
    world.storage(Camera).size.set(camera, size);
    world.storage(Camera).near.set(camera, 0);
    world.storage(Camera).far.set(camera, 30);
    world.storage(Transform).translation.set(camera, x, 20, z);
    world.storage(Transform).rotation.set(camera, ...lookAtRotation(x, 20, z, x, 0, z, 0, 0, -1));
    world.tick();
    resetInterpolation(world, camera);
    resetInterpolation(world, boulder);
    world.step(0);
}
/** World heights of the drawn surface under world points, from the top-down camera's depth. */
async function heightsAt(world: World, points: [number, number][]) {
    const { camera } = world.resource(SnowTrails);
    const s = world.resource(DepthProbe);
    const pixels = new Int32Array(POINTS * 2);
    for (const [i, [x, z]] of points.entries())
        pixels.set(worldToViewport(world, camera, [x, 0, z])!.map(Math.floor), i * 2);
    world.gpu.device.queue.writeBuffer(s.points!, 0, pixels);
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
                    pass.dispatchWorkgroups(POINTS / 64);
                    pass.end();
                },
            })
        ).bytes,
    );
    return points.map((_, i) => 20 - (1 - depth[i]) * 30);
}
test("the stepped snow frame is lower and differently shaded on the trail than beside it", async () => {
    const world = apps()[4].world;
    const rock = world.resource(SnowTrails).boulder;
    world.tick();
    setTransform(world, rock, [-4, groundAt(-4, 0) + RADIUS, 0], [0, 0, 0, 1]);
    setLinearVelocity(world, rock, { x: 0, y: 0, z: 0 });
    for (let i = 0; i < 3; i++) world.step(1 / 60);
    world.step(0);
    world.step(0);
    lookDown(world, 0, 0, 12);
    const [on, off] = await heightsAt(world, [
        [-4, 0],
        [-4, 2],
    ]);
    expect(off).toBeCloseTo(groundAt(-4, 2) + DEPTH, 1);
    expect(off - on + groundAt(-4, 0) - groundAt(-4, 2)).toBeGreaterThan(DEPTH * 0.8);
    const { camera } = world.resource(SnowTrails);
    const frame = await captureTexture(world, camera);
    const color = ([x, z]: number[]) => {
        const [u, v] = worldToViewport(world, camera, [x, 0, z])!;
        const i = (Math.floor(v) * frame.width + Math.floor(u)) * 4;
        return frame.rgba.slice(i, i + 3);
    };
    const a = color([-4, 0]),
        b = color([-4, 2]);
    expect(Math.abs(a[0] - b[0]) + Math.abs(a[1] - b[1]) + Math.abs(a[2] - b[2])).toBeGreaterThan(
        40,
    );
});
/** The stepped page's opening: 48 rendered steps of 1/20 s, returning the boulder's moving positions. */
function opening(world: World) {
    const { boulder } = world.resource(SnowTrails);
    const p = world.storage(GlobalTransform).translation;
    const path: [number, number][] = [];
    for (let i = 0; i < 48; i++) {
        world.step(1 / 20);
        const [x, z] = [p.x.get(boulder), p.z.get(boulder)];
        const last = path.at(-1);
        if (!last || Math.hypot(x - last[0], z - last[1]) > 0.01) path.push([x, z]);
    }
    return path;
}
/** Points every `spacing` meters along a polyline. */
function along(path: [number, number][], spacing: number) {
    const out: [number, number][] = [];
    for (let i = 1; i < path.length; i++) {
        const [x0, z0] = path[i - 1],
            [x1, z1] = path[i];
        const length = Math.hypot(x1 - x0, z1 - z0);
        for (let s = 0; s < length; s += spacing)
            out.push([x0 + ((x1 - x0) * s) / length, z0 + ((z1 - z0) * s) / length]);
    }
    return out;
}
/** Steps the opening and looks down on it, returning its path and its center line. */
function openingFromAbove(world: World) {
    const path = opening(world);
    world.step(0);
    const xs = path.map(([x]) => x),
        zs = path.map(([, z]) => z);
    const [x0, x1, z0, z1] = [Math.min(...xs), Math.max(...xs), Math.min(...zs), Math.max(...zs)];
    lookDown(world, (x0 + x1) / 2, (z0 + z1) / 2, Math.max(z1 - z0, (x1 - x0) / 2) / 2 + 1.5);
    // From one radius past the landing to one radius short of the last frame, every centimeter.
    return { path, line: along(path, 0.01).slice(RADIUS * 100, -RADIUS * 100) };
}
test("the opening groove's floor varies along its center line by under 3.75 mm", async () => {
    const world = apps()[8].world;
    const { line } = openingFromAbove(world);
    const tops = (await heightsAt(world, line)).map((y, i) => y - groundAt(...line[i]!));
    // Against the mean over 30 cm either side, which follows the boulder's own rise and fall.
    const window = 30;
    const deviation = tops.map((top, i) => {
        const near = tops.slice(Math.max(0, i - window), i + window + 1);
        return Math.abs(top - near.reduce((a, b) => a + b) / near.length);
    });
    expect(Math.max(...deviation)).toBeLessThan(0.00375);
});
test("no texel where the opening trail starts is darker than its floor by more than 40", async () => {
    const world = apps()[9].world;
    const { path, line } = openingFromAbove(world);
    const { camera } = world.resource(SnowTrails);
    const frame = await captureTexture(world, camera);
    const luminance = ([x, z]: [number, number]) => {
        const [u, v] = worldToViewport(world, camera, [x, 0, z])!;
        const i = (Math.floor(v) * frame.width + Math.floor(u)) * 4;
        return 0.2126 * frame.rgba[i]! + 0.7152 * frame.rgba[i + 1]! + 0.0722 * frame.rgba[i + 2]!;
    };
    const floor = line.map(luminance).sort((a, b) => a - b)[line.length >> 1]!;
    const trail = new Float32Array((await probeTexture(world, world.resource(Trail).image!)).bytes);
    // Every pressed point, a centimeter apart, within 1.5 m of where the boulder landed.
    const [lx, lz] = path[0]!;
    let darkest = Infinity;
    for (let x = lx - 1.5; x <= lx + 1.5; x += 0.01)
        for (let z = lz - 1.5; z <= lz + 1.5; z += 0.01)
            if (Math.hypot(x - lx, z - lz) < 1.5 && pixel(trail, x, z) > 0)
                darkest = Math.min(darkest, luminance([x, z]));
    expect(floor - darkest).toBeLessThan(40);
});
