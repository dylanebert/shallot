import { expect, test } from "bun:test";
import { resizeViewport, World } from "../../engine";
import { component } from "../../engine/ecs/component";
import { GlobalTransform } from "../transform";
import { Camera, CameraMode, computeViewProj } from "./camera";
import { type View, Views } from "./view";
import { generateRay, screenToRay, viewportToWorld, worldToViewport } from "./viewport-to-world";

test("the NDC-to-world ray drops the aspect ratio, the near offset or the camera rotation, so a pick would miss the object under the cursor on a non-square canvas or a turned camera", () => {
    const Id: [number, number, number, number] = [0, 0, 0, 1];
    const fov = 60;
    const aspect = 16 / 9;
    const near = 0.05;

    // identity camera at the origin: ndc (0,0) gives camera forward (0,0,-1), origin `near` along it
    const centre = generateRay(0, 0, aspect, fov, near, [0, 0, 0], Id);
    expect([centre.dir[0], centre.dir[1], centre.dir[2]]).toEqual([
        expect.closeTo(0, 9),
        expect.closeTo(0, 9),
        expect.closeTo(-1, 9),
    ]);
    expect([centre.origin[0], centre.origin[1], centre.origin[2]]).toEqual([
        expect.closeTo(0, 9),
        expect.closeTo(0, 9),
        expect.closeTo(-near, 9),
    ]);

    // ndc (1,0) gives camera-space dir (aspect*t, 0, -1); the gold is that raw dir, normalized
    const t = Math.tan(((fov / 2) * Math.PI) / 180);
    const len = Math.hypot(aspect * t, 0, -1);
    const right = generateRay(1, 0, aspect, fov, near, [0, 0, 0], Id);
    expect(right.dir[0]).toBeCloseTo((aspect * t) / len, 9);
    expect(right.dir[1]).toBeCloseTo(0, 9);
    expect(right.dir[2]).toBeCloseTo(-1 / len, 9);

    // quat = +90 about Y rotates camera-forward (0,0,-1) to (-1,0,0); origin from (3,0,0) steps -X
    const q: [number, number, number, number] = [0, Math.SQRT1_2, 0, Math.SQRT1_2];
    const yawed = generateRay(0, 0, aspect, fov, near, [3, 0, 0], q);
    expect([yawed.dir[0], yawed.dir[1], yawed.dir[2]]).toEqual([
        expect.closeTo(-1, 6),
        expect.closeTo(0, 6),
        expect.closeTo(0, 6),
    ]);
    expect(yawed.origin[0]).toBeCloseTo(3 - near, 6);
});

test("the pixel-to-NDC conversion drops the y flip or mis-centres the canvas, so a cursor above the centre would pick a body below it", () => {
    const Id: [number, number, number, number] = [0, 0, 0, 1];
    const W = 1600;
    const H = 900;

    const px = screenToRay(W / 2, H / 2, W, H, 60, 0.05, [0, 0, 0], Id);
    const ndc = generateRay(0, 0, W / H, 60, 0.05, [0, 0, 0], Id);
    expect(px.dir[0]).toBeCloseTo(ndc.dir[0], 9);
    expect(px.dir[1]).toBeCloseTo(ndc.dir[1], 9);
    expect(px.dir[2]).toBeCloseTo(ndc.dir[2], 9);

    // pixel (0,0) is screen top-left, i.e. ndc (-1, +1): the ray tilts -x and +y
    const corner = screenToRay(0, 0, W, H, 60, 0.05, [0, 0, 0], Id);
    expect(corner.dir[0]).toBeLessThan(0);
    expect(corner.dir[1]).toBeGreaterThan(0);
    expect(corner.dir[2]).toBeLessThan(0);
});

test("orthographic pixels shift ray origins by size and aspect while keeping parallel camera-forward directions", () => {
    const world = new World();
    world.registry.register(component("Camera", { ...Camera }));
    world.registry.register(component("GlobalTransform", { ...GlobalTransform }));
    const camera = world.create();
    world.add(camera, Camera, { mode: CameraMode.Orthographic, size: 4, near: 0.5, fov: 90 });
    world.add(camera, GlobalTransform);
    world.storage(GlobalTransform).translation.set(camera, 3, 2, 5, 0);
    world.storage(GlobalTransform).rotation.set(camera, 0, Math.SQRT1_2, 0, Math.SQRT1_2);
    resizeViewport(world, 0, 800, 400, 1);
    world.resource(Views).set(camera, { viewportIndex: 0, camera: world.ref(camera) } as View);
    try {
        const centre = viewportToWorld(world, camera, 400, 200)!;
        const offset = viewportToWorld(world, camera, 600, 100)!;
        // Size 4 and aspect 2 give local offset (2, 1, -0.5); +90 Y turns it to (-0.5, 1, -2).
        expect(centre.origin).toEqual([
            expect.closeTo(2.5, 6),
            expect.closeTo(2, 6),
            expect.closeTo(5, 6),
        ]);
        expect(offset.origin).toEqual([
            expect.closeTo(2.5, 6),
            expect.closeTo(3, 6),
            expect.closeTo(3, 6),
        ]);
        expect(centre.dir).toEqual([
            expect.closeTo(-1, 6),
            expect.closeTo(0, 6),
            expect.closeTo(0, 6),
        ]);
        expect(offset.dir).toEqual([
            expect.closeTo(-1, 6),
            expect.closeTo(0, 6),
            expect.closeTo(0, 6),
        ]);
    } finally {
        world.dispose();
    }
});

test("a recycled camera eid does not cast rays through its predecessor's viewport before the next frame", () => {
    const world = new World();
    world.registry.register(component("Camera", { ...Camera }));
    world.registry.register(component("GlobalTransform", { ...GlobalTransform }));
    const camera = world.create();
    world.add(camera, Camera);
    world.add(camera, GlobalTransform);
    resizeViewport(world, 0, 800, 400, 1);
    world.resource(Views).set(camera, { viewportIndex: 0, camera: world.ref(camera) } as View);
    try {
        expect(viewportToWorld(world, camera, 400, 200)).not.toBeNull();
        world.destroy(camera);
        const recycled = world.create();
        expect(recycled).toBe(camera);
        world.add(recycled, Camera);
        world.add(recycled, GlobalTransform);
        expect(viewportToWorld(world, recycled, 400, 200)).toBeNull();
    } finally {
        world.dispose();
    }
});

function placedCamera(mode: number, view: Partial<View> = { viewportIndex: 0 }) {
    const world = new World();
    world.registry.register(component("Camera", { ...Camera }));
    world.registry.register(component("GlobalTransform", { ...GlobalTransform }));
    const camera = world.create();
    world.add(camera, Camera, { mode, size: 3, near: 0.5, far: 40, fov: 70 });
    world.add(camera, GlobalTransform);
    world.storage(GlobalTransform).translation.set(camera, 1, 2, 9, 0);
    // A turned and tilted camera, so a dropped conjugate or axis shows.
    const half = 0.4;
    const tilt = -0.15;
    world
        .storage(GlobalTransform)
        .rotation.set(
            camera,
            Math.cos(half) * Math.sin(tilt),
            Math.sin(half) * Math.cos(tilt),
            -Math.sin(half) * Math.sin(tilt),
            Math.cos(half) * Math.cos(tilt),
        );
    world.storage(GlobalTransform).scale.set(camera, 1, 1, 1, 0);
    resizeViewport(world, 0, 800, 400, 1);
    world.resource(Views).set(camera, { ...view, camera: world.ref(camera) } as View);
    return { world, camera };
}

test("worldToViewport lands where viewportToWorld's perspective ray came from", () => {
    const { world, camera } = placedCamera(CameraMode.Perspective);
    try {
        for (const [x, y] of [
            [400, 200],
            [10, 390],
            [730, 25],
        ] as const) {
            const ray = viewportToWorld(world, camera, x, y)!;
            for (const distance of [1, 25]) {
                const point = ray.origin.map((v, i) => v + ray.dir[i]! * distance) as [
                    number,
                    number,
                    number,
                ];
                expect(worldToViewport(world, camera, point)).toEqual([
                    expect.closeTo(x, 4),
                    expect.closeTo(y, 4),
                ]);
            }
        }
    } finally {
        world.dispose();
    }
});

for (const [name, mode] of [
    ["perspective", CameraMode.Perspective],
    ["orthographic", CameraMode.Orthographic],
] as const) {
    test(`worldToViewport agrees with the ${name} view-projection the renderer draws with`, () => {
        const { world, camera } = placedCamera(mode);
        const clip = new Float32Array(16);
        computeViewProj(world, camera, 2, clip);
        try {
            for (const point of [
                [0, 1, 0],
                [3, 3, -4],
                [-2, 0.5, 2],
            ] as const) {
                const [x, y, z] = point;
                const cx = clip[0]! * x + clip[4]! * y + clip[8]! * z + clip[12]!;
                const cy = clip[1]! * x + clip[5]! * y + clip[9]! * z + clip[13]!;
                const cw = clip[3]! * x + clip[7]! * y + clip[11]! * z + clip[15]!;
                expect(worldToViewport(world, camera, point)).toEqual([
                    expect.closeTo(((cx / cw + 1) / 2) * 800, 2),
                    expect.closeTo(((1 - cy / cw) / 2) * 400, 2),
                ]);
            }
        } finally {
            world.dispose();
        }
    });
}

test("worldToViewport refuses a point nearer than near, an orthographic point beyond far, and a camera without placement or target", () => {
    const perspective = placedCamera(CameraMode.Perspective);
    const orthographic = placedCamera(CameraMode.Orthographic);
    const depthOnly = placedCamera(CameraMode.Perspective, { viewportIndex: -1 });
    try {
        const ahead = (world: World, camera: number, distance: number) => {
            const ray = viewportToWorld(world, camera, 400, 200)!;
            return ray.origin.map((v, i) => v + ray.dir[i]! * distance) as [number, number, number];
        };
        const { world, camera } = perspective;
        expect(worldToViewport(world, camera, ahead(world, camera, 0.01))).not.toBeNull();
        expect(worldToViewport(world, camera, ahead(world, camera, -0.01))).toBeNull();
        expect(worldToViewport(world, camera, ahead(world, camera, -20))).toBeNull();
        expect(worldToViewport(world, camera, ahead(world, camera, 1e6))).not.toBeNull();
        const ortho = orthographic;
        expect(
            worldToViewport(ortho.world, ortho.camera, ahead(ortho.world, ortho.camera, 39.4)),
        ).not.toBeNull();
        expect(
            worldToViewport(ortho.world, ortho.camera, ahead(ortho.world, ortho.camera, 39.6)),
        ).toBeNull();
        expect(worldToViewport(depthOnly.world, depthOnly.camera, [0, 0, 0])).toBeNull();
        expect(viewportToWorld(depthOnly.world, depthOnly.camera, 400, 200)).toBeNull();
        world.remove(camera, GlobalTransform);
        expect(worldToViewport(world, camera, [0, 0, 0])).toBeNull();
    } finally {
        perspective.world.dispose();
        orthographic.world.dispose();
        depthOnly.world.dispose();
    }
});
