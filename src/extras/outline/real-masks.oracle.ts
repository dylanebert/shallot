import { test } from "bun:test";
import { LocalPlayer, Player } from "../../../examples/interact/src/player";
import { Room, RoomPlugin } from "../../../examples/interact/src/room";
import { gpuApps } from "../../../scripts/gpu.fixture";
import { Meshes, MeshInstance } from "../../core/mesh";
import { attachTexture, Camera } from "../../core/rendering";
import { Transform } from "../../core/transform";
import { DEFAULT_PLUGINS } from "../../standard";
import { setTransform } from "../../standard/physics";
import { PointShadows, StandardRenderer } from "../../standard/rendering";
import { Outline, OutlinePlugin } from ".";
import { captureSeeds } from "./mask.fixture";
import { coverage } from "./reference.fixture";

const subjects = gpuApps(import.meta.path, [
    { defaults: false, plugins: [...DEFAULT_PLUGINS, OutlinePlugin] },
    ...[0, 1].map((antialias) => ({
        defaults: false,
        plugins: [
            ...DEFAULT_PLUGINS,
            RoomPlugin,
            {
                name: "OracleAntialias",
                dependencies: [RoomPlugin],
                initialize(world: import("../../engine").World) {
                    world.storage(Camera).antialias.set(world.resource(Room).camera, antialias);
                },
            },
        ],
    })),
]);

function characterize(
    mask: Uint16Array,
    nearest: Uint16Array,
    width: number,
    height: number,
    bandWidth: number,
    name: string,
    distances?: Float32Array,
) {
    const inside = new Uint8Array(width * height);
    let seedCount = 0;
    for (let i = 0; i < inside.length; i++)
        if (mask[i * 2] !== 30000) {
            inside[i] = 1;
            seedCount++;
        }
    let distanceErrors = 0,
        coverageErrors = 0,
        bandPixels = 0;
    const radius = Math.ceil(bandWidth);
    for (let y = 0; y < height; y++)
        for (let x = 0; x < width; x++) {
            const i = y * width + x;
            let exact = inside[i] ? 0 : Infinity;
            // Exhaust every possible in-band seed. Farther seeds cannot affect the band.
            if (exact !== 0)
                for (let sy = Math.max(0, y - radius); sy <= Math.min(height - 1, y + radius); sy++)
                    for (
                        let sx = Math.max(0, x - radius);
                        sx <= Math.min(width - 1, x + radius);
                        sx++
                    )
                        if (inside[sy * width + sx])
                            exact = Math.min(exact, (sx - x) ** 2 + (sy - y) ** 2);
            const flood = distances
                ? inside[i]
                    ? 0
                    : distances[i]! ** 2
                : (nearest[i * 2]! - x) ** 2 + (nearest[i * 2 + 1]! - y) ** 2;
            if (exact > 0 && exact < bandWidth ** 2) bandPixels++;
            if (exact < bandWidth ** 2 && exact !== flood) distanceErrors++;
            // Ideal coverage isolates geometric error from float rounding.
            if (coverage(Math.sqrt(exact), bandWidth) !== coverage(Math.sqrt(flood), bandWidth))
                coverageErrors++;
        }
    console.info(
        JSON.stringify({
            name,
            resolution: [width, height],
            width: bandWidth,
            seedCount,
            bandPixels,
            distanceErrors,
            coverageErrors,
        }),
    );
}

for (const c of [
    { name: "band cube", size: 48, width: 4 },
    { name: "turnover cube", size: 32, width: 4 },
    { name: "turnover triangle", size: 32, width: 4, triangle: true },
    { name: "stacked cube (45-degree rotation)", size: 48, width: 3, rotated: true },
    { name: "depth cube partially occluded", size: 48, width: 4, occluded: "partial" },
    { name: "band cube fully occluded", size: 48, width: 4, occluded: "full" },
]) {
    test(`${c.name}: real-mask brute-force characterization`, async () => {
        const { world } = subjects()[0];
        const camera = world.create();
        world.add(camera, Transform, { translation: [0, 0, 5] });
        world.add(camera, Camera, { clearColor: 0x204060 });
        world.add(camera, StandardRenderer);
        attachTexture(world, camera, { width: c.size, height: c.size });
        const entity = world.create();
        world.add(
            entity,
            Transform,
            c.rotated ? { rotation: [0, 0, Math.sin(Math.PI / 8), Math.cos(Math.PI / 8)] } : {},
        );
        world.add(entity, MeshInstance);
        const meshes = world.resource(Meshes);
        const mesh = c.triangle
            ? meshes.add({
                  name: "triangle",
                  vertices: new Float32Array([
                      0.2, 0.2, 0, 0, 0, 0, 1, 0, 1.6, 0.2, 0, 0, 0, 0, 1, 0, 0.2, 1.6, 0, 0, 0, 0,
                      1, 0,
                  ]),
                  indices: new Uint32Array([0, 1, 2]),
              })
            : undefined;
        if (mesh !== undefined) world.storage(MeshInstance).mesh.set(entity, mesh);
        const occluder = c.occluded ? world.create() : undefined;
        if (occluder !== undefined) {
            world.add(
                occluder,
                Transform,
                c.occluded === "partial"
                    ? { translation: [0.5, 0, 1.5] }
                    : { translation: [0, 0, 2], scale: [4, 4, 1] },
            );
            world.add(occluder, MeshInstance);
        }
        world.add(entity, Outline, { width: c.width, occlude: occluder === undefined ? 0 : 1 });
        const capture = captureSeeds(world, camera, c.size, c.size);
        try {
            world.step(0);
            world.step(0);
            const { mask, nearest, distances } = await capture.read();
            characterize(mask, nearest, c.size, c.size, c.width, c.name, distances);
        } finally {
            capture.restore();
            world.destroy(entity);
            world.destroy(camera);
            if (occluder !== undefined) world.destroy(occluder);
            if (mesh !== undefined) meshes.remove(mesh);
            world.step(0);
        }
    });
}

for (const [width, height] of [
    [320, 180],
    [1280, 720],
] as const) {
    test(`interact at ${width}x${height}: real-mask brute-force characterization`, async () => {
        const { world } = subjects()[width === 320 ? 1 : 2]!;
        const room = world.resource(Room);
        world.resource(PointShadows).atlas = 512;
        world.resource(PointShadows).casters = 1;
        attachTexture(world, room.camera, { width, height });
        const capture = captureSeeds(world, room.camera, width, height);
        try {
            // The example's stepped check places the player; its controller poses the camera.
            for (const [index, distance, yaw, pitch] of [
                [0, 2.6, 0, 0],
                [1, 2.6, 0, 0],
                [2, 2.6, 0, 0],
                [2, 1.2, 0, 0],
                [2, 1.8, -0.45, 0.25],
                [2, 2.6, 0.45, -0.25],
            ] as const) {
                const target = room.targets[index]!;
                const t = world.storage(Transform);
                setTransform(
                    world,
                    room.player,
                    [
                        t.translation.x.get(target) + Math.sin(yaw) * Math.cos(pitch) * distance,
                        t.translation.y.get(target) + Math.sin(pitch) * distance - 0.7,
                        t.translation.z.get(target) + Math.cos(yaw) * Math.cos(pitch) * distance,
                    ],
                    [0, 0, 0, 1],
                );
                world.storage(LocalPlayer).viewYaw.set(room.player, yaw);
                world.storage(LocalPlayer).viewPitch.set(room.player, -pitch);
                world.storage(Player).gravity.set(room.player, 0);
                world.storage(Player).friction.set(room.player, 1000);
                world.step(1 / 60);
                world.step(1 / 60);
                if (!world.has(target, Outline))
                    throw new Error(`Interact target ${index} is not focused`);
                const { mask, nearest, distances } = await capture.read();
                characterize(
                    mask,
                    nearest,
                    width,
                    height,
                    world.storage(Outline).width.get(target),
                    `${["lamp", "button", "held"][index]} d=${distance} yaw=${yaw} pitch=${pitch}`,
                    distances,
                );
            }
        } finally {
            capture.restore();
        }
    });
}
