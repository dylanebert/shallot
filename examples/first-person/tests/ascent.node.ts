import { setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";

setDefaultTimeout(TIMEOUT.node);

import {
    Body,
    Character,
    type createApp,
    GlobalTransform,
    GroundState,
    PlayerInput,
    Shape,
} from "@dylanebert/shallot";
import { attachTexture, Camera, type Capture, captureTexture } from "@dylanebert/shallot/rendering";
import { setupGlobals } from "@dylanebert/shallot/webgpu";
import { Route } from "../src/demo";
import { APP_PLUGINS } from "../src/plugins";
import { openGpuApp } from "./gpu-app";

// Before openApp, which wraps GPUAdapter's requestDevice to record the floor request.
await setupGlobals();

async function openApp(): Promise<{
    app: Awaited<ReturnType<typeof createApp>>;
    deviceRequest: GPUDeviceDescriptor | undefined;
}> {
    let deviceRequest: GPUDeviceDescriptor | undefined;
    const requestDevice = GPUAdapter.prototype.requestDevice;
    GPUAdapter.prototype.requestDevice = async function (descriptor) {
        deviceRequest = descriptor;
        return requestDevice.call(this, descriptor);
    };
    try {
        return { app: await openGpuApp({ plugins: APP_PLUGINS }), deviceRequest };
    } finally {
        GPUAdapter.prototype.requestDevice = requestDevice;
    }
}

type Goal = { name: string; z: number };
const GOALS: readonly Goal[] = [
    { name: "step-1", z: 3 },
    { name: "step-2", z: 0 },
    { name: "step-3", z: -3 },
    { name: "lift", z: -6.5 },
    { name: "tower-1", z: -10 },
    { name: "tower-2", z: -12.3 },
    { name: "perch", z: -14.5 },
];
// Minecraft walk speed reaches the perch near tick 500; the route allows 40% more.
const MAX_TICKS = 700;

function goalEntity(world: Awaited<ReturnType<typeof createApp>>["world"], goal: Goal): number {
    const positions = world.storage(Body).position;
    for (const eid of world.query([Body])) {
        if (Math.abs(positions.z.get(eid) - goal.z) < 0.0001) return eid;
    }
    throw new Error(`goal "${goal.name}" failed at tick 0: no authored route surface`);
}

function goalReached(
    world: Awaited<ReturnType<typeof createApp>>["world"],
    player: number,
    target: number,
): boolean {
    if (world.storage(Character).groundState.get(player) !== GroundState.OnGround) return false;
    const placement = world.storage(GlobalTransform).translation;
    const shape = world.storage(Shape);
    const radius = shape.radius.get(player);
    const dx = Math.abs(placement.x.get(player) - placement.x.get(target));
    const dz = Math.abs(placement.z.get(player) - placement.z.get(target));
    const top = placement.y.get(target) + shape.scale.y.get(target) / 2;
    return (
        dx <= shape.scale.x.get(target) / 2 + radius &&
        dz <= shape.scale.z.get(target) / 2 + radius &&
        placement.y.get(player) >= top + 0.6
    );
}

test("tick-addressed PlayerInput clears the route goals and captures the resulting frame", async () => {
    const { app, deviceRequest } = await openApp();
    try {
        const { world } = app;
        const route = world.resource(Route).entities;
        if (!route) throw new Error('goal "spawn" failed at tick 0: Demo did not create the route');
        const player = route.player;
        const camera = route.eye;
        if (!world.has(camera, Camera))
            throw new Error('goal "camera" failed at tick 0: route eye is not a camera');
        const goals = GOALS.map((goal) => ({ ...goal, eid: goalEntity(world, goal) }));
        if (!world.has(player, PlayerInput)) world.add(player, PlayerInput);
        const input = world.storage(PlayerInput);
        const frames = world.storage(GlobalTransform).translation;
        const liftBase = world.storage(Body).position.y.get(route.lift);
        const reached: { name: string; tick: number }[] = [];
        let goalIndex = 0;
        let tick = 0;

        attachTexture(world, camera, { width: 320, height: 180 });
        for (tick = 1; tick <= MAX_TICKS && goalIndex < goals.length; tick++) {
            // On the lift, wait for it to rise before stepping off; elsewhere walk forward.
            const waiting =
                goals[goalIndex]!.name === "lift" &&
                goalReached(world, player, goals[goalIndex]!.eid) &&
                frames.y.get(goals[goalIndex]!.eid) - liftBase <= 0.5;
            input.move.set(player, 0, waiting ? 0 : 1);
            if (tick % 80 === 1) input.jumpPresses.set(player, input.jumpPresses.get(player) + 1);
            world.tick();
            const goal = goals[goalIndex]!;
            if (
                goalReached(world, player, goal.eid) &&
                (goal.name !== "lift" || frames.y.get(goal.eid) - liftBase > 0.5)
            ) {
                reached.push({ name: goal.name, tick });
                goalIndex++;
            }
        }
        if (goalIndex < goals.length) {
            const goal = goals[goalIndex]!;
            throw new Error(
                `goal "${goal.name}" failed at tick ${MAX_TICKS}: player=(${frames.x.get(player).toFixed(3)}, ${frames.y.get(player).toFixed(3)}, ${frames.z.get(player).toFixed(3)}) did not reach its route surface`,
            );
        }

        for (const goal of reached)
            console.info(`first-person goal ${goal.name}: tick ${goal.tick}`);
        const { adapter, device } = world.gpu;
        if (!device)
            throw new Error(`goal "final-frame" failed at tick ${tick - 1}: no rendering device`);
        console.info(`first-person ascent adapter: ${adapter.class} (${adapter.identity})`);
        const floorFeatures = ["indirect-first-instance", "rg11b10ufloat-renderable"];
        const grantedFeatures = [...device.features]
            .filter((feature) => feature !== "core-features-and-limits")
            .sort();
        if (
            grantedFeatures.join() !== floorFeatures.join() ||
            device.limits.maxStorageBuffersPerShaderStage !== 8
        )
            throw new Error(
                `goal "rendering-floor" failed at tick ${tick - 1}: built-in floor was not granted`,
            );
        if (
            deviceRequest?.requiredFeatures?.toSorted().join() !== floorFeatures.join() ||
            Object.hasOwn(deviceRequest?.requiredLimits ?? {}, "maxStorageBuffersPerShaderStage")
        )
            throw new Error(
                `goal "rendering-floor" failed at tick ${tick - 1}: request exceeded the declared features or storage-buffer floor`,
            );

        // Exact ticks drive simulation only; one zero-delta frame updates the follow camera and renders the goal state.
        world.step(0);
        let first: Capture;
        let second: Capture;
        try {
            first = await captureTexture(world, camera);
            second = await captureTexture(world, camera);
        } catch (cause) {
            throw new Error(`goal "final-frame" failed at tick ${tick - 1}: capture failed`, {
                cause,
            });
        }
        if (first.width !== 320 || first.height !== 180 || first.rgba.length !== 320 * 180 * 4)
            throw new Error(
                `goal "final-frame" failed at tick ${tick - 1}: capture dimensions were invalid`,
            );
        const clear = first.rgba.subarray(0, 4);
        if (!first.rgba.some((value, index) => value !== clear[index % 4]))
            throw new Error(
                `goal "final-frame" failed at tick ${tick - 1}: capture contained only clear color`,
            );
        if (!second.rgba.every((value, index) => value === first.rgba[index]))
            throw new Error(
                `goal "stable-frame" failed at tick ${tick - 1}: repeated read differed`,
            );
    } finally {
        app.dispose();
    }
});
