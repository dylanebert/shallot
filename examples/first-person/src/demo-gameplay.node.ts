import { setDefaultTimeout, test } from "bun:test";

import { CEILING } from "../../../scripts/test-tiers";

setDefaultTimeout(CEILING.node);

import {
    Body,
    CharacterPlugin,
    createApp,
    Devices,
    InputPlugin,
    PhysicsPlugin,
    Player,
    PlayerPlugin,
    readBody,
    Time,
    Transform,
    type World,
} from "@dylanebert/shallot";
import { Demo, Route } from "./demo";

const peerModule = "bun-webgpu";
const { setupGlobals } = (await import(peerModule)) as { setupGlobals(): Promise<void> };
await setupGlobals();

async function ascent() {
    // The CPU rows use the actual code route and local Demo plugin. Player and rendering
    // remain in the exact-project browser row because those are device-bound defaults.
    return createApp({
        defaults: false,
        plugins: [PhysicsPlugin, CharacterPlugin, InputPlugin, Demo],
    });
}

type Ascent = Awaited<ReturnType<typeof ascent>>;

function entity(app: Ascent, id: string): number {
    const held = app.world.resource(Route).entities;
    if (held && (id === "player" || id === "eye" || id === "lift")) return held[id];
    const z = {
        ground: -5,
        "step-1": 3,
        "step-2": 0,
        "step-3": -3,
        "tower-1": -10,
        "tower-2": -12.3,
        "tower-3": -14.5,
    }[id];
    for (const eid of app.world.query([Body])) {
        if (z !== undefined && Math.abs(app.world.storage(Body).position.z.get(eid) - z) < 0.0001)
            return eid;
    }
    throw new Error(`actual ascent route has no ${id} entity`);
}

function step(app: Ascent, ticks: number): void {
    for (let i = 0; i < ticks; i++) app.world.step(Time.FIXED_DT);
}

function horizontalSpeed(velocity: readonly [number, number, number]): number {
    return Math.hypot(velocity[0], velocity[2]);
}

function placeRiderOnActualLift(world: World, player: number, lift: number): void {
    const body = world.storage(Body);
    const liftX = body.position.x.get(lift);
    const liftY = body.position.y.get(lift);
    const liftZ = body.position.z.get(lift);
    const riderBottomOffset = body.halfExtents.y.get(player) + body.halfExtents.w.get(player);
    body.position.x.set(player, liftX);
    body.position.y.set(player, liftY + body.halfExtents.y.get(lift) + riderBottomOffset);
    body.position.z.set(player, liftZ);
}

function tangentGap(world: World, player: number, lift: number): number {
    const body = world.storage(Body);
    const capsuleBottom =
        body.position.y.get(player) -
        body.halfExtents.y.get(player) -
        body.halfExtents.w.get(player);
    const liftTop = body.position.y.get(lift) + body.halfExtents.y.get(lift);
    return capsuleBottom - liftTop;
}

function extent(world: World, eid: number, axis: "x" | "z", radius = 0): readonly [number, number] {
    const body = world.storage(Body);
    const center = axis === "x" ? body.position.x.get(eid) : body.position.z.get(eid);
    const half =
        (axis === "x" ? body.halfExtents.x.get(eid) : body.halfExtents.z.get(eid)) + radius;
    return [center - half, center + half];
}

function authoredStepRise(app: Ascent, player: number, lift: number): number {
    const heights = [...app.world.query([Body])]
        .filter(
            (eid) => eid !== player && eid !== lift && app.world.storage(Body).mass.get(eid) <= 0,
        )
        .map((eid) => app.world.storage(Body).position.y.get(eid))
        .filter((y) => y > 0 && y < 1.6)
        .sort((a, b) => a - b);
    if (heights.length < 2)
        throw new Error(`actual ascent scene has only ${heights.length} authored step heights`);
    return heights[heights.length - 1] - heights[0];
}

test("the actual first-person scene gives the player a tangent spawn, a contained route, a clear lift, and an adjacent upper tower stop", async () => {
    const app = await ascent();
    try {
        const player = entity(app, "player");
        const ground = entity(app, "ground");
        const step1 = entity(app, "step-1");
        const step3 = entity(app, "step-3");
        const lift = entity(app, "lift");
        const tower1 = entity(app, "tower-1");
        const groundTop =
            app.world.storage(Body).position.y.get(ground) +
            app.world.storage(Body).halfExtents.y.get(ground);
        const playerBottom =
            app.world.storage(Body).position.y.get(player) -
            app.world.storage(Body).halfExtents.y.get(player) -
            app.world.storage(Body).halfExtents.w.get(player);
        if (Math.abs(playerBottom - groundTop) > 0.0001)
            throw new Error(
                `player was not tangent to ground: bottom=${playerBottom} top=${groundTop}`,
            );
        const spawnGap =
            app.world.storage(Body).position.z.get(player) -
            app.world.storage(Body).halfExtents.w.get(player) -
            (app.world.storage(Body).position.z.get(step1) +
                app.world.storage(Body).halfExtents.z.get(step1));
        if (!(spawnGap > 0)) throw new Error(`spawn-to-first-step gap was ${spawnGap}`);
        const route = [
            player,
            step1,
            entity(app, "step-2"),
            step3,
            lift,
            tower1,
            entity(app, "tower-2"),
            entity(app, "tower-3"),
        ];
        for (const routeEntity of route) {
            for (const axis of ["x", "z"] as const) {
                const radius =
                    routeEntity === player ? app.world.storage(Body).halfExtents.w.get(player) : 0;
                const [min, max] = extent(app.world, routeEntity, axis, radius);
                const [groundMin, groundMax] = extent(app.world, ground, axis);
                if (min < groundMin || max > groundMax)
                    throw new Error(
                        `ground did not contain ${axis} route footprint ${min}..${max}`,
                    );
            }
        }
        const liftTop =
            app.world.storage(Body).position.y.get(lift) +
            app.world.storage(Body).halfExtents.y.get(lift);
        const finalStepTop =
            app.world.storage(Body).position.y.get(step3) +
            app.world.storage(Body).halfExtents.y.get(step3);
        if (Math.abs(liftTop - finalStepTop) > 0.0001)
            throw new Error(`lift lower stop missed final step: ${liftTop} vs ${finalStepTop}`);
        const liftBottom =
            app.world.storage(Body).position.y.get(lift) -
            app.world.storage(Body).halfExtents.y.get(lift);
        if (!(liftBottom > groundTop))
            throw new Error(
                `lift lower stop entered ground: bottom=${liftBottom} top=${groundTop}`,
            );
        const upperLiftNear =
            app.world.storage(Body).position.z.get(lift) -
            app.world.storage(Body).halfExtents.z.get(lift);
        const towerNear =
            app.world.storage(Body).position.z.get(tower1) +
            app.world.storage(Body).halfExtents.z.get(tower1);
        const towerGap = upperLiftNear - towerNear;
        if (!(towerGap > 0 && towerGap < 1))
            throw new Error(`lift upper stop was not adjacent to tower: gap=${towerGap}`);
        const defaults = app.world.create();
        app.world.add(defaults, Player);
        const tuning = app.world.storage(Player);
        for (const field of ["speed", "sprint", "sensitivity", "yaw", "pitch"] as const) {
            if (tuning[field].get(player) !== tuning[field].get(defaults))
                throw new Error(`first-person player overrides default ${field}`);
        }
        app.world.destroy(defaults);
        // The eye is authored where Player would pose it at spawn: the capsule centre raised by
        // the default eye height, since the player entity authors no tuning of its own.
        const eyeHeight = (
            PlayerPlugin.traits?.Player?.defaults?.(app.world) as { eyeHeight: number } | undefined
        )?.eyeHeight;
        if (eyeHeight === undefined) throw new Error("Player declares no default eyeHeight");
        const eyeTransform = app.world.storage(Transform).translation;
        const eyeEid = entity(app, "eye");
        const eye = [
            eyeTransform.x.get(eyeEid),
            eyeTransform.y.get(eyeEid),
            eyeTransform.z.get(eyeEid),
        ];
        const spawn = [
            app.world.storage(Body).position.x.get(player),
            app.world.storage(Body).position.y.get(player) + eyeHeight,
            app.world.storage(Body).position.z.get(player),
        ];
        if (!eye || eye.some((value, i) => Math.abs(value - spawn[i]) > 0.0001))
            throw new Error(
                `first-person eye ${eye} was not authored at the default-height spawn ${spawn}`,
            );
    } finally {
        app.dispose();
    }
});

test("a Character standing on the actual recipe lift rises through its public kinematic trajectory by more than one authored step", async () => {
    const app = await ascent();
    try {
        const player = entity(app, "player");
        const lift = entity(app, "lift");
        placeRiderOnActualLift(app.world, player, lift);
        const stepRise = authoredStepRise(app, player, lift);
        const initialGap = tangentGap(app.world, player, lift);
        if (initialGap < 0 || initialGap > 0.0001)
            throw new Error(
                `invalid actual lift premise: capsule/lift gap was ${initialGap.toFixed(4)}m`,
            );
        step(app, 2);
        const before = readBody(app.world, player);
        const liftBefore = readBody(app.world, lift);
        if (!before || !liftBefore) throw new Error("actual ascent bodies never became live");
        step(app, 100);
        const after = readBody(app.world, player);
        const liftAfter = readBody(app.world, lift);
        if (!after || !liftAfter) throw new Error("actual ascent bodies disappeared");
        const liftRise = liftAfter.position[1] - liftBefore.position[1];
        const riderRise = after.position[1] - before.position[1];
        if (liftRise <= stepRise || riderRise <= stepRise)
            throw new Error(
                `lift/rider rise ${liftRise.toFixed(3)}m/${riderRise.toFixed(3)}m did not clear authored step ${stepRise.toFixed(3)}m`,
            );
        if (Math.abs(riderRise - liftRise) > stepRise)
            throw new Error(
                `rider lost actual lift carry: lift ${liftRise.toFixed(3)}m, rider ${riderRise.toFixed(3)}m`,
            );
        if (app.world.resource(Devices).keys.held.size !== 0)
            throw new Error("actual lift evidence received unexpected input");
    } finally {
        app.dispose();
    }
});

test("the actual moving lift carries the Character vertically without delivering horizontal velocity", async () => {
    const app = await ascent();
    try {
        const player = entity(app, "player");
        const lift = entity(app, "lift");
        placeRiderOnActualLift(app.world, player, lift);
        step(app, 2);
        const before = readBody(app.world, lift);
        if (!before) throw new Error("actual lift never became live");
        step(app, 50);
        const after = readBody(app.world, player);
        const liftAfter = readBody(app.world, lift);
        if (!after || !liftAfter) throw new Error("actual ascent bodies disappeared");
        if (liftAfter.position[1] <= before.position[1])
            throw new Error("actual lift did not move upward during sample");
        const speed = horizontalSpeed(after.linearVelocity);
        if (speed > 0.001)
            throw new Error(`actual lift delivered ${speed.toFixed(4)}m/s horizontal velocity`);
    } finally {
        app.dispose();
    }
});

test("the actual lift rises monotonically from its authored base, turns repeatedly, stays between that base and the tower top, and keeps fixed X/Z, so live-pose accumulation reds independently", async () => {
    const app = await ascent();
    try {
        const lift = entity(app, "lift");
        const base = [
            app.world.storage(Body).position.x.get(lift),
            app.world.storage(Body).position.y.get(lift),
            app.world.storage(Body).position.z.get(lift),
        ] as const;
        const tower = entity(app, "tower-3");
        const ceiling =
            app.world.storage(Body).position.y.get(tower) +
            app.world.storage(Body).halfExtents.y.get(tower) -
            app.world.storage(Body).halfExtents.y.get(lift);
        const stepRise = authoredStepRise(app, entity(app, "player"), lift);
        let previous = base[1];
        let rising = true;
        let turns = 0;
        for (let tick = 1; tick <= 600; tick++) {
            app.world.step(Time.FIXED_DT);
            const pose = readBody(app.world, lift);
            if (!pose) throw new Error(`actual lift disappeared at tick ${tick}`);
            const y = pose.position[1];
            const falling = y < previous - 1e-6;
            if ((rising && falling) || (!rising && y > previous + 1e-6)) {
                if (turns === 0 && !(previous > base[1] + stepRise))
                    throw new Error(
                        `lift turned at ${previous} before a monotone rise above one authored step ${stepRise}`,
                    );
                rising = !rising;
                turns++;
            }
            previous = y;
            if (
                y < base[1] - 0.002 ||
                y > ceiling ||
                Math.abs(pose.position[0] - base[0]) > 0.002 ||
                Math.abs(pose.position[2] - base[2]) > 0.002 ||
                Math.abs(pose.linearVelocity[0]) > 0.002 ||
                Math.abs(pose.linearVelocity[2]) > 0.002
            )
                throw new Error(
                    `lift left its vertical band at tick ${tick}: pos=${pose.position} vel=${pose.linearVelocity} base=${base} ceiling=${ceiling}`,
                );
        }
        if (turns < 3) throw new Error(`lift turned ${turns} times in 600 ticks`);
    } finally {
        app.dispose();
    }
});
