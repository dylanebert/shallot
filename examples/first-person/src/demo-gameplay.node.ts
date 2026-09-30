import { setDefaultTimeout, test } from "bun:test";

import { CEILING } from "../../../scripts/test-tiers";

setDefaultTimeout(CEILING.node);

import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import {
    Body,
    build,
    CharacterPlugin,
    Devices,
    InputPlugin,
    PhysicsPlugin,
    PlayerPlugin,
    readBody,
    type State,
    Time,
} from "@dylanebert/shallot";
import { Demo } from "./demo";

const peerModule = "bun-webgpu";
const { setupGlobals } = (await import(peerModule)) as { setupGlobals(): Promise<void> };
await setupGlobals();

const SCENE = resolve(import.meta.dir, "../public/scenes/first-person.scene");

async function ascent() {
    // The CPU rows use the actual manifest-selected scene and local Demo plugin. Player and rendering
    // remain in the exact-project browser row because those are device-bound defaults.
    return build({
        defaults: false,
        plugins: [PhysicsPlugin, CharacterPlugin, InputPlugin, Demo],
        scene: SCENE,
    });
}

type Ascent = Awaited<ReturnType<typeof ascent>>;

function entity(app: Ascent, id: string): number {
    for (const eid of app.state.entities()) if (app.state.identity.id(eid) === id) return eid;
    throw new Error(`actual ascent scene has no ${id} entity`);
}

function step(app: Ascent, ticks: number): void {
    for (let i = 0; i < ticks; i++) app.state.step(Time.FIXED_DT);
}

function horizontalSpeed(velocity: readonly [number, number, number]): number {
    return Math.hypot(velocity[0], velocity[2]);
}

function placeRiderOnActualLift(state: State, player: number, lift: number): void {
    const body = state.of(Body);
    const liftX = body.pos.x.get(lift);
    const liftY = body.pos.y.get(lift);
    const liftZ = body.pos.z.get(lift);
    const riderBottomOffset = body.halfExtents.y.get(player) + body.halfExtents.w.get(player);
    body.pos.x.set(player, liftX);
    body.pos.y.set(player, liftY + body.halfExtents.y.get(lift) + riderBottomOffset);
    body.pos.z.set(player, liftZ);
}

function tangentGap(state: State, player: number, lift: number): number {
    const body = state.of(Body);
    const capsuleBottom =
        body.pos.y.get(player) - body.halfExtents.y.get(player) - body.halfExtents.w.get(player);
    const liftTop = body.pos.y.get(lift) + body.halfExtents.y.get(lift);
    return capsuleBottom - liftTop;
}

function extent(state: State, eid: number, axis: "x" | "z", radius = 0): readonly [number, number] {
    const body = state.of(Body);
    const center = axis === "x" ? body.pos.x.get(eid) : body.pos.z.get(eid);
    const half =
        (axis === "x" ? body.halfExtents.x.get(eid) : body.halfExtents.z.get(eid)) + radius;
    return [center - half, center + half];
}

function authoredStepRise(app: Ascent, player: number, lift: number): number {
    const heights = [...app.state.query([Body])]
        .filter((eid) => eid !== player && eid !== lift && app.state.of(Body).mass.get(eid) <= 0)
        .map((eid) => app.state.of(Body).pos.y.get(eid))
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
            app.state.of(Body).pos.y.get(ground) + app.state.of(Body).halfExtents.y.get(ground);
        const playerBottom =
            app.state.of(Body).pos.y.get(player) -
            app.state.of(Body).halfExtents.y.get(player) -
            app.state.of(Body).halfExtents.w.get(player);
        if (Math.abs(playerBottom - groundTop) > 0.0001)
            throw new Error(
                `player was not tangent to ground: bottom=${playerBottom} top=${groundTop}`,
            );
        const spawnGap =
            app.state.of(Body).pos.z.get(player) -
            app.state.of(Body).halfExtents.w.get(player) -
            (app.state.of(Body).pos.z.get(step1) + app.state.of(Body).halfExtents.z.get(step1));
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
                    routeEntity === player ? app.state.of(Body).halfExtents.w.get(player) : 0;
                const [min, max] = extent(app.state, routeEntity, axis, radius);
                const [groundMin, groundMax] = extent(app.state, ground, axis);
                if (min < groundMin || max > groundMax)
                    throw new Error(
                        `ground did not contain ${axis} route footprint ${min}..${max}`,
                    );
            }
        }
        const liftTop =
            app.state.of(Body).pos.y.get(lift) + app.state.of(Body).halfExtents.y.get(lift);
        const finalStepTop =
            app.state.of(Body).pos.y.get(step3) + app.state.of(Body).halfExtents.y.get(step3);
        if (Math.abs(liftTop - finalStepTop) > 0.0001)
            throw new Error(`lift lower stop missed final step: ${liftTop} vs ${finalStepTop}`);
        const liftBottom =
            app.state.of(Body).pos.y.get(lift) - app.state.of(Body).halfExtents.y.get(lift);
        if (!(liftBottom > groundTop))
            throw new Error(
                `lift lower stop entered ground: bottom=${liftBottom} top=${groundTop}`,
            );
        const upperLiftNear =
            app.state.of(Body).pos.z.get(lift) - app.state.of(Body).halfExtents.z.get(lift);
        const towerNear =
            app.state.of(Body).pos.z.get(tower1) + app.state.of(Body).halfExtents.z.get(tower1);
        const towerGap = upperLiftNear - towerNear;
        if (!(towerGap > 0 && towerGap < 1))
            throw new Error(`lift upper stop was not adjacent to tower: gap=${towerGap}`);
        const scene = readFileSync(SCENE, "utf8");
        const playerBlock = scene.match(/id="player"[\s\S]*?\/>/)?.[0] ?? "";
        if (/\b(speed|sprint|sensitivity|yaw|pitch)\s*:/.test(playerBlock))
            throw new Error("first-person player entity authors movement/look tuning");
        // The eye is authored where Player would pose it at spawn: the capsule centre raised by
        // the default eye height, since the player entity authors no tuning of its own.
        const eyeHeight = (
            PlayerPlugin.traits?.Player?.defaults?.(app.state) as { eyeHeight: number } | undefined
        )?.eyeHeight;
        if (eyeHeight === undefined) throw new Error("Player declares no default eyeHeight");
        const eye = scene
            .match(/id="eye"[^>]*pos: (\S+) (\S+) ([^;"]+)/)
            ?.slice(1)
            .map(Number);
        const spawn = [
            app.state.of(Body).pos.x.get(player),
            app.state.of(Body).pos.y.get(player) + eyeHeight,
            app.state.of(Body).pos.z.get(player),
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
        placeRiderOnActualLift(app.state, player, lift);
        const stepRise = authoredStepRise(app, player, lift);
        const initialGap = tangentGap(app.state, player, lift);
        if (initialGap < 0 || initialGap > 0.0001)
            throw new Error(
                `invalid actual lift premise: capsule/lift gap was ${initialGap.toFixed(4)}m`,
            );
        step(app, 2);
        const before = readBody(app.state, player);
        const liftBefore = readBody(app.state, lift);
        if (!before || !liftBefore) throw new Error("actual ascent bodies never became live");
        step(app, 100);
        const after = readBody(app.state, player);
        const liftAfter = readBody(app.state, lift);
        if (!after || !liftAfter) throw new Error("actual ascent bodies disappeared");
        const liftRise = liftAfter.pos[1] - liftBefore.pos[1];
        const riderRise = after.pos[1] - before.pos[1];
        if (liftRise <= stepRise || riderRise <= stepRise)
            throw new Error(
                `lift/rider rise ${liftRise.toFixed(3)}m/${riderRise.toFixed(3)}m did not clear authored step ${stepRise.toFixed(3)}m`,
            );
        if (Math.abs(riderRise - liftRise) > stepRise)
            throw new Error(
                `rider lost actual lift carry: lift ${liftRise.toFixed(3)}m, rider ${riderRise.toFixed(3)}m`,
            );
        if (app.state.resource(Devices).keys.held.size !== 0)
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
        placeRiderOnActualLift(app.state, player, lift);
        step(app, 2);
        const before = readBody(app.state, lift);
        if (!before) throw new Error("actual lift never became live");
        step(app, 50);
        const after = readBody(app.state, player);
        const liftAfter = readBody(app.state, lift);
        if (!after || !liftAfter) throw new Error("actual ascent bodies disappeared");
        if (liftAfter.pos[1] <= before.pos[1])
            throw new Error("actual lift did not move upward during sample");
        const speed = horizontalSpeed(after.vel);
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
            app.state.of(Body).pos.x.get(lift),
            app.state.of(Body).pos.y.get(lift),
            app.state.of(Body).pos.z.get(lift),
        ] as const;
        const tower = entity(app, "tower-3");
        const ceiling =
            app.state.of(Body).pos.y.get(tower) +
            app.state.of(Body).halfExtents.y.get(tower) -
            app.state.of(Body).halfExtents.y.get(lift);
        const stepRise = authoredStepRise(app, entity(app, "player"), lift);
        let previous = base[1];
        let rising = true;
        let turns = 0;
        for (let tick = 1; tick <= 600; tick++) {
            app.state.step(Time.FIXED_DT);
            const pose = readBody(app.state, lift);
            if (!pose) throw new Error(`actual lift disappeared at tick ${tick}`);
            const y = pose.pos[1];
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
                Math.abs(pose.pos[0] - base[0]) > 0.002 ||
                Math.abs(pose.pos[2] - base[2]) > 0.002 ||
                Math.abs(pose.vel[0]) > 0.002 ||
                Math.abs(pose.vel[2]) > 0.002
            )
                throw new Error(
                    `lift left its vertical band at tick ${tick}: pos=${pose.pos} vel=${pose.vel} base=${base} ceiling=${ceiling}`,
                );
        }
        if (turns < 3) throw new Error(`lift turned ${turns} times in 600 ticks`);
    } finally {
        app.dispose();
    }
});
