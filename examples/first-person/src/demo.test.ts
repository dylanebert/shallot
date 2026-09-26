import { expect } from "bun:test";
import { execFileSync } from "node:child_process";
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import {
    Body,
    build,
    CharacterPlugin,
    devices,
    InputPlugin,
    PhysicsPlugin,
    PlayerPlugin,
    readBody,
    Time,
} from "@dylanebert/shallot";
import {
    allocationFailure,
    sampleAllocation,
    windowBytes,
} from "@dylanebert/shallot/harness/allocation";
import { check } from "@dylanebert/shallot/harness/check";
import { Demo } from "./demo";

const SCENE = resolve(import.meta.dir, "../public/scenes/first-person.scene");

check(
    "first-person ignore rules keep outputs out and project files visible",
    {
        claim: "Git ignores first-person build outputs while reporting source and lockfiles as untracked",
    },
    () => {
        const root = mkdtempSync(join(tmpdir(), "shallot-example-ignore-"));
        try {
            writeFileSync(
                join(root, ".gitignore"),
                readFileSync(resolve(import.meta.dir, "../.gitignore")),
            );
            for (const dir of ["src", "dist", "build", ".artifacts"])
                mkdirSync(join(root, dir), { recursive: true });
            for (const file of [
                "src/demo.ts",
                "bun.lock",
                "dist/index.js",
                "build/app.js",
                ".artifacts/run",
            ])
                writeFileSync(join(root, file), "fixture\n");
            execFileSync("git", ["-C", root, "init", "-q"]);

            const ignored = execFileSync(
                "git",
                ["-C", root, "check-ignore", "dist/index.js", "build/app.js", ".artifacts/run"],
                { encoding: "utf8" },
            );
            const status = execFileSync(
                "git",
                ["-C", root, "status", "--short", "--untracked-files=all"],
                { encoding: "utf8" },
            );
            expect(ignored).toContain("dist/index.js");
            expect(ignored).toContain("build/app.js");
            expect(ignored).toContain(".artifacts/run");
            expect(status).toContain("?? bun.lock");
            expect(status).toContain("?? src/demo.ts");
            expect(status).not.toContain("dist/");
            expect(status).not.toContain("build/");
            expect(status).not.toContain(".artifacts/");
        } finally {
            rmSync(root, { recursive: true, force: true });
        }
    },
);

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

function placeRiderOnActualLift(player: number, lift: number): void {
    const liftX = Body.pos.x.get(lift);
    const liftY = Body.pos.y.get(lift);
    const liftZ = Body.pos.z.get(lift);
    const riderBottomOffset = Body.halfExtents.y.get(player) + Body.halfExtents.w.get(player);
    Body.pos.x.set(player, liftX);
    Body.pos.y.set(player, liftY + Body.halfExtents.y.get(lift) + riderBottomOffset);
    Body.pos.z.set(player, liftZ);
}

function tangentGap(player: number, lift: number): number {
    const capsuleBottom =
        Body.pos.y.get(player) - Body.halfExtents.y.get(player) - Body.halfExtents.w.get(player);
    const liftTop = Body.pos.y.get(lift) + Body.halfExtents.y.get(lift);
    return capsuleBottom - liftTop;
}

function extent(eid: number, axis: "x" | "z", radius = 0): readonly [number, number] {
    const center = axis === "x" ? Body.pos.x.get(eid) : Body.pos.z.get(eid);
    const half =
        (axis === "x" ? Body.halfExtents.x.get(eid) : Body.halfExtents.z.get(eid)) + radius;
    return [center - half, center + half];
}

function authoredStepRise(app: Ascent, player: number, lift: number): number {
    const heights = [...app.state.query([Body])]
        .filter((eid) => eid !== player && eid !== lift && Body.mass.get(eid) <= 0)
        .map((eid) => Body.pos.y.get(eid))
        .filter((y) => y > 0 && y < 1.6)
        .sort((a, b) => a - b);
    if (heights.length < 2)
        throw new Error(`actual ascent scene has only ${heights.length} authored step heights`);
    return heights[heights.length - 1] - heights[0];
}

check(
    "first-person presentation geometry remains relationally valid",
    {
        claim: "the actual first-person scene gives the player a tangent spawn, a contained route, a clear lift, and an adjacent upper tower stop",
    },
    async () => {
        const app = await ascent();
        try {
            const player = entity(app, "player");
            const ground = entity(app, "ground");
            const step1 = entity(app, "step-1");
            const step3 = entity(app, "step-3");
            const lift = entity(app, "lift");
            const tower1 = entity(app, "tower-1");
            const groundTop = Body.pos.y.get(ground) + Body.halfExtents.y.get(ground);
            const playerBottom =
                Body.pos.y.get(player) -
                Body.halfExtents.y.get(player) -
                Body.halfExtents.w.get(player);
            if (Math.abs(playerBottom - groundTop) > 0.0001)
                throw new Error(
                    `player was not tangent to ground: bottom=${playerBottom} top=${groundTop}`,
                );
            const spawnGap =
                Body.pos.z.get(player) -
                Body.halfExtents.w.get(player) -
                (Body.pos.z.get(step1) + Body.halfExtents.z.get(step1));
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
                    const radius = routeEntity === player ? Body.halfExtents.w.get(player) : 0;
                    const [min, max] = extent(routeEntity, axis, radius);
                    const [groundMin, groundMax] = extent(ground, axis);
                    if (min < groundMin || max > groundMax)
                        throw new Error(
                            `ground did not contain ${axis} route footprint ${min}..${max}`,
                        );
                }
            }
            const liftTop = Body.pos.y.get(lift) + Body.halfExtents.y.get(lift);
            const finalStepTop = Body.pos.y.get(step3) + Body.halfExtents.y.get(step3);
            if (Math.abs(liftTop - finalStepTop) > 0.0001)
                throw new Error(`lift lower stop missed final step: ${liftTop} vs ${finalStepTop}`);
            const liftBottom = Body.pos.y.get(lift) - Body.halfExtents.y.get(lift);
            if (!(liftBottom > groundTop))
                throw new Error(
                    `lift lower stop entered ground: bottom=${liftBottom} top=${groundTop}`,
                );
            const upperLiftNear = Body.pos.z.get(lift) - Body.halfExtents.z.get(lift);
            const towerNear = Body.pos.z.get(tower1) + Body.halfExtents.z.get(tower1);
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
                PlayerPlugin.traits?.Player?.defaults?.() as { eyeHeight: number } | undefined
            )?.eyeHeight;
            if (eyeHeight === undefined) throw new Error("Player declares no default eyeHeight");
            const eye = scene
                .match(/id="eye"[^>]*pos: (\S+) (\S+) ([^;"]+)/)
                ?.slice(1)
                .map(Number);
            const spawn = [
                Body.pos.x.get(player),
                Body.pos.y.get(player) + eyeHeight,
                Body.pos.z.get(player),
            ];
            if (!eye || eye.some((value, i) => Math.abs(value - spawn[i]) > 0.0001))
                throw new Error(
                    `first-person eye ${eye} was not authored at the default-height spawn ${spawn}`,
                );
        } finally {
            app.dispose();
        }
    },
);

check(
    "first-person lift carries the actual character upward",
    {
        claim: "a Character standing on the actual recipe lift rises through its public kinematic trajectory by more than one authored step",
    },
    async () => {
        const app = await ascent();
        try {
            const player = entity(app, "player");
            const lift = entity(app, "lift");
            placeRiderOnActualLift(player, lift);
            const stepRise = authoredStepRise(app, player, lift);
            const initialGap = tangentGap(player, lift);
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
            if (devices(app.state).keys.held.size !== 0)
                throw new Error("actual lift evidence received unexpected input");
        } finally {
            app.dispose();
        }
    },
);

check(
    "first-person lift does not shove the actual character horizontally",
    {
        claim: "the actual moving lift carries the Character vertically without delivering horizontal velocity",
    },
    async () => {
        const app = await ascent();
        try {
            const player = entity(app, "player");
            const lift = entity(app, "lift");
            placeRiderOnActualLift(player, lift);
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
    },
);

check(
    "first-person warm frame allocates nothing",
    {
        claim: "a warm fixed step of the actual first-person CPU composition allocates no JavaScript heap, so no periodic scavenge follows play",
        size: "integration",
        requires: ["node"],
        subject: ["examples/first-person"],
    },
    async () => {
        // 6,000 frames: the once-per-escape refit path (`commitRefit`, the fat-AABB write, the tree enlarge)
        // is called about once a frame, so it reaches TurboFan late; at 1,200 it runs Maglev code inside
        // every window and at 2,400 it can still tier inside the first. From 3,600 all three windows agree.
        const sample = await sampleAllocation(resolve(import.meta.dir, "allocation.entry.ts"), {
            warm: 6000,
            frames: 600,
            input: readFileSync(SCENE, "utf8"),
        });
        // The entry's control literal, attributed as the windows are, proves the sampler sees subject
        // allocation; without it an empty site set proves nothing.
        const control = { label: "control", sites: sample.control };
        if (control.sites.length === 0 || windowBytes(control) <= 0)
            throw new Error(
                "inconclusive: the sampler attributed no site to the entry's control literal",
            );
        const failure = allocationFailure(sample);
        if (failure !== undefined) throw new Error(failure);
    },
);

check(
    "first-person lift oscillates within its tower",
    {
        claim: "the actual lift rises monotonically from its authored base, turns repeatedly, stays between that base and the tower top, and keeps fixed X/Z, so live-pose accumulation reds independently",
    },
    async () => {
        const app = await ascent();
        try {
            const lift = entity(app, "lift");
            const base = [
                Body.pos.x.get(lift),
                Body.pos.y.get(lift),
                Body.pos.z.get(lift),
            ] as const;
            const tower = entity(app, "tower-3");
            const ceiling =
                Body.pos.y.get(tower) +
                Body.halfExtents.y.get(tower) -
                Body.halfExtents.y.get(lift);
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
    },
);
