import {
    Body,
    build,
    CharacterPlugin,
    devices,
    InputPlugin,
    PhysicsPlugin,
    readBody,
    Time,
} from "@dylanebert/shallot";
import { check } from "@dylanebert/shallot/harness/check";
import { Demo } from "./demo";

const SCENE = `<scene>
    <a id="ground" body="pos: 0 0 -4; half-extents: 10 0.5 16; mass: 0" />
    <a id="lower-step" body="pos: 0 0.75 3; half-extents: 3 0.25 1.5; mass: 0" />
    <a id="upper-step" body="pos: 0 1.25 0; half-extents: 3 0.25 1.5; mass: 0" />
    <a id="lift" body="pos: 0 1.75 -6.5; half-extents: 3 0.25 2; mass: 0" lift />
    <a id="player" body="pos: 0 2.9 -6.5; shape: 2; half-extents: 0 0.6 0 0.3; mass: 0" character />
</scene>`;

async function ascent() {
    return build({
        defaults: false,
        plugins: [PhysicsPlugin, CharacterPlugin, InputPlugin, Demo],
        scene: SCENE,
    });
}

type Ascent = Awaited<ReturnType<typeof ascent>>;

function entity(app: Ascent, id: string): number {
    for (const eid of app.state.entities()) if (app.state.identity.id(eid) === id) return eid;
    throw new Error(`ascent scene has no ${id} entity`);
}

function step(app: Ascent, ticks: number): void {
    for (let i = 0; i < ticks; i++) app.state.step(Time.FIXED_DT);
}

function horizontalSpeed(velocity: readonly [number, number, number]): number {
    return Math.hypot(velocity[0], velocity[2]);
}

check(
    "first-person lift carries the character upward",
    {
        claim: "a Character standing on the recipe lift rises through its public kinematic trajectory by more than one authored step",
    },
    async () => {
        const app = await ascent();
        try {
            const player = entity(app, "player");
            const lift = entity(app, "lift");
            const lowerStep = entity(app, "lower-step");
            const upperStep = entity(app, "upper-step");
            const stepRise = Math.abs(Body.pos.y.get(upperStep) - Body.pos.y.get(lowerStep));
            const capsuleBottom =
                Body.pos.y.get(player) -
                Body.halfExtents.y.get(player) -
                Body.halfExtents.w.get(player);
            const liftTop = Body.pos.y.get(lift) + Body.halfExtents.y.get(lift);
            const initialGap = capsuleBottom - liftTop;
            if (initialGap < 0 || initialGap > 0.0001)
                throw new Error(
                    `invalid lift premise: capsule/lift vertical gap was ${initialGap.toFixed(4)}m, expected tangent`,
                );
            step(app, 2);
            const before = readBody(app.state, player);
            const liftBefore = readBody(app.state, lift);
            if (!before || !liftBefore) throw new Error("ascent bodies never became live");
            step(app, 60);
            const after = readBody(app.state, player);
            const liftAfter = readBody(app.state, lift);
            if (!after || !liftAfter) throw new Error("ascent bodies disappeared");
            const liftRise = liftAfter.pos[1] - liftBefore.pos[1];
            const riderRise = after.pos[1] - before.pos[1];
            if (liftRise <= stepRise || riderRise <= stepRise)
                throw new Error(
                    `lift/rider rise ${liftRise.toFixed(3)}m/${riderRise.toFixed(3)}m did not clear the authored ${stepRise.toFixed(3)}m step`,
                );
            if (Math.abs(riderRise - liftRise) > stepRise)
                throw new Error(
                    `rider lost lift carry: lift ${liftRise.toFixed(3)}m, rider ${riderRise.toFixed(3)}m`,
                );
            if (devices(app.state).keys.held.size !== 0)
                throw new Error("lift evidence received unexpected input");
        } finally {
            app.dispose();
        }
    },
);

check(
    "first-person lift does not shove the character horizontally",
    {
        claim: "the moving lift carries the Character vertically without delivering horizontal velocity",
    },
    async () => {
        const app = await ascent();
        try {
            const player = entity(app, "player");
            const lift = entity(app, "lift");
            step(app, 2);
            const before = readBody(app.state, lift);
            if (!before) throw new Error("lift never became live");
            step(app, 50);
            const after = readBody(app.state, player);
            const liftAfter = readBody(app.state, lift);
            if (!after || !liftAfter) throw new Error("ascent bodies disappeared");
            if (liftAfter.pos[1] <= before.pos[1])
                throw new Error("lift did not move upward during the sample");
            const speed = horizontalSpeed(after.vel);
            if (speed > 0.001)
                throw new Error(`lift delivered ${speed.toFixed(4)}m/s horizontal velocity`);
        } finally {
            app.dispose();
        }
    },
);
