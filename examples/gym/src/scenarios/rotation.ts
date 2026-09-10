import {
    AmbientLight,
    Body,
    Camera,
    CameraMode,
    Color,
    DirectionalLight,
    GlazePlugin,
    InputPlugin,
    Orbit,
    OrbitPlugin,
    Part,
    PartPlugin,
    Physics,
    PhysicsPlugin,
    type Plugin,
    RenderPlugin,
    run,
    Sear,
    SearPlugin,
    ShapeKind,
    SlabPlugin,
    type State,
    type System,
    Transform,
    TransformsPlugin,
} from "@dylanebert/shallot";
import { ProfilePlugin } from "@dylanebert/shallot/extras";
import { StepSystem } from "@dylanebert/shallot/physics/core";
import { type Check, frames, register, type Scenario } from "../gym";

// rotation — free rigid-body angular dynamics in a zero-gravity world (`Physics.world.setGravity(0)`, the
// escape hatch — the substrate world runs at −10). Two effects with crisp invariants in one scene: the
// Dzhanibekov flip (a flat "book" spun about its intermediate axis of inertia physicss chaotically while the
// major- and minor-axis spins stay stable) and a parallel joint (it locks a panel's orientation while leaving
// it free to translate, so a torqued held panel barely turns while its unconstrained twin physicss). The
// bodies are substrate `Body` entities; only the parallel joint + the zero-g override reach through the hatch.

const BOOK_HALF: [number, number, number] = [0.35, 0.08, 0.5]; // I_z < I_x < I_y → x is the intermediate axis
const SPIN = 5;
const TICKS = 450; // enough fixed ticks for the intermediate-axis flip to develop

let bookXEid = -1; // spun about the intermediate axis — physicss
let bookYEid = -1; // spun about the max-inertia axis — stable
let bookZEid = -1; // spun about the min-inertia axis — stable
let refEid = -1;
let heldEid = -1;
let freeEid = -1;
let seeded = false;
let bookXMaxOff = 0;
let bookYMaxOff = 0;

function body(
    state: State,
    x: number,
    y: number,
    half: [number, number, number],
    mass: number,
    color: [number, number, number],
): number {
    const eid = state.create();
    state.add(eid, Body);
    Body.shape.set(eid, ShapeKind.Box);
    Body.pos.set(eid, x, y, 0, 0);
    Body.halfExtents.set(eid, half[0], half[1], half[2], 0);
    Body.mass.set(eid, mass);
    state.add(eid, Part);
    state.add(eid, Color);
    Color.rgba.set(eid, color[0], color[1], color[2], 1);
    return eid;
}

// the fixed-tick driver. once every body has marshaled (`Physics.body` resolves), seed the book spins once and
// wire the parallel joint — seeding must be one-shot, or re-writing the angular velocity each tick would erase
// the physics it is meant to develop. thereafter it torques the panels and tracks each book's off-axis angular
// speed (energy leaked off its spin axis — the physics signature).
const driver: System = {
    name: "rotation-driver",
    group: "fixed",
    before: [StepSystem],
    update() {
        const world = Physics.world;
        const bookX = Physics.body(bookXEid);
        const bookY = Physics.body(bookYEid);
        const bookZ = Physics.body(bookZEid);
        const ref = Physics.body(refEid);
        const held = Physics.body(heldEid);
        if (!seeded) {
            if (!world || !bookX || !bookY || !bookZ || !ref || !held) return;
            bookX.setAngularVelocity({ x: SPIN, y: 0.01, z: 0.01 });
            bookY.setAngularVelocity({ x: 0.01, y: SPIN, z: 0.01 });
            bookZ.setAngularVelocity({ x: 0.01, y: 0.01, z: SPIN });
            world.createParallelJoint(ref, held, { maxTorque: 200, hertz: 4, dampingRatio: 1 });
            seeded = true;
        }
        held?.applyTorque({ x: 1.5, y: 1.5, z: 0 }, true);
        Physics.body(freeEid)?.applyTorque({ x: 1.5, y: 1.5, z: 0 }, true);
        if (bookX)
            bookXMaxOff = Math.max(
                bookXMaxOff,
                Math.hypot(bookX.getAngularVelocity().y, bookX.getAngularVelocity().z),
            );
        if (bookY)
            bookYMaxOff = Math.max(
                bookYMaxOff,
                Math.hypot(bookY.getAngularVelocity().x, bookY.getAngularVelocity().z),
            );
    },
};

const scenario: Scenario = {
    name: "rotation",
    params: [],

    async build() {
        const { state, dispose } = await run({
            defaults: false,
            capacity: 32,
            plugins: [
                ProfilePlugin,
                SlabPlugin,
                TransformsPlugin,
                InputPlugin,
                OrbitPlugin,
                RenderPlugin,
                PhysicsPlugin,
                PartPlugin,
                SearPlugin,
                GlazePlugin,
            ] as Plugin[],
        });
        Physics.world?.setGravity({ x: 0, y: 0, z: 0 });

        state.add(state.create(), AmbientLight);
        state.add(state.create(), DirectionalLight);

        // centered on the orbit target (origin): the chunky held/free panels sit at the middle so the
        // render gate's center-vs-corner probe reads content, not background (the off-center layout read
        // blank — verify.ts hasStructure). y=0 for all: zero-g means position is purely framing, never
        // dynamics — PROVIDED spacing exceeds rotational reach. Each book sweeps ~0.62 from its center
        // (hypot of BOOK_HALF) while tumbling/spinning, so the 2.0 book-to-book gaps keep the three books'
        // contact geometrically impossible. The bookZ↔held gap (1.5) budgets BOTH bodies' swept reach:
        // held's x-extent ~0.9 plus bookZ's swept ~0.36 = 1.26 < 1.5 — but that ~0.9 holds only while the
        // parallel joint pins held's orientation (the heldTilt < 0.15 rad assert). A broken joint lets held
        // physics to its ~1.27 corner reach (1.27 + 0.36 = 1.63 > 1.5, contact) — but the joint assert
        // reddens first, so this budget never rests on a silent near-miss.
        bookXEid = body(state, -5.5, 0, BOOK_HALF, 1, [0.9, 0.5, 0.4]);
        bookYEid = body(state, -3.5, 0, BOOK_HALF, 1, [0.5, 0.7, 0.6]);
        bookZEid = body(state, -1.5, 0, BOOK_HALF, 1, [0.5, 0.6, 0.85]);

        refEid = body(state, 0, 0, [0.15, 0.15, 0.15], 0, [0.55, 0.57, 0.6]);
        heldEid = body(state, 0, 0, [0.9, 0.12, 0.9], 1, [0.5, 0.7, 0.6]);
        freeEid = body(state, 2.5, 0, [0.9, 0.12, 0.9], 1, [0.9, 0.5, 0.4]);

        state.addSystem(driver);

        const cam = state.create();
        state.add(cam, Transform);
        state.add(cam, Camera);
        state.add(cam, Sear);
        state.add(cam, Orbit);
        Camera.mode.set(cam, CameraMode.Perspective);
        Orbit.yaw.set(cam, 0.4);
        Orbit.pitch.set(cam, 0.25);
        Orbit.distance.set(cam, 14);

        for (let i = 0; i < 2000 && state.time.fixedTick < TICKS; i++) await frames(1);

        return {
            state,
            dispose() {
                bookXEid = -1;
                bookYEid = -1;
                bookZEid = -1;
                refEid = -1;
                heldEid = -1;
                freeEid = -1;
                seeded = false;
                bookXMaxOff = 0;
                bookYMaxOff = 0;
                dispose();
            },
        };
    },

    assert(): Promise<Check[]> {
        const tilt = (eid: number): number => {
            const b = Physics.body(eid);
            if (!b) return Number.NaN;
            const q = b.getRotation();
            return Math.hypot(q.v.x, q.v.y, q.v.z);
        };
        const heldTilt = tilt(heldEid);
        const freeTilt = tilt(freeEid);
        return Promise.resolve([
            {
                name: "Dzhanibekov: intermediate-axis book physicss (energy leaks off its spin axis)",
                pass: bookXMaxOff > 1.5,
                detail: `book-x off-axis peak ${bookXMaxOff.toFixed(2)} (expect > 1.5)`,
            },
            {
                name: "major-axis book stays stable (spin axis holds)",
                pass: bookYMaxOff < 0.5,
                detail: `book-y off-axis peak ${bookYMaxOff.toFixed(2)} (expect < 0.5)`,
            },
            {
                name: "parallel joint locks the held panel's orientation",
                pass: Number.isFinite(heldTilt) && heldTilt < 0.15,
                detail: `held tilt ${heldTilt.toFixed(3)} (expect < 0.15)`,
            },
            {
                name: "the free panel physicss under the same torque",
                pass: Number.isFinite(freeTilt) && freeTilt > 0.3,
                detail: `free tilt ${freeTilt.toFixed(3)} (expect > 0.3)`,
            },
        ]);
    },

    live(): string {
        return `rotation — book-x off ${bookXMaxOff.toFixed(2)} (physicss), book-y off ${bookYMaxOff.toFixed(2)} (stable)`;
    },
};

register(scenario);
