import { build, Time } from "@dylanebert/shallot";
import { check } from "@dylanebert/shallot/harness/check";
import {
    Body,
    hash,
    PhysicsPlugin,
    physicsStepConfig,
    physicsWorld,
    restore,
    snapshot,
} from "@dylanebert/shallot/physics";

const FALLING_SCENE = `<scene><a body="shape: 1; pos: 0 3 0; half-extents: 0 0 0 0.5; mass: 1" /></scene>`;
const EULER_SCENE = `<scene><a id="wheel" body="shape: 1; pos: 0 1.5 0; half-extents: 0 0 0 0.4; mass: 0.5; quat: 90 0 0" /></scene>`;

function rotateY(quat: readonly [number, number, number, number]): [number, number, number] {
    const [x, y, z, w] = quat;
    return [2 * (x * y - z * w), 1 - 2 * (x * x + z * z), 2 * (y * z + x * w)];
}

check(
    "Physics reports the fixed-step values used by its production system",
    {
        claim: "vehicle trajectory bounds can duplicate gravity and substeps instead of reading the initialized Physics system's fixed-step configuration",
    },
    async () => {
        const app = await build({
            defaults: false,
            plugins: [PhysicsPlugin],
            scene: FALLING_SCENE,
        });
        try {
            const config = physicsStepConfig(app.state);
            app.state.step(config.dt);
            const world = physicsWorld(app.state);
            if (!world) throw new Error("Physics world did not warm");
            const gravity = world.getGravity();
            if (
                config.dt !== Time.FIXED_DT ||
                gravity.x !== 0 ||
                gravity.z !== 0 ||
                gravity.y !== config.gravity
            )
                throw new Error(
                    `reported ${JSON.stringify(config)} but the world runs ${JSON.stringify(gravity)}`,
                );
            const before = snapshot(app.state);
            app.state.step(config.dt);
            const production = hash(app.state);
            const replay = (substeps: number) => {
                restore(app.state, before);
                world.step(config.dt, substeps);
                return hash(app.state);
            };
            if (replay(config.substeps) !== production)
                throw new Error(
                    `reported ${config.substeps} substeps but the production step differs`,
                );
            if (replay(config.substeps + 1) === production)
                throw new Error(
                    "substep count does not change the step, so the replay proves nothing",
                );
        } finally {
            app.dispose();
        }
    },
);

check(
    "scene Body Euler authoring produces a unit wheel rotation",
    {
        claim: "a Body authored with +90 degrees about X stores a unit quaternion rotating local Y to world Z, so raw Euler lanes cannot reach the solver",
    },
    async () => {
        const app = await build({ defaults: false, plugins: [PhysicsPlugin], scene: EULER_SCENE });
        try {
            const eid = [...app.state.query([Body])][0];
            if (eid === undefined) throw new Error("Euler scene did not create a Body");
            const quat: [number, number, number, number] = [
                Body.quat.x.get(eid),
                Body.quat.y.get(eid),
                Body.quat.z.get(eid),
                Body.quat.w.get(eid),
            ];
            const norm = Math.hypot(...quat);
            const yAxis = rotateY(quat);
            if (Math.abs(norm - 1) > 1e-5)
                throw new Error(`Body Euler quaternion was not unit: [${quat.join(", ")}]`);
            if (Math.hypot(yAxis[0], yAxis[1], yAxis[2] - 1) > 1e-5)
                throw new Error(`+90 X did not rotate local Y to world Z: [${yAxis.join(", ")}]`);
        } finally {
            app.dispose();
        }
    },
);
