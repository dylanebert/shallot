import { build, Time } from "@dylanebert/shallot";
import { check } from "@dylanebert/shallot/harness/check";
import { Body, PhysicsPlugin, physicsStepConfig } from "@dylanebert/shallot/physics";

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
        const app = await build({ defaults: false, plugins: [PhysicsPlugin] });
        try {
            const config = physicsStepConfig(app.state);
            if (config.dt !== Time.FIXED_DT || config.gravity !== -10 || config.substeps !== 4)
                throw new Error(`unexpected Physics step configuration: ${JSON.stringify(config)}`);
            app.state.step(config.dt);
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
