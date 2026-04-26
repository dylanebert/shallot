import {
    Transform,
    rotate,
    traits,
    type State,
    type System,
    type Plugin,
    type Config,
} from "@dylanebert/shallot";
import { OrbitPlugin } from "@dylanebert/shallot/extras";
import { svelteOverlay } from "./overlay";

export const RotateCube = { speedX: [] as number[], speedY: [] as number[] };
traits(RotateCube, { defaults: () => ({ speedX: 60, speedY: 30 }) });

const RotateCubeSystem: System = {
    group: "simulation",
    update(state: State) {
        const dt = state.time.deltaTime;
        for (const eid of state.query([RotateCube, Transform])) {
            const q = rotate(
                Transform.quatX[eid],
                Transform.quatY[eid],
                Transform.quatZ[eid],
                Transform.quatW[eid],
                RotateCube.speedX[eid] * dt,
                RotateCube.speedY[eid] * dt,
                0,
            );
            Transform.quatX[eid] = q.x;
            Transform.quatY[eid] = q.y;
            Transform.quatZ[eid] = q.z;
            Transform.quatW[eid] = q.w;
        }
    },
};

export const DemoPlugin: Plugin = {
    name: "Demo",
    systems: [RotateCubeSystem],
    components: { RotateCube },
};

export const config: Config = {
    plugins: [OrbitPlugin, DemoPlugin],
    scene: "/scenes/demo.scene",
    ui: svelteOverlay,
};
