import { createApp, StandardPhysicsPlugin } from "@dylanebert/shallot";
import { manifoldSubject } from "./standalone-manifold.fixture";
export let controlSink: { frame: number } | undefined;
export const control = () => {
    controlSink = { frame: 0 };
};
export default async function create() {
    const app = await createApp({ defaults: false, plugins: [StandardPhysicsPlugin] });
    const subjects = Array.from({ length: 9 }, () => manifoldSubject(app.world));
    let frame = 0;
    const heights = [0.25, 0.5, 0.51, 6];
    return {
        step() {
            const y = heights[frame++ % 4];
            for (let i = 0; i < 9; i++) {
                const s = subjects[i];
                s.transform.p.y = y;
                for (let j = 0; j < 3; j++) s.triangle[j].y = -y;
                s.call(10 + i);
            }
        },
        dispose() {
            app.dispose();
        },
    };
}
