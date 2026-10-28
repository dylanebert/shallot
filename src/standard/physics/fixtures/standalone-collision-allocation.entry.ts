import { createApp } from "@dylanebert/shallot";
import {
    getSweepTransform,
    StandardPhysicsPlugin,
    shapeCast,
    shapeDistance,
    timeOfImpact,
} from "@dylanebert/shallot/standard/physics";
import { collisionCases, emptyCache } from "../standalone-collision.fixture";

export let controlSink: object | undefined;
export const control = () => {
    controlSink = { query: 0 };
};
export let leaked: object | undefined;
export default async function create(input: string) {
    const app = await createApp({ defaults: false, plugins: [StandardPhysicsPlugin] });
    const zero = { x: 0, y: 0, z: 0 },
        q = { v: zero, s: 1 };
    const sweepA = { localCenter: zero, c1: zero, c2: zero, q1: q, q2: q };
    const cases = collisionCases.map(({ input }) => ({
        distance: input,
        cache: emptyCache(),
        cast: { ...input, translationB: { x: -10, y: 0, z: 0 }, maxFraction: 1, canEncroach: true },
        toi: {
            ...input,
            sweepA,
            sweepB: {
                ...sweepA,
                c1: input.transform.p,
                c2: { x: input.transform.p.x - 10, y: input.transform.p.y, z: input.transform.p.z },
            },
            maxFraction: 1,
        },
    }));
    return {
        step() {
            for (const c of cases) {
                c.cache.count = 0;
                const distance = shapeDistance(app.world, c.distance, c.cache, 32);
                shapeDistance(app.world, c.distance, c.cache, 0);
                shapeCast(app.world, c.cast);
                timeOfImpact(app.world, c.toi);
                getSweepTransform(app.world, c.toi.sweepB, 0.5);
                if (input === "leak") leaked = { ...distance };
            }
        },
        dispose() {
            app.dispose();
        },
    };
}
