import { getWorldCounters, PhysicsWorldDefinition } from "@dylanebert/shallot/standard/physics";
import { counterPile } from "./world-counters";

export let controlSink: object | undefined;
export const control = () => {
    controlSink = { counter: 0 };
};
export let leaked: object | undefined;

export default async function create(input: string) {
    const app = await counterPile(true);
    app.world.resource(PhysicsWorldDefinition).enableAutomaticStep = true;
    app.world.tick();
    const first = getWorldCounters(app.world)!;
    const colors = first.colorCounts;
    const manifolds = first.manifoldCounts;
    return {
        step() {
            const counters = getWorldCounters(app.world)!;
            if (
                counters !== first ||
                counters.colorCounts !== colors ||
                counters.manifoldCounts !== manifolds ||
                counters.bodyCount !== 27
            )
                throw new Error("counter record was not reused");
            if (input === "leak") leaked = { ...counters };
        },
        dispose() {
            app.dispose();
        },
    };
}
