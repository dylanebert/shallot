import type { World } from "@dylanebert/shallot";
import { type Scene, scenes } from "../src/scenes";

export function advanceIteration(world: World, scene: Scene): void {
    const entry: {
        construct(world: World): number[];
        iterationComplete?: (world: World) => boolean;
    } = scenes[scene];
    for (let ticks = 0; ticks < 4; ticks++) {
        world.tick();
        if (!entry.iterationComplete || entry.iterationComplete(world)) return;
    }
    throw new Error(`${scene}: sample iteration did not complete within 4 ticks`);
}
