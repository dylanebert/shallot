import type { World } from "@dylanebert/shallot";
import { createScene, type Scene } from "./scenes";

export function sceneSwitcher(world: World, view?: (world: World, scene: Scene) => () => void) {
    world.step(0);
    const empty = world.snapshot();
    let disposeView: (() => void) | undefined;
    return (scene: Scene): number[] => {
        disposeView?.();
        world.restore(empty);
        const bodies = createScene(world, scene);
        disposeView = view?.(world, scene);
        return bodies;
    };
}
