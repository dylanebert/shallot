import type { World } from "@dylanebert/shallot";
import { SceneRestart } from "./scene-restart";
import { createScene, type Scene } from "./scenes";

export function sceneSwitcher(world: World, view?: (world: World, scene: Scene) => () => void) {
    world.step(0);
    const empty = world.snapshot();
    let disposeView: (() => void) | undefined;
    return (scene: Scene): number[] => {
        const bodies: number[] = [];
        const restart = (construct: (world: World) => number[]) => {
            disposeView?.();
            world.restore(empty);
            bodies.splice(0, bodies.length, ...construct(world));
            disposeView = view?.(world, scene);
        };
        world.resource(SceneRestart).restart = restart;
        restart((world) => createScene(world, scene));
        return bodies;
    };
}
