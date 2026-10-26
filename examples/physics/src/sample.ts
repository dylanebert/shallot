import type { World } from "@dylanebert/shallot";
import { showProfiler } from "@dylanebert/shallot/extras";
import { SceneRestart } from "./scene-restart";
import { categories, createScene, type Scene } from "./scenes";

export function sceneSwitcher(world: World, view?: (world: World, scene: Scene) => () => void) {
    world.step(0);
    const empty = world.snapshot();
    let disposeView: (() => void) | undefined;
    let priorProfiler: boolean | undefined;
    const benchmarks = new Set(
        categories
            .find((category) => category.name === "Benchmark")!
            .scenes.map((scene) => scene.id),
    );
    return (scene: Scene): number[] => {
        if (benchmarks.has(scene)) {
            const previous = showProfiler(world);
            if (priorProfiler === undefined) priorProfiler = previous;
        } else if (priorProfiler !== undefined) {
            showProfiler(world, priorProfiler);
            priorProfiler = undefined;
        }
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
