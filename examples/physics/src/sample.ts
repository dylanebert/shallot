import type { World } from "@dylanebert/shallot";
import { showProfiler } from "@dylanebert/shallot/extras";
import { MotorizedJoint } from "./roles";
import { SceneRestart } from "./scene-restart";
import { categories, createScene, type Scene } from "./scenes";

type Bounds = Pick<DOMRectReadOnly, "left" | "right" | "top" | "bottom">;

export function profilerFits(bounds: Bounds, controls: readonly Bounds[]): boolean {
    return controls.every(
        (control) =>
            control.right <= control.left ||
            control.bottom <= control.top ||
            bounds.left >= control.right ||
            bounds.right <= control.left ||
            bounds.top >= control.bottom ||
            bounds.bottom <= control.top,
    );
}

export function sceneSwitcher(
    world: World,
    view?: (world: World, scene: Scene) => () => void,
    profilerFits = () => true,
) {
    world.storage(MotorizedJoint);
    world.step(0);
    const empty = world.snapshot();
    let disposeView: (() => void) | undefined;
    let priorProfiler: boolean | undefined;
    let autoOpened = false;
    if (typeof window !== "undefined")
        window.addEventListener(
            "keydown",
            (event) => {
                if (event.key === "F3") autoOpened = false;
            },
            { signal: world.signal },
        );
    const benchmarks = new Set(
        categories
            .find((category) => category.name === "Benchmark")!
            .scenes.map((scene) => scene.id),
    );
    const select = (scene: Scene): number[] => {
        autoOpened = false;
        if (benchmarks.has(scene) && profilerFits()) {
            const previous = showProfiler(world);
            if (priorProfiler === undefined) priorProfiler = previous;
            autoOpened = priorProfiler === false;
        } else if (priorProfiler !== undefined) {
            showProfiler(world, priorProfiler);
            priorProfiler = undefined;
        }
        const bodies: number[] = [];
        const restart = (construct: (world: World) => number[]) => {
            disposeView?.();
            world.restore(empty);
            bodies.length = 0;
            for (const eid of construct(world)) bodies.push(eid);
            disposeView = view?.(world, scene);
        };
        world.resource(SceneRestart).restart = restart;
        restart((world) => createScene(world, scene));
        return bodies;
    };
    select.reconcileProfiler = () => {
        if (autoOpened && priorProfiler !== undefined && !profilerFits()) {
            showProfiler(world, priorProfiler);
            autoOpened = false;
        }
    };
    return select;
}
