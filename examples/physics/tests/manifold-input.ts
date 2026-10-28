import type { World } from "@dylanebert/shallot";
import { manifoldDrag } from "../src/manifold-drag";
import { changeControl, type Scene } from "../src/scenes";
export function manifoldInput(
    world: World,
    scene: Scene,
    values: Record<string, number>,
    ray: { origin: { x: number; y: number; z: number }; held: boolean },
    name: string,
    value: number,
): void {
    if (name.startsWith("drag")) {
        if (name === "drag") {
            ray.held = Boolean(value);
            manifoldDrag(world, ray.origin, { x: 0, y: 0, z: -20 }, ray.held, ray.held);
        } else ray.origin[name.slice(4).toLowerCase() as "x" | "y" | "z"] = Math.fround(value);
        return;
    }
    const feature = ["auto", "faceA", "faceB", "edgePair"].indexOf(name);
    changeControl(
        world,
        scene,
        [],
        values,
        feature >= 0 ? "feature" : name,
        feature >= 0 ? feature : value,
    );
}
