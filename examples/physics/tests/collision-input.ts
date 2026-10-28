import type { World } from "@dylanebert/shallot";
import { CollisionState } from "../src/collision";
import { collisionDrag } from "../src/collision-drag";
import { changeControl, type Scene } from "../src/scenes";
export function collisionInput(
    world: World,
    scene: Scene,
    bodies: number[],
    values: Record<string, number>,
    name: string,
    value: number,
): void {
    if (name !== "queryY" && name !== "queryZ") {
        changeControl(world, scene, bodies, values, name, value);
        return;
    }
    const s = world.resource(CollisionState),
        axis = name === "queryY" ? "y" : "z";
    const center = s.sweeps[0]!.points[0]!,
        origin = { ...center, z: center.z - 10 },
        direction = { x: 0, y: 0, z: 20 };
    if (!collisionDrag(world, origin, direction, true, true))
        throw new Error("query drag missed its sphere");
    const target = { ...origin };
    target[axis] += value - s.offset[axis];
    collisionDrag(world, target, direction, false, true);
    collisionDrag(world, target, direction, false, false);
}
