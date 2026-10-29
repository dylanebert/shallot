import type { World } from "@dylanebert/shallot";
import { PhysicsWorldDefinition } from "@dylanebert/shallot/standard/physics";
import { Geometry } from "../../src/geometry";
import type { Controls } from "../../src/scenes";
import { IssueState, IssueVelocities, issueScenes } from "./issues";

type Scene = keyof typeof issueScenes;
export function createScene(world: World, scene: Scene): number[] {
    world.resource(IssueVelocities).length = 0;
    Object.assign(world.resource(IssueState), IssueState.create());
    Object.assign(world.resource(Geometry), Geometry.create());
    Object.assign(world.resource(PhysicsWorldDefinition), {
        contactHertz: 30,
        contactDampingRatio: 10,
        contactSpeed: 3,
        enableAutomaticStep: true,
        workerCount: 1,
        maximumMemory: 4 * 1024 ** 3,
    });
    return issueScenes[scene].construct(world);
}
export function controlValues(scene: Scene): Record<string, number> {
    return Object.fromEntries(
        Object.entries(issueScenes[scene].controls ?? {}).map(([name, control]) => [
            name,
            control.value,
        ]),
    );
}
export function changeControl(
    world: World,
    scene: Scene,
    bodies: readonly number[],
    values: Record<string, number>,
    name: string,
    value: number,
): void {
    const controls: Controls | undefined = issueScenes[scene].controls;
    const selected = controls?.[name];
    if (!selected || selected.read) throw new Error(`Unknown ${scene} control: ${name}`);
    if (selected.when && !selected.when.every((name) => values[name])) return;
    if (selected.enabled && !selected.enabled(world)) return;
    values[name] = value;
    selected.set(world, bodies, value, values);
}
