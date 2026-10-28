import { Resolution, type World } from "@dylanebert/shallot";
import { Posterize } from "./posterize";
import { Scene } from "./scene";
export const settings = [
    { label: "Resolution height", min: 180, max: 720, value: 360 },
    { label: "Color levels", min: 2, max: 16, value: 8 },
] as const;
export function setSetting(world: World, index: number, value: number): void {
    const camera = world.resource(Scene).camera;
    if (index === 0) world.storage(Resolution).height.set(camera, value);
    else world.storage(Posterize).levels.set(camera, value);
}
export function setDither(world: World, enabled: boolean): void {
    world.storage(Posterize).dither.set(world.resource(Scene).camera, Number(enabled));
}
