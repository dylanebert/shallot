import {
    ColorGrading,
    MeshInstance,
    Tonemapping,
    TonemappingMethod,
    type World,
} from "@dylanebert/shallot";
import { Scene } from "./scene";

export const methods = Object.entries(TonemappingMethod);
export function setMethod(world: World, method: number): void {
    const camera = world.resource(Scene).camera;
    world.add(camera, Tonemapping, { method });
    const exposure =
        method === TonemappingMethod.Reinhard || method === TonemappingMethod.ReinhardLuminance
            ? 0.5
            : method === TonemappingMethod.AcesFitted
              ? 0.35
              : method === TonemappingMethod.AgX
                ? -0.2
                : 0;
    const saturation = method === TonemappingMethod.AgX ? 1.1 : 1;
    world.remove(camera, ColorGrading);
    world.add(camera, ColorGrading, {
        exposure,
        postSaturation: saturation,
        saturation: [saturation, saturation, saturation],
    });
}
export function setScene(world: World, pattern: boolean): void {
    const scene = world.resource(Scene);
    scene.subjects.forEach((eid, index) => {
        if (pattern) world.remove(eid, MeshInstance);
        else world.add(eid, MeshInstance, { mesh: scene.meshes[index] });
    });
    if (pattern)
        world.add(scene.pattern, MeshInstance, { mesh: scene.meshes[scene.subjects.length] });
    else world.remove(scene.pattern, MeshInstance);
}
