import { type Plugin, type Resource, SpotLight, Transform } from "@dylanebert/shallot";
import { UpdatePlayerControlSystem } from "./player";

export const Flashlight: Resource<{ camera: number; light: number }> = {
    create: () => ({ camera: 0, light: 0 }),
};
export const FlashlightPlugin: Plugin = {
    name: "Flashlight",
    systems: [
        {
            name: "carry",
            group: "simulation",
            after: [UpdatePlayerControlSystem],
            update(world) {
                const { camera, light } = world.resource(Flashlight);
                if (!camera || !light) return;
                const t = world.storage(Transform);
                const p = t.translation,
                    q = t.rotation;
                p.set(light, p.x.get(camera), p.y.get(camera), p.z.get(camera));
                q.set(light, q.x.get(camera), q.y.get(camera), q.z.get(camera), q.w.get(camera));
            },
        },
    ],
    components: [SpotLight],
};
