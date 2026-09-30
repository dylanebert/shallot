import { createApp, CharacterPlugin, InputPlugin, PhysicsPlugin } from "../../../src/index";
import { Demo } from "../../../examples/first-person/src/demo";
const FIXED_DT = 1 / 60;
const NoPhysicsStep = { ...PhysicsPlugin, name: "NoPhysicsStep", systems: PhysicsPlugin.systems.filter((system) => system.name !== "step") };
const VariantDemo = { ...Demo, name: "VariantDemo", dependencies: [NoPhysicsStep, CharacterPlugin, InputPlugin], systems: [] };
export let controlSink: { frame: number } | undefined;
export const control = () => {
    controlSink = { frame: 0 };
};
export default async function create(scene: string) {
    const app = await createApp({ defaults: false, plugins: [NoPhysicsStep, CharacterPlugin, InputPlugin, VariantDemo], scene });
    const world = app.world;
    return {
        step: () => world.step(FIXED_DT),
        wait: () => world.gpu.device.queue.onSubmittedWorkDone(),
        dispose: () => app.dispose(),
    };
}
