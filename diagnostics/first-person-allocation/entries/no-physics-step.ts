import { build, CharacterPlugin, InputPlugin, PhysicsPlugin } from "../../../src/index";
import { Demo } from "../../../examples/first-person/src/demo";
const FIXED_DT = 1 / 60;
const NoPhysicsStep = { ...PhysicsPlugin, name: "NoPhysicsStep", systems: PhysicsPlugin.systems.filter((system) => system.name !== "step") };
const VariantDemo = { ...Demo, name: "VariantDemo", dependencies: [NoPhysicsStep, CharacterPlugin, InputPlugin], systems: [] };
export let controlSink: { frame: number } | undefined;
export const control = () => {
    controlSink = { frame: 0 };
};
export default async function create(scene: string) {
    const app = await build({ defaults: false, plugins: [NoPhysicsStep, CharacterPlugin, InputPlugin, VariantDemo], scene });
    const state = app.state;
    return { step: () => state.step(FIXED_DT), dispose: () => app.dispose() };
}
