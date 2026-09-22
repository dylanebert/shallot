import { build, CharacterPlugin, InputPlugin, PhysicsPlugin } from "../../../src/index";
import { Demo } from "../../../examples/first-person/src/demo";
export let controlSink: { frame: number } | undefined;
export const control = () => {
    controlSink = { frame: 0 };
};
export default async function create(scene: string) {
    const app = await build({ defaults: false, plugins: [PhysicsPlugin, CharacterPlugin, InputPlugin, Demo], scene });
    return { step: () => {}, dispose: () => app.dispose() };
}
