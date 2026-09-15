import { build, CharacterPlugin, InputPlugin, PhysicsPlugin, Time } from "@dylanebert/shallot";
import { Demo } from "./demo";

// Bundled for Node by the allocation row, which passes the scene as XML text since the runtime's
// file loader is Bun or fetch. Same CPU composition as the demo rows.
export default async function create(scene: string) {
    const app = await build({
        defaults: false,
        plugins: [PhysicsPlugin, CharacterPlugin, InputPlugin, Demo],
        scene,
    });
    return {
        step: () => app.state.step(Time.FIXED_DT),
        dispose: () => app.dispose(),
    };
}
