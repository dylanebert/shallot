import { createApp, mountOverlay } from "@dylanebert/shallot";
import { Orbit } from "@dylanebert/shallot/extras";
import { PhysicsWorldDefinition } from "@dylanebert/shallot/standard/physics";
import { flushSync, mount, unmount } from "svelte";
import { AppPlugin, Viewer } from "../src/app";
import Hud from "../src/Hud.svelte";
import { issueScenes } from "../src/issues";
import { sceneSwitcher } from "../src/sample";
import { changeControl, type Scene } from "../src/scenes";

const requested = new URL(location.href).searchParams.get("scene")!;
if (!Object.hasOwn(issueScenes, requested)) throw new Error(`Unknown Issues scene: ${requested}`);
const app = await createApp({
    plugins: [AppPlugin],
    setup(world) {
        world.resource(PhysicsWorldDefinition).workerCount = 1;
    },
});
const world = app.world;
const container = mountOverlay(document.querySelector("canvas")!, world);
const replace = sceneSwitcher(world, world.resource(Viewer));
let component: ReturnType<typeof mount> | undefined;
function select(scene: Scene) {
    const bodies = replace(scene);
    if (component) void unmount(component);
    flushSync(() => {
        component = mount(Hud, {
            target: container,
            props: {
                world,
                scene,
                select,
                change: (name: string, value: number, values: Record<string, number>) =>
                    changeControl(world, scene, bodies, values, name, value),
            },
        });
    });
}
select(requested as Scene);
const definition = world.resource(PhysicsWorldDefinition);
const automaticStep = definition.enableAutomaticStep;
definition.enableAutomaticStep = false;
for (const camera of world.query([Orbit])) world.storage(Orbit).smoothness.set(camera, 1);
// Sync the starting bodies and snap the authored orbit without advancing the solver.
await world.step(1 / 60);
definition.enableAutomaticStep = automaticStep;
Object.assign(window, {
    issuesCaptureReady: true,
    async drawIssuesCapture() {
        await world.step(0);
    },
    async advanceIssuesCapture() {
        for (let i = 0; i < 120; i++) world.tick();
        await world.step(0);
    },
});
