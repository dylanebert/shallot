import { type App, MeshPlugin, runApp, StandardPhysicsPlugin } from "@dylanebert/shallot";
import { OrbitPlugin } from "@dylanebert/shallot/extras";
import { GrabPlugin } from "./grab";
import { createScene, type Scene, scenes } from "./scenes";
import { createView } from "./view";

const select = document.querySelector<HTMLSelectElement>("#scene")!;
const gravity = document.querySelector<HTMLInputElement>("#gravity")!;
const control = document.querySelector<HTMLElement>("#gravity-control")!;
let app: App | undefined;
let bodies: number[] = [];

async function switchScene(): Promise<void> {
    select.disabled = true;
    app?.dispose();
    const scene = select.value as Scene;
    gravity.value = "1";
    control.hidden = !scenes[scene].controls;
    app = await runApp({
        plugins: [
            StandardPhysicsPlugin,
            OrbitPlugin,
            GrabPlugin,
            {
                name: "SampleView",
                dependencies: [MeshPlugin],
                initialize: (world) => createView(world, scene),
            },
        ],
        setup(world) {
            bodies = createScene(world, scene);
        },
    });
    select.disabled = false;
}
select.addEventListener("change", () => void switchScene());
gravity.addEventListener("input", () => {
    if (app)
        scenes[select.value as Scene].controls?.gravityScale(
            app.world,
            bodies,
            Number(gravity.value),
        );
});
await switchScene();
