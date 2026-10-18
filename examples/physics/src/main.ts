import { MeshPlugin, runApp, StandardPhysicsPlugin } from "@dylanebert/shallot";
import { LinesPlugin, OrbitPlugin } from "@dylanebert/shallot/extras";
import { mount, unmount } from "svelte";
import { GrabPlugin } from "./grab";
import Hud from "./Hud.svelte";
import { changeControl, createScene, type Scene } from "./scenes";
import { createView } from "./view";

async function switchScene(scene: Scene): Promise<void> {
    let bodies: number[] = [];
    const app = await runApp({
        plugins: [
            StandardPhysicsPlugin,
            OrbitPlugin,
            LinesPlugin,
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
        ui(container, world) {
            const component = mount(Hud, {
                target: container,
                props: {
                    scene,
                    select: (next: Scene) => {
                        app.dispose();
                        void switchScene(next);
                    },
                    change: (name: string, value: number, values: Record<string, number>) =>
                        changeControl(world, scene, bodies, values, name, value),
                },
            });
            return () => void unmount(component);
        },
    });
}
await switchScene("boxStack");
