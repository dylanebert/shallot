import { runApp, StandardPhysicsPlugin } from "@dylanebert/shallot";
import { LinesPlugin, OrbitPlugin, TextPlugin } from "@dylanebert/shallot/extras";
import { mount, unmount } from "svelte";
import { EventSamplePlugin, SampleInputPlugin } from "./events";
import { GrabPlugin } from "./grab";
import Hud from "./Hud.svelte";
import { sceneSwitcher } from "./sample";
import { changeControl, type Scene, scenes } from "./scenes";
import { createViewer } from "./view";

const Viewer = { create: createViewer };
const requested = new URL(location.href).searchParams.get("scene");
const initial = requested && Object.hasOwn(scenes, requested) ? (requested as Scene) : "boxStack";
await runApp({
    plugins: [
        StandardPhysicsPlugin,
        EventSamplePlugin,
        SampleInputPlugin,
        TextPlugin,
        OrbitPlugin,
        LinesPlugin,
        GrabPlugin,
        {
            name: "SampleViewer",
            dependencies: [OrbitPlugin],
            initialize(world) {
                world.resource(Viewer);
            },
        },
    ],
    ui(container, world) {
        const replace = sceneSwitcher(world, world.resource(Viewer));
        let component: ReturnType<typeof mount>;
        function select(scene: Scene): void {
            const bodies = replace(scene);
            if (component) void unmount(component);
            component = mount(Hud, {
                target: container,
                props: {
                    scene,
                    select,
                    change: (name: string, value: number, values: Record<string, number>) =>
                        changeControl(world, scene, bodies, values, name, value),
                },
            });
        }
        select(initial);
        return () => void unmount(component);
    },
});
