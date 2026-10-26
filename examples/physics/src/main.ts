import { runApp } from "@dylanebert/shallot";
import { mount, unmount } from "svelte";
import { AppPlugin, Viewer } from "./app";
import Hud from "./Hud.svelte";
import { sceneSwitcher } from "./sample";
import { changeControl, type Scene, scenes } from "./scenes";

const requested = new URL(location.href).searchParams.get("scene");
const initial = requested && Object.hasOwn(scenes, requested) ? (requested as Scene) : "boxStack";
await runApp({
    plugins: [AppPlugin],
    ui(container, world) {
        const replace = sceneSwitcher(world, world.resource(Viewer));
        let component: ReturnType<typeof mount>;
        function select(scene: Scene): void {
            const bodies = replace(scene);
            if (component) void unmount(component);
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
        }
        select(initial);
        return () => void unmount(component);
    },
});
