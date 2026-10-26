import { runApp } from "@dylanebert/shallot";
import { profilerBounds } from "@dylanebert/shallot/extras";
import { flushSync, mount, unmount } from "svelte";
import { AppPlugin, Viewer } from "./app";
import Hud from "./Hud.svelte";
import { profilerFits, sceneSwitcher } from "./sample";
import { changeControl, type Scene, scenes } from "./scenes";

const requested = new URL(location.href).searchParams.get("scene");
const initial = requested && Object.hasOwn(scenes, requested) ? (requested as Scene) : "boxStack";
await runApp({
    plugins: [AppPlugin],
    ui(container, world) {
        const replace = sceneSwitcher(world, world.resource(Viewer), () => {
            const bounds = profilerBounds(world);
            return (
                !!bounds &&
                profilerFits(
                    bounds,
                    Array.from(
                        container.querySelectorAll(".controls, button, input, select, summary"),
                        (control) => control.getBoundingClientRect(),
                    ),
                )
            );
        });
        window.addEventListener("resize", () => replace.reconcileProfiler(), {
            signal: world.signal,
        });
        let component: ReturnType<typeof mount>;
        function select(scene: Scene): void {
            let bodies: number[];
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
            bodies = replace(scene);
        }
        select(initial);
        return () => void unmount(component);
    },
});
