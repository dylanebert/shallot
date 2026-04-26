import { run, type Config } from "@dylanebert/shallot";
import { Runtime } from "@dylanebert/shallot/runtime";
import { Audio } from "@dylanebert/shallot/audio/core";

const config: Config = {
    plugins: [],
    scene: "/scenes/workstation.scene",
};

let app: Record<string, unknown> | null = null;

async function init() {
    const ecs = await run(config);

    const audio = Audio.from(ecs)!;

    if (Runtime === "web") {
        const { mount, unmount } = await import("svelte");
        const { default: App } = await import("./App.svelte");
        app = mount(App, {
            target: document.getElementById("app")!,
            props: { audio },
        });

        if (import.meta.hot) {
            import.meta.hot.dispose(() => {
                if (app) unmount(app);
                ecs.dispose();
            });
        }
    } else {
        if (import.meta.hot) {
            import.meta.hot.dispose(() => ecs.dispose());
        }
    }
}

init();
