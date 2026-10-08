import { createApp } from "@dylanebert/shallot";

export let controlSink: { tick: number } | undefined;
export const control = () => {
    controlSink = { tick: 0 };
};

export default async function create() {
    const app = await createApp({ defaults: false, plugins: [] });
    app.world.addSystem({
        group: "fixed",
        update: (world) => {
            if (world.time.elapsed !== world.time.fixedTick * (1 / 60))
                throw new Error("tick clock diverged");
        },
    });
    return {
        step: () => app.world.tick(),
        wait: () => app.world.gpu.device.queue.onSubmittedWorkDone(),
        dispose: () => app.dispose(),
    };
}
