import { type App, createApp } from "../engine/app";
import { setupGlobals } from "@dylanebert/shallot/webgpu";
import { adapterNotice } from "../engine/runtime";

/**
 * Install WebGPU globals and open a check's app. Missing and null-backend adapters are refused
 * before device creation. The engine warns once on a non-real adapter;
 * the check owns that notice, so captured warnings must equal exactly the adapter's notice.
 */
export async function openGpuApp(config: Parameters<typeof createApp>[0]): Promise<App> {
    await setupGlobals();
    const adapter = config.device ? config.adapter : await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error("No WebGPU adapter is available for the GPU check");
    if (adapter.info.device === "null-backend") {
        throw new Error("WebGPU adapter null-backend cannot run a GPU check");
    }
    const notices: string[] = [];
    const warn = console.warn;
    console.warn = (...args) => notices.push(args.map(String).join(" "));
    let app: App;
    try {
        app = await createApp(config);
    } finally {
        console.warn = warn;
    }
    const notice = adapterNotice(app.world.gpuIfAvailable?.adapter);
    const expected = notice === undefined ? [] : [notice];
    if (JSON.stringify(notices) !== JSON.stringify(expected)) {
        app.dispose();
        throw new Error(
            `Expected adapter notices ${JSON.stringify(expected)}, observed ${JSON.stringify(notices)}`,
        );
    }
    return app;
}
