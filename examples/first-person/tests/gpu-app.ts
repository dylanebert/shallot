import { expect } from "bun:test";
import { createApp } from "@dylanebert/shallot";
import { setupGlobals } from "@dylanebert/shallot/webgpu";

type App = Awaited<ReturnType<typeof createApp>>;

/**
 * Open a GPU app on Dawn, installing its globals here so a check never relies on another file having
 * done so in the same process. The engine warns once when the adapter is not real hardware; the check
 * owns that notice: it is captured and must be exactly the one the adapter's verdict calls for.
 */
export async function openGpuApp(config: Parameters<typeof createApp>[0]): Promise<App> {
    await setupGlobals();
    const notices: string[] = [];
    const warn = console.warn;
    console.warn = (...args: unknown[]) => notices.push(args.map(String).join(" "));
    let app: App;
    try {
        app = await createApp(config);
    } finally {
        console.warn = warn;
    }
    const adapter = app.world.gpuIfAvailable?.adapter;
    expect(notices).toEqual(
        !adapter || adapter.class === "real"
            ? []
            : [`[shallot] ${adapter.reason ?? `${adapter.class} adapter: ${adapter.identity}`}`],
    );
    return app;
}
