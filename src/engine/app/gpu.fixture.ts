import { afterAll } from "bun:test";
import { rawDevice } from "../runtime";
import { build } from "./index";

/** Acquire outside individual tests; small worlds in one file reuse the native device. */
export async function sharedGpuBuild(): Promise<typeof build> {
    const owner = await build({ defaults: false, plugins: [] });
    const device = rawDevice(owner.state.gpu.device);
    afterAll(() => owner.dispose());
    return (options) => build({ ...options, device: options.device ?? device });
}
