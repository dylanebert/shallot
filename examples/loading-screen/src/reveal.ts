import { Compute, type Plugin } from "@dylanebert/shallot";
import type { HostFrame } from "./host";

/** Reveal the host canvas only after the first draw has been submitted and completed. */
export function revealAfterFirstFrame(host: HostFrame): Plugin {
    let waiting = false;
    let disposed = false;

    return {
        name: "LoadingScreenReveal",
        systems: [
            {
                name: "first-frame",
                group: "draw",
                update() {
                    if (waiting || disposed) return;
                    waiting = true;
                    // This draw system runs before the terminal submission. Ask for the queue fence only
                    // after state.step returns, when the frame's commands have been submitted.
                    queueMicrotask(() => {
                        if (disposed) return;
                        const device = Compute.device;
                        if (!device) {
                            host.fail(new Error("the first frame had no WebGPU device"));
                            return;
                        }
                        void device.queue
                            .onSubmittedWorkDone()
                            .then(() => {
                                if (!disposed) host.reveal();
                            })
                            .catch(host.fail);
                    });
                },
            },
        ],
        dispose() {
            disposed = true;
        },
    };
}
