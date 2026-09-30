import type { Loading, Plugin, State } from "@dylanebert/shallot";
import type { HostFrame } from "./host";

/** Reveal the host canvas only after the first draw has been submitted and completed. */
export function revealAfterFirstFrame(host: HostFrame, loading: Pick<Loading, "error">): Plugin {
    let waiting = false;
    let disposed = false;
    const fail = (error: unknown): void => {
        loading.error?.(error);
        // Loading.error has no overlay to update after cleanup; HostFrame renders the in-frame fallback then.
        host.fail(error);
    };

    return {
        name: "LoadingScreenReveal",
        systems: [
            {
                name: "first-frame",
                group: "draw",
                update(state: State) {
                    if (waiting || disposed) return;
                    waiting = true;
                    // This draw system runs before the terminal submission. Ask for the queue fence only
                    // after state.step returns, when the frame's commands have been submitted.
                    queueMicrotask(() => {
                        if (disposed) return;
                        const device = state.gpu?.device;
                        if (!device) {
                            fail(new Error("the first frame had no WebGPU device"));
                            return;
                        }
                        void device.queue
                            .onSubmittedWorkDone()
                            .then(() => {
                                if (!disposed) host.reveal();
                            })
                            .catch(fail);
                    });
                },
            },
        ],
        dispose() {
            disposed = true;
        },
    };
}
