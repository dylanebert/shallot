import type { Loading, Plugin, World } from "@dylanebert/shallot";
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
        name: "EmbeddedSceneReveal",
        systems: [
            {
                name: "first-frame",
                group: "draw",
                update(world: World) {
                    if (waiting || disposed) return;
                    waiting = true;
                    world.onFrameSubmit(() => {
                        const fence = world.frameFence;
                        if (!fence) {
                            fail(new Error("the first frame had no WebGPU device"));
                            return;
                        }
                        fence
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
