import { minimalDark, runApp } from "@dylanebert/shallot";
import { mountHost } from "../../examples/loading-screen/src/host";
import { LoadingWorld } from "../../examples/loading-screen/src/scene";
import { revealAfterFirstFrame } from "../../examples/loading-screen/src/reveal";
import { installScheduleRecorder } from "./recorder";
import { GpuWorkPlugin } from "./scene";

const host = mountHost();
const loading = minimalDark({ container: host.frame });
installScheduleRecorder("gpu-bound");

let app: Awaited<ReturnType<typeof runApp>> | undefined;
let pageClosed = false;
window.addEventListener("pagehide", () => {
    pageClosed = true;
    app?.dispose();
});

void runApp({
    plugins: [LoadingWorld, GpuWorkPlugin, revealAfterFirstFrame(host, loading)],
    loading,
    pixelRatio: 1,
})
    .then((running) => {
        if (pageClosed) running.dispose();
        else app = running;
    })
    .catch(host.fail);
