import { minimalDark, run } from "@dylanebert/shallot";
import { OrbitPlugin } from "@dylanebert/shallot/extras";
import { mountHost } from "./host";
import { revealAfterFirstFrame } from "./reveal";
import { SCENE } from "./scene";

const host = mountHost();
let app: Awaited<ReturnType<typeof run>> | undefined;
let pageClosed = false;
window.addEventListener("pagehide", () => {
    pageClosed = true;
    app?.dispose();
});

void run({
    plugins: [OrbitPlugin, revealAfterFirstFrame(host)],
    scene: SCENE,
    loading: minimalDark({ container: host.frame }),
    pixelRatio: 1,
})
    .then((running) => {
        if (pageClosed) running.dispose();
        else app = running;
    })
    .catch(host.fail);
