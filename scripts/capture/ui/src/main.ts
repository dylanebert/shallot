import { run } from "@dylanebert/shallot";
import { config } from "./lib";

// boots the standalone run() app; the ui-containment flow waits on __uiReady + the sized canvas, then
// samples the host chrome to prove config.ui's overlay stayed inside the canvas region.
const { dispose } = await run(config);
(window as Window & { __uiReady?: boolean }).__uiReady = true;

if (import.meta.hot) {
    import.meta.hot.dispose(() => dispose());
}
