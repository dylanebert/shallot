import { run } from "@dylanebert/shallot";
import { config } from "./lib";

async function init(): Promise<void> {
    await run(config);
}

if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
} else {
    init();
}
