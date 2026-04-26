import { run } from "@dylanebert/shallot";
import { config } from "./lib";

run(config).catch((e) => {
    console.error("Fatal:", e);
    process.exit(1);
});
