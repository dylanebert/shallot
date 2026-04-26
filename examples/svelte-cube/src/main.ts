import { run } from "@dylanebert/shallot";
import { config } from "./lib";

const state = await run(config);

if (import.meta.hot) {
    import.meta.hot.dispose(() => state.dispose());
}
