import { run } from "@dylanebert/shallot";
import { config } from "./lib";

const useTwoLevelBVH = new URLSearchParams(location.search).get("twolevel") === "true";
(window as any).__USE_TWO_LEVEL_BVH__ = useTwoLevelBVH;

const state = await run(config);

if (import.meta.hot) {
    import.meta.hot.dispose(() => state.dispose());
}

import { schema, schemas, inspect, snapshot, dump } from "@dylanebert/shallot/ecs/core";
Object.assign(window, {
    schema,
    schemas,
    inspect,
    snapshot,
    dump,
});
