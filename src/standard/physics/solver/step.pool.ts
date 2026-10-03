import { init } from "../kernel/kernel";

// Initialize before corpus helpers can lazily acquire the standalone kernel.
await init(undefined, { threads: 4 });
