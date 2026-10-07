import { plugin } from "bun";
import { oracleWasm } from "./oracle-kernel";

// Loaded before the subjects, so their public API uses the oracle build rather than committed bytes.
plugin({
    name: "box3d-oracle-kernel",
    setup(build) {
        build.onResolve({ filter: /(?:^|\/)kernel\.wasm(?:\.ts)?$/ }, () => ({
            path: "kernel",
            namespace: "box3d-oracle",
        }));
        build.onLoad({ filter: /^kernel$/, namespace: "box3d-oracle" }, () => ({
            contents: `export const KERNEL_WASM_BASE64 = ${JSON.stringify(Buffer.from(oracleWasm).toString("base64"))};`,
            loader: "ts",
        }));
    },
});
