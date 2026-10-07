import { mock } from "bun:test";
import { oracleWasm } from "./oracle-kernel";

// Replace the module in this test process without rewriting imports in the transpiler cache.
mock.module("../../src/standard/physics/kernel/kernel.wasm.ts", () => ({
    KERNEL_WASM_BASE64: Buffer.from(oracleWasm).toString("base64"),
}));
