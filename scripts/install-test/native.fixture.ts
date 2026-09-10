import assert from "node:assert/strict";
import { createRequire } from "node:module";
import { dirname, join } from "node:path";

// Copied into the installed consumer: every product import resolves there, never in this checkout.
const require = createRequire(import.meta.url);
const engine = dirname(dirname(require.resolve("@dylanebert/shallot")));
const watchdog = setTimeout(() => {
    console.error("native fixture watchdog");
    process.exit(2);
}, 4000);
try {
    const { loadNative } = await import(join(engine, "bin/bun-native.ts"));
    const native = await loadNative();
    await native.setupGlobals();
    const gpu = navigator.gpu as any;
    const { BASE_FEATURES, deviceLimits } = await import(join(engine, "src/engine/runtime/gpu.ts"));
    const adapter = await gpu.requestAdapter();
    assert(adapter, "native adapter");
    try {
        const device = await adapter.requestDevice({
            label: "installed-native",
            requiredFeatures: [...BASE_FEATURES],
            requiredLimits: deviceLimits(adapter.limits),
        });
        try {
            for (const feature of BASE_FEATURES) assert(device.features.has(feature));
            assert.equal(device.limits.maxStorageBuffersPerShaderStage, 10);
        } finally {
            device.destroy();
        }
    } finally {
        adapter.destroy();
        gpu.destroy();
    }
    console.log("NATIVE_ACQUIRED");
} finally {
    clearTimeout(watchdog);
}
