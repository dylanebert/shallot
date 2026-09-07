import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFileSync, realpathSync, writeFileSync } from "node:fs";
import { createRequire } from "node:module";
import { dirname, join } from "node:path";

// Copied into the installed consumer: every product import resolves there, never in this checkout.
const require = createRequire(import.meta.url);
const engine = dirname(dirname(require.resolve("@dylanebert/shallot")));
const loader = join(engine, "bin/bun-native.ts");
const mode = process.argv[2] ?? "acquire";
const watchdog = setTimeout(() => {
    console.error("native fixture watchdog");
    process.exit(2);
}, 4000);
try {
    const { loadNative } = await import(loader);
    const native = await loadNative();
    if (mode === "foreign") {
        const foreign = {};
        Object.defineProperty(navigator, "gpu", { value: foreign, configurable: true });
        await assert.rejects(native.setupGlobals(), /refuses an existing foreign GPU instance/);
        assert.equal(navigator.gpu, foreign);
        console.log("FOREIGN_REFUSED_UNCHANGED");
    } else if (mode === "override") {
        assert.throws(
            () => native.createGPUInstance("/not/the/peer/library"),
            /library override is unsupported/,
        );
        await assert.rejects(
            native.setupGlobals({ libPath: "/not/the/peer/library" }),
            /library override is unsupported/,
        );
        console.log("OVERRIDE_REFUSED");
    } else {
        await native.setupGlobals();
        const gpu = navigator.gpu as any;
        const constructors = Object.entries(native.globalConstructors);
        assert.equal(constructors.length, 13, "complete carried global table");
        assert.equal(typeof native.globalConstructors.GPUDevice, "function");
        const gpuConstructor = gpu.constructor;
        assert.equal(typeof gpuConstructor, "function");
        await native.setupGlobals();
        const again = await loadNative();
        await again.setupGlobals();
        assert.equal(navigator.gpu, gpu);
        assert.equal(gpu.constructor, gpuConstructor);
        assert.equal(again.globalConstructors, native.globalConstructors);
        for (const [name, value] of constructors) {
            assert.equal((globalThis as any)[name], value, name);
        }
        const { BASE_FEATURES, deviceLimits } = await import(
            join(engine, "src/engine/runtime/gpu.ts")
        );
        const adapter = await gpu.requestAdapter();
        assert(adapter, "actual packed adapter");
        try {
            const device = await adapter.requestDevice({
                label: "packed-native",
                defaultQueue: { label: "packed-native-queue" },
                requiredFeatures: [...BASE_FEATURES],
                requiredLimits: deviceLimits(adapter.limits),
            });
            try {
                for (const feature of BASE_FEATURES) assert(device.features.has(feature));
                assert.equal(device.limits.maxStorageBuffersPerShaderStage, 10);
                const peer = realpathSync(createRequire(loader).resolve("bun-webgpu"));
                const platform = realpathSync(
                    createRequire(peer).resolve(
                        `bun-webgpu-${process.platform}-${process.arch}/index.ts`,
                    ),
                );
                const library = realpathSync((await import(platform)).default);
                const js = realpathSync(join(engine, "dist/native.js"));
                const hash = (file: string) =>
                    createHash("sha256").update(readFileSync(file)).digest("hex");
                const mapping =
                    process.platform === "darwin"
                        ? Bun.spawnSync(["vmmap", "-w", String(process.pid)], {
                              stdout: "pipe",
                              stderr: "pipe",
                          })
                        : null;
                const maps = mapping
                    ? `${mapping.stdout.toString()}\n${mapping.stderr.toString()}`
                    : readFileSync(`/proc/${process.pid}/maps`, "utf8");
                writeFileSync("native-maps.txt", maps);
                assert(!mapping || mapping.exitCode === 0, "OS mapped-library command");
                assert(maps.includes(library), "resolved peer library is OS-mapped in this child");
                console.log(
                    JSON.stringify({
                        pid: process.pid,
                        runtime: Bun.version,
                        platform: process.platform,
                        arch: process.arch,
                        engine: realpathSync(engine),
                        loader: realpathSync(loader),
                        loaderHash: hash(loader),
                        js,
                        jsHash: hash(js),
                        peer,
                        peerHash: hash(peer),
                        peerVersion: JSON.parse(
                            readFileSync(join(dirname(peer), "package.json"), "utf8"),
                        ).version,
                        platformEntry: platform,
                        platformVersion: JSON.parse(
                            readFileSync(join(dirname(platform), "package.json"), "utf8"),
                        ).version,
                        library,
                        libraryHash: hash(library),
                        identity: true,
                    }),
                );
            } finally {
                device.destroy();
                await device.lost;
                await Promise.resolve();
                assert.equal(device._acquisition.handles.size, 0);
            }
        } finally {
            adapter.destroy();
            gpu.destroy();
        }
        assert.equal(gpu._ticker.acquisitions.size, 0);
        console.log("PACKED_NATIVE_ACQUIRED_DRAINED");
    }
} finally {
    clearTimeout(watchdog);
}
