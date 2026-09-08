// Builds the Rust physics kernel to wasm-simd128 and inlines it as base64 into
// src/standard/tumble/engine/. Inlining (not a separate .wasm asset) keeps the engine pure JS — no
// asset-path resolution for downstream bundlers, identical in browser / bun / node / deno. The
// generated files are committed so `bun test` and `bun run build` work without a Rust toolchain;
// regenerate here after any rust/tumble/ change.
//
// Two artifacts, from the same source:
//
//   engine/kernel.wasm.ts        the single-thread default. Stable toolchain, exported memory. Every
//                                host gets this; it is the shipping path and must never move.
//   engine/kernel.shared.wasm.ts the multithreaded artifact. Needs +atomics, which
//                                LLD only accepts alongside --shared-memory, and a shared memory can
//                                only be instantiated under cross-origin isolation — so one module
//                                genuinely cannot serve both. std is rebuilt with atomics (nightly
//                                -Zbuild-std); the kernel touches no std API, so this is a toolchain
//                                switch, not a code change. Loaded behind a dynamic import
//                                (engine/kernel.ts), so single-thread consumers never parse it.
//
// The shared build exports the two globals + the TLS initializer each worker needs to bootstrap its own
// stack slice (LLD's start function only assigns __tls_base on the CAS winner). It gets its own
// --target-dir so the two builds don't thrash one target/.
//
// Usage: bun run scripts/build-tumble-kernel.ts

import { spawnSync } from "node:child_process";
import { readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";

const pkgRoot = resolve(import.meta.dir, "..");
const kernelDir = resolve(pkgRoot, "rust/tumble");
const engineDir = resolve(pkgRoot, "src/standard/tumble/engine");

/** Shadow-stack size of the shared build (link arg). Partitioned into per-thread slices by the pool
 * (src/pool.ts); raising it raises the thread ceiling and the module's declared page count. */
const SHARED_STACK_SIZE = 3145728;
/** Link-time memory ceiling of the shared build, in bytes (16384 pages). A JS-created shared memory may
 * only lower it — the import's declared maximum is the hard cap. */
const SHARED_MAX_MEMORY = 1073741824;

const SHARED_RUSTFLAGS = [
    "-C target-feature=+simd128,+atomics,+bulk-memory,+mutable-globals",
    "-C link-arg=--import-memory",
    "-C link-arg=--shared-memory",
    `-C link-arg=--max-memory=${SHARED_MAX_MEMORY}`,
    `-C link-arg=-zstack-size=${SHARED_STACK_SIZE}`,
    "-C link-arg=--export=__stack_pointer",
    "-C link-arg=--export=__tls_base",
    "-C link-arg=--export=__tls_size",
    "-C link-arg=--export=__tls_align",
    "-C link-arg=--export=__wasm_init_tls",
].join(" ");

function build(args: string[], env: Record<string, string> | undefined, targetDir: string): Buffer {
    console.log(`[build-kernel] cargo ${args.join(" ")}`);
    const r = spawnSync("cargo", args, {
        cwd: kernelDir,
        stdio: "inherit",
        env: env ? { ...process.env, ...env } : process.env,
    });
    if (r.status !== 0) process.exit(r.status ?? 1);
    return readFileSync(
        resolve(kernelDir, targetDir, "wasm32-unknown-unknown/release/tumble_kernel.wasm"),
    );
}

// The imported memory's declared limits, in pages. A JS-created shared memory must be at least the
// module's `initial` (the module owns its data + stack layout, so it declares the floor) and at most its
// `maximum`. Read them out of the binary rather than deriving them from the link args — the floor moves
// with the data section, and a stale constant would fail at instantiation, off in a worker.
function memoryImportLimits(wasm: Buffer): { initial: number; maximum: number } {
    let p = 8; // magic + version
    const u8 = () => wasm[p++];
    const uleb = () => {
        let x = 0;
        let shift = 0;
        for (;;) {
            const b = u8();
            x |= (b & 0x7f) << shift;
            if ((b & 0x80) === 0) return x >>> 0;
            shift += 7;
        }
    };
    const skipName = () => {
        // `p += uleb()` would drop uleb's own advance of `p` — the compound assignment reads `p` before
        // calling the RHS.
        const n = uleb();
        p += n;
    };
    const limits = () => {
        const flags = uleb();
        const initial = uleb();
        const maximum = flags & 0x01 ? uleb() : 0;
        return { initial, maximum };
    };
    while (p < wasm.length) {
        const id = u8();
        const size = uleb();
        const end = p + size;
        if (id !== 2) {
            p = end;
            continue;
        }
        for (let i = uleb(); i > 0; --i) {
            skipName();
            skipName();
            const kind = u8();
            if (kind === 0x02) return limits();
            if (kind === 0x00) uleb();
            else if (kind === 0x01) {
                u8();
                limits();
            } else if (kind === 0x03) {
                u8();
                u8();
            } else throw new Error(`unknown import kind ${kind}`);
        }
        p = end;
    }
    throw new Error("no imported memory in the shared kernel — check --import-memory");
}

function emit(path: string, contents: string, wasm: number, base64: number): void {
    writeFileSync(path, contents);
    console.log(`[build-kernel] wrote ${path} (${wasm} B wasm -> ${base64} B base64)`);
}

const st = build(["build", "--release", "--target", "wasm32-unknown-unknown"], undefined, "target");
const stBase64 = st.toString("base64");
emit(
    resolve(engineDir, "kernel.wasm.ts"),
    `// Generated by scripts/build-tumble-kernel.ts from rust/tumble/ — do not edit. Regenerate: bun run scripts/build-tumble-kernel.ts
// wasm-simd128 physics kernel, base64-inlined (${st.length} bytes).
export const KERNEL_WASM_BASE64 =
    "${stBase64}";
`,
    st.length,
    stBase64.length,
);

const shared = build(
    [
        "+nightly",
        "build",
        "--release",
        "--target",
        "wasm32-unknown-unknown",
        "-Z",
        "build-std=std,panic_abort",
        // The staged solve (kernel/src/solve.rs): the shared artifact alone, so the single-thread one
        // doesn't carry the stage/block tables it can never run.
        "--features",
        "mt",
        "--target-dir",
        "target-shared",
    ],
    { RUSTFLAGS: SHARED_RUSTFLAGS },
    "target-shared",
);
const sharedBase64 = shared.toString("base64");
const { initial, maximum } = memoryImportLimits(shared);
emit(
    resolve(engineDir, "kernel.shared.wasm.ts"),
    `// Generated by scripts/build-tumble-kernel.ts from rust/tumble/ — do not edit. Regenerate: bun run scripts/build-tumble-kernel.ts
// wasm-simd128 physics kernel, shared-memory build (${shared.length} bytes), base64-inlined. Loaded
// behind a dynamic import so single-thread consumers never parse it.
export const KERNEL_SHARED_WASM_BASE64 =
    "${sharedBase64}";

/** Pages the module's imported memory declares as its minimum — the floor for the JS-created memory. */
export const SHARED_INITIAL_PAGES = ${initial};
/** Pages the module's imported memory declares as its maximum — the ceiling (a JS memory may only lower it). */
export const SHARED_MAX_PAGES = ${maximum};
/** Bytes of shadow stack the module was linked with (-zstack-size), partitioned into per-thread slices. */
export const SHARED_STACK_SIZE = ${SHARED_STACK_SIZE};
`,
    shared.length,
    sharedBase64.length,
);
