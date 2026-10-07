// Builds the Rust physics kernel to wasm-simd128 and inlines it as base64 into
// src/standard/physics/kernel/. Inlining (not a separate .wasm asset) keeps the engine pure JS — no
// asset-path resolution for downstream bundlers, identical in browser / bun / node / deno. The
// generated files are committed so tests and package consumers run without a Rust toolchain;
// `bun run build` regenerates them, as does running this after any crates/physics/ change.
//
// Two artifacts, from the same source:
//
//   engine/kernel.wasm.ts        the single-thread default. Pinned stable toolchain, exported memory. Every
//                                host gets this; it is the shipping path and must never move.
//   engine/kernel.shared.wasm.ts the multithreaded artifact. Needs +atomics, which
//                                LLD only accepts alongside --shared-memory, and a shared memory can
//                                only be instantiated under cross-origin isolation — so one module
//                                genuinely cannot serve both. std is rebuilt with atomics (nightly
//                                -Zbuild-std, on the dated NIGHTLY); the kernel touches no std API, so this is a toolchain
//                                switch, not a code change. Loaded behind a dynamic import
//                                (engine/kernel.ts), so single-thread consumers never parse it.
//
// The shared build exports the two globals + the TLS initializer each worker needs to bootstrap its own
// stack slice (LLD's start function only assigns __tls_base on the CAS winner). It gets its own
// --target-dir so the two builds don't thrash one target/.
//
// Toolchains are pinned so the committed bytes reproduce: stable comes from `rust-toolchain.toml`, the
// shared build uses the dated NIGHTLY below, and both pass through the pinned wasm-opt.
//
// Usage: bun run crates/physics/scripts/build-kernel.ts [--checked]
//
// `--checked` keeps debug assertions and overflow checks in both release builds: `Col` bounds and std's
// unsafe preconditions then panic, and the panic hook prints the message before the abort traps. CI tests
// physics against it; a checked artifact is never committed.

import { spawn, spawnSync } from "node:child_process";
import { readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { optimize, version } from "../../../scripts/wasm-opt";
import { kernelInputHash } from "./kernel-inputs";

const pkgRoot = resolve(import.meta.dir, "../../..");
const kernelDir = resolve(pkgRoot, "crates/physics");
const engineDir = resolve(pkgRoot, "src/standard/physics/kernel");

/** The dated nightly for the shared build; needs rust-src for -Zbuild-std. */
const NIGHTLY = "nightly-2026-09-10";
const INSTALL = `rustup toolchain install ${NIGHTLY} --component rust-src --target wasm32-unknown-unknown`;

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
    // RUSTFLAGS replaces .cargo/config.toml's, so repeat its workspace-prefix remap (see there).
    "--remap-path-prefix=crates/physics/=",
].join(" ");

async function build(
    args: string[],
    env: Record<string, string> | undefined,
    targetDir: string,
): Promise<Buffer> {
    console.log(`[build-kernel] cargo ${args.join(" ")}`);
    const status = await new Promise<number | null>((done) =>
        spawn("cargo", args, {
            cwd: kernelDir,
            stdio: "inherit",
            env: env ? { ...process.env, ...env } : process.env,
        }).on("close", done),
    );
    if (status !== 0) process.exit(status ?? 1);
    const raw = resolve(
        kernelDir,
        targetDir,
        "wasm32-unknown-unknown/release/shallot_physics.wasm",
    );
    const opt = resolve(kernelDir, targetDir, "shallot_physics.opt.wasm");
    await optimize(raw, opt);
    return readFileSync(opt);
}

function rustc(toolchain?: string): string {
    const args = toolchain ? [`+${toolchain}`, "--version"] : ["--version"];
    const r = spawnSync("rustc", args, { cwd: kernelDir, encoding: "utf8" });
    if (r.status !== 0) throw new Error(`[build-kernel] rustc ${args.join(" ")} failed`);
    return r.stdout.trim();
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

const checked = process.argv.includes("--checked");
if (checked) {
    process.env.CARGO_PROFILE_RELEASE_DEBUG_ASSERTIONS = "true";
    process.env.CARGO_PROFILE_RELEASE_OVERFLOW_CHECKS = "true";
}
const inputHash = `${kernelInputHash(pkgRoot)}${checked ? "; checked, never commit" : ""}`;
const opt = `${await version()}${checked ? "; checked, never commit" : ""}`;
const probe = spawnSync("rustup", ["component", "list", "--installed", "--toolchain", NIGHTLY], {
    encoding: "utf8",
});
if (probe.status !== 0 || !/^rust-src/m.test(probe.stdout)) {
    console.error(`[build-kernel] ${NIGHTLY} with rust-src is missing. Install it: ${INSTALL}`);
    process.exit(1);
}
const sharedBuild = build(
    [
        `+${NIGHTLY}`,
        "build",
        "--release",
        "--target",
        "wasm32-unknown-unknown",
        "-Z",
        "build-std=std,panic_abort",
        "--target-dir",
        "target-shared",
    ],
    { RUSTFLAGS: SHARED_RUSTFLAGS },
    "target-shared",
);
// Each build is one LTO unit on one core, so the two run side by side.
const [st, shared] = await Promise.all([
    build(
        ["build", "--release", "--target", "wasm32-unknown-unknown"],
        undefined,
        // A workspace member: the stable build lands in the workspace root target/.
        resolve(pkgRoot, "target"),
    ),
    sharedBuild,
]);
const layoutModule = new WebAssembly.Module(new Uint8Array(st));
const layoutImports: WebAssembly.Imports = {};
for (const { module, name, kind } of WebAssembly.Module.imports(layoutModule)) {
    if (kind !== "function")
        throw new Error(`unsupported layout import: ${module}.${name} (${kind})`);
    (layoutImports[module] ??= {})[name] = () => {
        throw new Error(`layout discovery called ${module}.${name}`);
    };
}
const layoutKernel = new WebAssembly.Instance(layoutModule, layoutImports);
const recordOffset = layoutKernel.exports.contactRecordOffset as (index: number) => number;
const layoutNames = [
    "DIR_STRIDE",
    "DIR_COUNT",
    "DIR_BLOCK",
    "DIR_FLAGS",
    "DIR_FRICTION",
    "DIR_RESTITUTION",
    "DIR_ROLLING_RESISTANCE",
    "DIR_TANGENT_VELOCITY",
    "MANIFOLD_STRIDE",
    "M_NORMAL",
    "M_TWIST",
    "M_FRICTION",
    "M_ROLLING",
    "M_POINT_COUNT",
    "M_POINTS",
    "POINT_STRIDE",
    "P_ANCHOR_A",
    "P_ANCHOR_B",
    "P_SEPARATION",
    "P_BASE_SEPARATION",
    "P_NORMAL_IMPULSE",
    "P_TOTAL_NORMAL_IMPULSE",
    "P_NORMAL_VELOCITY",
    "P_FEATURE_ID",
    "P_TRIANGLE_INDEX",
    "P_PERSISTED",
];
const fieldNames = [
    "flags",
    "manifoldCount",
    "bodySimIndexA",
    "bodySimIndexB",
    "setIndex",
    "colorIndex",
    "localIndex",
    "bodyIdA",
    "prevKeyA",
    "nextKeyA",
    "bodyIdB",
    "prevKeyB",
    "nextKeyB",
    "shapeIdA",
    "shapeIdB",
    "childIndex",
    "islandId",
    "islandIndex",
    "contactId",
    "generation",
];
writeFileSync(
    resolve(engineDir, "contact-layout.ts"),
    [
        "// Generated from Rust resident record offsets by build-kernel.ts.",
        ...layoutNames.map((name, i) => `export const ${name} = ${recordOffset(i)};`),
        "export const ContactField = {",
        ...fieldNames.map((name, i) => `    ${name}: ${recordOffset(layoutNames.length + i)},`),
        "} as const;",
        "",
    ].join("\n"),
);
const jointNamePtr = layoutKernel.exports.jointLayoutNamePtr as (index: number) => number;
const jointNameLen = layoutKernel.exports.jointLayoutNameLen as (index: number) => number;
const jointOffset = layoutKernel.exports.jointLayoutOffset as (index: number) => number;
const jointMemory = (layoutKernel.exports.memory as WebAssembly.Memory).buffer;
const jointLayout = ["// Generated from Rust joint-sim offsets by build-kernel.ts."];
for (let i = 0; ; ++i) {
    const ptr = jointNamePtr(i);
    if (ptr === 0) break;
    const name = new TextDecoder().decode(new Uint8Array(jointMemory, ptr, jointNameLen(i)));
    jointLayout.push(`export const ${name} = ${jointOffset(i)};`);
}
writeFileSync(resolve(engineDir, "joint-layout.ts"), `${jointLayout.join("\n")}\n`);
const stBase64 = st.toString("base64");
emit(
    resolve(engineDir, "kernel.wasm.ts"),
    `// Generated by crates/physics/scripts/build-kernel.ts from crates/physics/ — do not edit. Regenerate: bun run crates/physics/scripts/build-kernel.ts
// Build inputs: sha256:${inputHash}
// Toolchain: ${rustc()}; ${opt}.
// wasm-simd128 physics kernel, base64-inlined (${st.length} bytes).
export const KERNEL_WASM_BASE64 =
    "${stBase64}";
`,
    st.length,
    stBase64.length,
);

const sharedBase64 = shared.toString("base64");
const { initial, maximum } = memoryImportLimits(shared);
emit(
    resolve(engineDir, "kernel.shared.wasm.ts"),
    `// Generated by crates/physics/scripts/build-kernel.ts from crates/physics/ — do not edit. Regenerate: bun run crates/physics/scripts/build-kernel.ts
// Build inputs: sha256:${inputHash}
// Toolchain: ${rustc(NIGHTLY)}; ${opt}.
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
