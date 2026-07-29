import tgpu, { type Namespace } from "typegpu";
import * as d from "typegpu/data";
import { isBeingTranspiled } from "typegpu/std";

// The escape vocabulary: WGSL constructs TGSL has no binding for. Each is a typed leaf — the smallest
// function that reaches the construct — so a call site stays one idiom and never splices WGSL text.
//
// Two shapes, and the choice is forced by whether the leaf can run on the CPU:
//
//   1. **Dual.** `isBeingTranspiled() ? <raw leaf>(args) : <plain JS>` — a ternary, never an `if`. The
//      condition folds at shader-gen time and ONLY the taken arm is transpiled, so the CPU arm may use
//      anything JS (`Math.round`, `&`, `<<`), and the GPU arm emits the native intrinsic. An `if`
//      statement does not fold: both arms are transpiled (the untaken one then fails on the first
//      non-TGSL expression) and the emitted WGSL carries dead code after the `return`.
//   2. **GPU-only.** A `tgpu.fn` whose implementation is a WGSL string. Not callable on the CPU (typegpu
//      throws), which is correct for a construct with no CPU meaning — a workgroup pointer, an atomic.
//
// The raw arms are named `*Wgsl` and stay module-private: both they and their dual wrapper are emitted
// into the resolved WGSL, and the wrapper owns the public name.
//
// Four of these are upstream-PR candidates (the 2×16 packs); when typegpu binds them in `typegpu/std`,
// delete the leaf and import theirs. `typegpu/std`'s own CPU implementations are not automatically
// trustworthy: `std.pack4x8unorm` truncates instead of rounding and never clamps, so it disagrees with
// the WGSL intrinsic it names (asserted in tgsl.test.ts) — which is why the 8-bit pack is a leaf here
// too, rather than a re-export.

const packSnorm2x16Wgsl = tgpu.fn(
    [d.vec2f],
    d.u32,
)(/* wgsl */ `(v: vec2f) -> u32 { return pack2x16snorm(v); }`);

const unpackSnorm2x16Wgsl = tgpu.fn(
    [d.u32],
    d.vec2f,
)(/* wgsl */ `(e: u32) -> vec2f { return unpack2x16snorm(e); }`);

const packUnorm2x16Wgsl = tgpu.fn(
    [d.vec2f],
    d.u32,
)(/* wgsl */ `(v: vec2f) -> u32 { return pack2x16unorm(v); }`);

const unpackUnorm2x16Wgsl = tgpu.fn(
    [d.u32],
    d.vec2f,
)(/* wgsl */ `(e: u32) -> vec2f { return unpack2x16unorm(e); }`);

const packUnorm4x8Wgsl = tgpu.fn(
    [d.vec4f],
    d.u32,
)(/* wgsl */ `(v: vec4f) -> u32 { return pack4x8unorm(v); }`);

const bitcastF32toU32Wgsl = tgpu.fn(
    [d.f32],
    d.u32,
)(/* wgsl */ `(v: f32) -> u32 { return bitcast<u32>(v); }`);

const idivWgsl = tgpu.fn(
    [d.u32, d.u32],
    d.u32,
)(/* wgsl */ `(a: u32, b: u32) -> u32 { return a / b; }`);

// The lattices themselves, shared by each leaf's CPU arm and by the hot-path mirrors in encode.ts: one
// definition of where a quantized value lands, whatever calls it. Each rounds its input to f32 first —
// a vector schema stores f32 and so does the GPU, so quantizing an f64 value would put the rare
// half-way case on the other lattice point from the shader. One residual CPU↔GPU seam: `Math.round` is
// half-up and WGSL's `round` is half-to-even, so a product that lands exactly on a lattice midpoint
// (x.5 after scaling) differs by 1 LSB between the two arms.
/** @internal */
export const snorm16 = (v: number): number =>
    Math.round(Math.max(-1, Math.min(1, Math.fround(v))) * 32767) & 0xffff;
/** @internal */
export const unorm16 = (v: number): number =>
    Math.round(Math.max(0, Math.min(1, Math.fround(v))) * 65535) & 0xffff;
/** @internal */
export const unorm8 = (v: number): number =>
    Math.round(Math.max(0, Math.min(1, Math.fround(v))) * 255) & 0xff;

const _bits = new ArrayBuffer(4);
const _f32 = new Float32Array(_bits);
const _u32 = new Uint32Array(_bits);

function f32Bits(v: number): number {
    _f32[0] = v;
    return _u32[0];
}

/** pack two [-1,1] lanes into an snorm16x2 `u32` (lane x → low 16 bits): WGSL `pack2x16snorm`.
 *  The (-1,1) ↔ (-32767,32767) lattice puts 0 ↔ 0 and ±1 ↔ ±32767 on exact rails.
 *  @example const enc = packSnorm2x16(vec2f(0, -1)); // 0x80010000 */
export const packSnorm2x16 = tgpu.fn(
    [d.vec2f],
    d.u32,
)((v) => {
    "use gpu";
    return isBeingTranspiled() ? packSnorm2x16Wgsl(v) : ((snorm16(v.y) << 16) | snorm16(v.x)) >>> 0;
});

/** unpack an snorm16x2 `u32` to two [-1,1] lanes (low 16 bits → x): WGSL `unpack2x16snorm`, the inverse
 *  of {@link packSnorm2x16}.
 *  @example const v = unpackSnorm2x16(enc); // vec2f */
export const unpackSnorm2x16 = tgpu.fn(
    [d.u32],
    d.vec2f,
)((e) => {
    "use gpu";
    return isBeingTranspiled()
        ? unpackSnorm2x16Wgsl(e)
        : d.vec2f(Math.max(-1, ((e << 16) >> 16) / 32767), Math.max(-1, (e >> 16) / 32767));
});

/** pack two [0,1] lanes into a unorm16x2 `u32` (lane x → low 16 bits): WGSL `pack2x16unorm`. Uniform
 *  1.5e-5 spacing across the range (gpu.md rule 6), so it beats f16 in any bounded range.
 *  @example const enc = packUnorm2x16(vec2f(1, 0)); // 0x0000ffff */
export const packUnorm2x16 = tgpu.fn(
    [d.vec2f],
    d.u32,
)((v) => {
    "use gpu";
    return isBeingTranspiled() ? packUnorm2x16Wgsl(v) : ((unorm16(v.y) << 16) | unorm16(v.x)) >>> 0;
});

/** unpack a unorm16x2 `u32` to two [0,1] lanes (low 16 bits → x): WGSL `unpack2x16unorm`, the inverse
 *  of {@link packUnorm2x16}.
 *  @example const v = unpackUnorm2x16(enc); // vec2f */
export const unpackUnorm2x16 = tgpu.fn(
    [d.u32],
    d.vec2f,
)((e) => {
    "use gpu";
    return isBeingTranspiled()
        ? unpackUnorm2x16Wgsl(e)
        : d.vec2f((e & 0xffff) / 65535, (e >>> 16) / 65535);
});

/** pack four [0,1] lanes into a `u32` of four unorm8 bytes (lane x → byte 0): WGSL `pack4x8unorm`.
 *  Rounds and clamps, exactly like the intrinsic — unlike `typegpu/std`'s CPU implementation, which
 *  truncates and wraps. Named lane-count-first like its 2×16 siblings, and **not** `pack4x8unorm`: an
 *  authored name that collides with a WGSL builtin resolves to `pack4x8unorm_1` even under strict
 *  naming, so a raw splice site could not call it (asserted in tgsl.test.ts).
 *  @example const enc = packUnorm4x8(vec4f(1, 0, 0.5, 1)); // 0xff8000ff */
export const packUnorm4x8 = tgpu.fn(
    [d.vec4f],
    d.u32,
)((v) => {
    "use gpu";
    return isBeingTranspiled()
        ? packUnorm4x8Wgsl(v)
        : (unorm8(v.x) | (unorm8(v.y) << 8) | (unorm8(v.z) << 16) | (unorm8(v.w) << 24)) >>> 0;
});

/** reinterpret an `f32`'s bits as a `u32`: WGSL `bitcast<u32>(e)`. The f32→u32 direction only;
 *  `typegpu/std` binds the u32→f32 one as `bitcastU32toF32`.
 *  @example const bits = bitcastF32toU32(1); // 0x3f800000 */
export const bitcastF32toU32 = tgpu.fn(
    [d.f32],
    d.u32,
)((v) => {
    "use gpu";
    return isBeingTranspiled() ? bitcastF32toU32Wgsl(v) : f32Bits(v);
});

/** integer division. TGSL emits float `/` for integer operands, which is silently wrong above 2²⁴ —
 *  reach for this wherever the operand range isn't provably small (gpu.md; the BVH bounds path is the
 *  live case). A zero divisor is the one input the two arms disagree on: WGSL's `u32` `/` returns the
 *  dividend, the CPU arm returns `Infinity`. Don't divide by a value that can be zero.
 *  @example const lane = idiv(index, 32); */
export const idiv = tgpu.fn(
    [d.u32, d.u32],
    d.u32,
)((a, b) => {
    "use gpu";
    return isBeingTranspiled() ? idivWgsl(a, b) : Math.floor(a / b);
});

/** WGSL `workgroupUniformLoad(p)`: a control barrier whose result the uniformity analysis treats as
 *  uniform, which is what makes a `workgroupBarrier` inside a flag-gated loop legal (the decoupled-
 *  fallback scan's early-exit, gpu.md "the decoupled-scan exception"). GPU-only — a workgroup pointer
 *  has no CPU meaning.
 *  @example if (uniformLoad(&flag) == 1u) { break; } */
export const uniformLoad = tgpu.fn(
    [d.ptrWorkgroup(d.u32)],
    d.u32,
)(/* wgsl */ `(p: ptr<workgroup, u32>) -> u32 { return workgroupUniformLoad(p); }`);

/** WGSL `atomicCompareExchangeWeak(p, cmp, val).old_value`: the compare-and-swap a chained scan's
 *  publish/claim step needs. Returns the prior value — equal to `cmp` exactly when the exchange took.
 *  GPU-only.
 *  @example const prev = compareExchange(&slot, 0, claim); */
export const compareExchange = tgpu.fn(
    [d.ptrStorage(d.atomic(d.u32), "read-write"), d.u32, d.u32],
    d.u32,
)(
    /* wgsl */ `(p: ptr<storage, atomic<u32>, read_write>, cmp: u32, val: u32) -> u32 {
    return atomicCompareExchangeWeak(p, cmp, val).old_value;
}`,
);

/**
 * a lazily-resolved, memoized WGSL chunk: `chunk(...)` returns the thunk, and the first call resolves
 * the given TGSL resources under `names: "strict"` so the emitted function names are the authored ones
 * and a raw-WGSL splice site can call them.
 *
 * Lazy on purpose. Resolution needs the build transform, so resolving at module scope would make
 * *importing* a codec module fail in any tool that never touches the GPU (the scene formatter is the
 * live case). Pass a shared `ns` when two chunks are always spliced together and must not both emit a
 * shared dependency — the namespace emits it into whichever chunk resolves first.
 *
 * The rethrow names the likely cause: without the plugin the failure is otherwise an opaque resolution
 * error from deep inside typegpu.
 * @internal
 */
export function chunk(
    label: string,
    items: Parameters<typeof tgpu.resolve>[0],
    ns?: Namespace,
): () => string {
    let wgsl: string | undefined;
    return () => {
        if (wgsl !== undefined) return wgsl;
        try {
            wgsl = tgpu.resolve(items as never, { names: ns ?? "strict" });
        } catch (cause) {
            throw new Error(
                `the "${label}" WGSL chunk failed to resolve — most likely this bundle was built ` +
                    "without the typegpu transform (`unplugin-typegpu`); see `checkTgsl` in " +
                    "engine/runtime/gpu.ts",
                { cause },
            );
        }
        return wgsl;
    };
}

/** the module-scope `diagnostic(off, subgroup_uniformity);` directive — the sanctioned opt-out for
 *  WGSL's uniformity analysis rejecting a subgroup op inside a loop whose condition derives from a
 *  subgroup reduction (gpu.md "Subgroup ops in data-dependent loops"). **Always pair it with a fixed
 *  iteration cap**, so a logic error degrades to a wrong result instead of a GPU watchdog hang.
 *  Not a function: pass it alongside the kernel being resolved, since `$uses` is unavailable on a
 *  function whose metadata came from `unplugin-typegpu`.
 *  @example tgpu.resolve([subgroupUniformityOff, myKernel]) */
export const subgroupUniformityOff = tgpu["~unstable"].declare(
    "diagnostic(off, subgroup_uniformity);",
);
