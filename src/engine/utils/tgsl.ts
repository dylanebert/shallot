import tgpu, { type Namespace, type TgpuVar } from "typegpu";
import * as d from "typegpu/data";
import { isBeingTranspiled } from "typegpu/std";

// The escape vocabulary: WGSL constructs neither TGSL nor `typegpu/std` binds. Each is a typed leaf, the
// smallest function that reaches the construct, so a call site stays one idiom and never splices WGSL
// text. A construct `typegpu/std` binds, such as the pack builtins or `bitcastF32toU32`, is imported
// from there.
//
// Two shapes, and the choice is forced by whether the leaf can run on the CPU:
//
//   1. **Dual.** `isBeingTranspiled() ? <raw leaf>(args) : <plain JS>` — the condition folds at
//      shader-gen time and ONLY the taken arm is transpiled, so the CPU arm may use anything JS
//      (`Math.floor`), and the GPU arm emits the native construct. The raw arm is named `*Wgsl` and
//      stays module-private: both it and its dual wrapper are emitted into the resolved WGSL, and the
//      wrapper owns the public name.
//   2. **GPU-only.** Raw WGSL with no CPU meaning, such as a workgroup pointer. Not callable on the CPU.

const idivWgsl = tgpu.fn(
    [d.u32, d.u32],
    d.u32,
)(/* wgsl */ `(a: u32, b: u32) -> u32 { return a / b; }`);

/** integer division — the only correct way to divide integers in TGSL. `a / b` on integer operands
 *  transpiles to `f32(a) / f32(b)`: a *fractional* quotient, so it is wrong for any inexact division
 *  (not merely above 2²⁴, where even an exact quotient loses bits). Audit every `/` in a ported kernel.
 *  A zero divisor is the one input the two arms disagree on: WGSL's `u32` `/` returns the dividend, the
 *  CPU arm returns `Infinity`. Don't divide by a value that can be zero. */
export const idiv = tgpu.fn(
    [d.u32, d.u32],
    d.u32,
)((a, b) => {
    "use gpu";
    return isBeingTranspiled() ? idivWgsl(a, b) : Math.floor(a / b);
});

/** WGSL `workgroupUniformLoad(&v)`: a control barrier whose result the uniformity analysis treats as
 *  uniform, which is what makes a `workgroupBarrier` inside a flag-gated loop legal (the decoupled-
 *  fallback scan's early-exit). GPU-only — a workgroup variable
 *  has no CPU meaning. Bind one loader per variable at module scope and read it as `.$` at the call
 *  site, which emits the intrinsic inline against that variable.
 *
 *  Inline rather than a shared `fn(p: ptr<workgroup, u32>)` leaf: WGSL 1.0 forbids workgroup-address-
 *  space pointer parameters, and only Tint admits them (`unrestricted_pointer_parameters`). naga —
 *  Firefox's front end — rejects such a function outright, so the pointer-parameter spelling made every
 *  shader reaching it uncompilable off Chromium. */
export const uniformLoad = (flag: TgpuVar<"workgroup", d.U32>) =>
    tgpu["~unstable"]
        .rawCodeSnippet(/* wgsl */ `workgroupUniformLoad(&flag)`, d.u32, "runtime", true)
        .$uses({ flag });

/**
 * a lazily-resolved, memoized WGSL chunk: `chunk(...)` returns the thunk, and the first call resolves
 * the given TGSL resources under `names: "strict"` so the emitted function names are the authored ones
 * and a raw-WGSL splice site can call them.
 *
 * Lazy on purpose. Resolution needs the build transform, so resolving at module scope would make
 * importing a module fail in tools that never touch the GPU. Pass the items as a thunk when a
 * schema's shape depends on a config value that is only
 * final after this module loads (the standard renderer's caster-count-sized uniforms). Pass a shared `ns` when two
 * chunks are always spliced together and must not both emit a shared dependency — the namespace emits
 * it into whichever chunk resolves first.
 *
 * The rethrow names the likely cause: without the plugin the failure is otherwise an opaque resolution
 * error from deep inside typegpu.
 * @internal
 */
export function chunk(
    label: string,
    items: Parameters<typeof tgpu.resolve>[0] | (() => Parameters<typeof tgpu.resolve>[0]),
    ns?: Namespace,
): () => string {
    let wgsl: string | undefined;
    return () => {
        if (wgsl !== undefined) return wgsl;
        try {
            const resolved = typeof items === "function" ? items() : items;
            wgsl = tgpu.resolve(resolved as never, { names: ns ?? "strict" });
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
