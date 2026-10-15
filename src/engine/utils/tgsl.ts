import tgpu, { type Namespace, type TgpuVar } from "typegpu";
import * as d from "typegpu/data";
import { isBeingTranspiled } from "typegpu/std";

const idivWgsl = tgpu.fn(
    [d.u32, d.u32],
    d.u32,
)(/* wgsl */ `(a: u32, b: u32) -> u32 { return a / b; }`);

/** Unsigned integer division. TGSL's `/` on integer operands transpiles to `f32(a) / f32(b)`,
 *  which can return a fractional quotient or lose integer precision above 2²⁴.
 *  The divisor must be nonzero; CPU and WGSL zero-divisor behavior differs. */
export const idiv = tgpu.fn(
    [d.u32, d.u32],
    d.u32,
)((a, b) => {
    "use gpu";
    return isBeingTranspiled() ? idivWgsl(a, b) : Math.floor(a / b);
});

/** GPU-only WGSL `workgroupUniformLoad(&v)`: a control barrier with a uniform result,
 *  permitting a `workgroupBarrier` in a loop gated by that result.
 *  Bind one loader per variable at module scope and read it as `.$` at the call site.
 *  Emitted inline because WGSL 1.0 forbids workgroup-address-space pointer parameters. */
export const uniformLoad = (flag: TgpuVar<"workgroup", d.U32>) =>
    tgpu["~unstable"]
        .rawCodeSnippet(/* wgsl */ `workgroupUniformLoad(&flag)`, d.u32, "runtime", true)
        .$uses({ flag });

/**
 * Memoized WGSL chunk, resolved on first call, not at import. Default strict names let raw WGSL
 * call the authored function names. Pass items as a thunk for config-dependent schemas.
 * Shared `ns` emits a dependency only into the first chunk that resolves it; use it only for
 * chunks that are always spliced together. Resolution failures retain their cause and name the chunk.
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
