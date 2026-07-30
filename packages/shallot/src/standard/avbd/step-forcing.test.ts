import { expect, test } from "bun:test";
import { accessors } from "./step";

// Base-forcing for the step's shared chunks, in a file of its own for the reason collide.ts's
// `forcing.test.ts` has one: each chunk memoizes and a variant's `Namespace` emits a shared definition
// into whichever chunk resolves FIRST, so a dependent chunk that forgot to force its base would swallow
// the base's definitions — and a pass splicing both then declares them twice, which Dawn rejects at
// pipeline creation with nothing pointing at the cause. The property is resolution-order dependent, so
// this builds its OWN variant (untouched by step.test.ts) and resolves the DEEPEST chunk first.

// every kind of top-level declaration a chunk can emit — fn, struct, const, and a module-scope `var`
// (a binding, or a workgroup array). A duplicated `const` is as fatal as a duplicated `fn`.
const defs = (wgsl: string): string[] =>
    [
        ...wgsl.matchAll(
            /^(?:fn|struct|const)\s+([A-Za-z0-9_]+)|^(?:@group[^;]*?)?var<[^>]*>\s+([A-Za-z0-9_]+)/gm,
        ),
    ].map((m) => m[1] ?? m[2]);

test("the deepest chunk resolved first still leaves its bases their definitions", () => {
    const V = accessors("readonly", "readonly");
    // boxExtent → bodyPose → bodyRest → math is the longest chain
    expect(defs(V.boxExtentWgsl())).toEqual(["boxExtent"]);
    // the `Step` struct + the `params` declaration ride the first chunk that touches the layout
    expect(defs(V.bodyWgsl())).toContain("bPos");
    expect(defs(V.bodyRestWgsl())).toContain("bCol");
    expect(defs(V.bodyRestWgsl())).toContain("Step");
    expect(defs(V.mathWgsl())).toContain("qRotateW");
    // a contact-only consumer (the CSR count/scatter) resolves `cc` FIRST, and `params` is shared with
    // the body accessors — so the contact chunk forces them, or the CSR passes lose the declaration
    const W = accessors("readonly", "mutable");
    // `pairContacts` is the contact chunk's own binding; `params` + `Step` stay with the body chunk
    expect(defs(W.contactWgsl())).toEqual(["pairContacts", "cc"]);
    expect(defs(W.bodyRestWgsl())).toContain("Step");
    expect(defs(W.mathWgsl())).toContain("qMulW");
});
