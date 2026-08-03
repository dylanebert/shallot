import { describe, expect, test } from "bun:test";
import { body, flat, integerDiscipline, noDivision } from "../../../../packages/shallot/tests/wgsl";
import { chainWgsl, runChainArms } from "./chain";

const wgsl = chainWgsl();

describe("chain kernel reference", () => {
    test("encodes N dispatch phases followed by one barrier chain", () => {
        const calls: string[] = [];
        runChainArms(
            4,
            () => calls.push("dispatch"),
            () => calls.push("barrier"),
        );
        expect(calls).toEqual(["dispatch", "dispatch", "dispatch", "dispatch", "barrier"]);
    });

    test("preserves the dispatch phase and in-kernel storage-barrier recurrence", () => {
        expect(flat(body(wgsl, "@compute @workgroup_size(64) fn phase("))).toBe(
            "@compute @workgroup_size(64) fn phase(@builtin(local_invocation_index) lid: u32) { " +
                "if ((lid == 0u)) { data[0u] = (data[0u] + 1u); } }",
        );
        expect(flat(body(wgsl, "@compute @workgroup_size(64) fn barrierChain("))).toBe(
            "@compute @workgroup_size(64) fn barrierChain" +
                "(@builtin(local_invocation_index) lid: u32) { " +
                "for (var i = 0u; (i < cp.phases); i = (i + 1u)) { " +
                "if ((lid == 0u)) { data[0u] = (data[0u] + 1u); } storageBarrier(); } }",
        );
    });

    test("keeps every integer local unsigned and uses no division", () => {
        integerDiscipline(wgsl);
        noDivision(wgsl);
    });
});
