import { describe, expect, test } from "bun:test";
import { body, flat, integerDiscipline, noDivision } from "../../../packages/shallot/tests/wgsl";
import { loadWgsl } from "./load";

const wgsl = loadWgsl();

describe("load kernel reference", () => {
    test("preserves the dynamic-uniform recurrence and scratch mapping", () => {
        expect(flat(body(wgsl, "@compute"))).toBe(
            "@compute @workgroup_size(64) fn load(@builtin(global_invocation_id) gid: vec3u) { " +
                "var x = ((f32(gid.x) * 9.999999747378752e-5f) + 1f); " +
                "for (var i = 0u; (i < cfg.iter); i = (i + 1u)) { " +
                "x = (((sin(x) * 1.000100016593933f) + " +
                "(cos((x * 1.2999999523162842f)) * 0.5f)) + 0.699999988079071f); " +
                "} buf[(gid.x % 4096u)] = x; }",
        );
    });

    test("keeps every integer local unsigned and uses no division", () => {
        integerDiscipline(wgsl);
        noDivision(wgsl);
    });
});
