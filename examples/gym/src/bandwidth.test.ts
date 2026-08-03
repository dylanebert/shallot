import { describe, expect, test } from "bun:test";
import { body, flat, integerDiscipline, noDivision } from "../../../packages/shallot/tests/wgsl";
import { bandwidthWgsl } from "./bandwidth";

const wgsl = bandwidthWgsl();

describe("bandwidth kernel reference", () => {
    test("preserves the sweep rotation, wrapped index, update, and grid stride", () => {
        expect(flat(body(wgsl, "@compute"))).toBe(
            "@compute @workgroup_size(64) fn bandwidth(@builtin(global_invocation_id) gid: vec3u) { " +
                "for (var s = 0u; (s < cfg.sweeps); s = (s + 1u)) { " +
                "let off = ((s * 1048573u) % 67108864u); var i = gid.x; while (true) { " +
                "if ((i >= 67108864u)) { break; } var idx = (i + off); " +
                "if ((idx >= 67108864u)) { idx = (idx - 67108864u); } " +
                "buf[idx] = ((buf[idx] * 1.000100016593933f) + vec4f(0.5)); " +
                "i = (i + 2097152u); } } }",
        );
    });

    test("keeps every integer local unsigned and uses no division", () => {
        integerDiscipline(wgsl);
        noDivision(wgsl);
    });
});
