import { describe, expect, test } from "bun:test";
import { fieldDocs, plain } from "./fielddocs";

// the committed artifact, the editor's runtime view of the source comments. jsdoc.test pins the parser;
// this pins the generation pipeline's output shape — kebab component key, kebab field keys (the inspector's
// labels), description + type + default per field — so the inspector can index it the way it indexes schema().
describe("fieldDocs — the generated UI reference artifact", () => {
    test("orbit fields carry description, type, default, keyed by kebab label", () => {
        // yaw carries the `angle` input — the artifact shows the unit menu as the type and the default
        // in the unit shown first (degrees), not the stored radians
        const yaw = fieldDocs.orbit?.fields.yaw;
        expect(yaw).toEqual({
            description: "horizontal orbit angle around the target, radians",
            type: "deg | rad",
            default: "30 deg",
        });
        // mode is an enum — the artifact shows the options and the default option name, not a raw int
        expect(fieldDocs.orbit.fields.mode).toEqual({
            description:
                "Free orbits, pans, and zooms; Locked disables orbit rotation, leaving pan and zoom",
            type: "free | locked",
            default: "free",
        });
        expect(fieldDocs.orbit.fields["min-pitch"].description).toBe("lower pitch clamp, radians");
        expect(fieldDocs.orbit.summary).toContain("orbit camera controls");
    });

    test("an un-annotated component still carries its fields, description null", () => {
        // transform ships no per-field comments yet (it hardens later); type/default come from reflection,
        // so the inspector reads a full entry — only the prose is absent, never the whole key.
        expect(fieldDocs.transform.fields.pos).toEqual({
            description: null,
            type: "vec3",
            default: "0 0 0",
        });
    });
});

describe("plain — JSDoc link cleanup for non-markdown surfaces", () => {
    test("{@link Target} collapses to the word; {@link Target|text} to the text", () => {
        expect(plain("drives a {@link Transform}'s pos")).toBe("drives a Transform's pos");
        expect(plain("the {@link jump|buffered jump} sets it")).toBe("the buffered jump sets it");
        expect(plain("no links here")).toBe("no links here");
    });
});
