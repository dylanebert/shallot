import { describe, expect, test } from "bun:test";
import { type Banner, pushToast, type Toast, upsertBanner } from "./notify-core";

const b = (id: string, text: string): Banner => ({ id, severity: "error", text });
const t = (id: number): Toast => ({ id, severity: "info", text: `t${id}` });

describe("pushToast", () => {
    test("appends under the cap", () => {
        const out = pushToast([t(0), t(1)], t(2), 3);
        expect(out.map((e) => e.id)).toEqual([0, 1, 2]);
    });

    test("evicts the oldest at the cap, never towering", () => {
        const out = pushToast([t(0), t(1), t(2)], t(3), 3);
        expect(out.map((e) => e.id)).toEqual([1, 2, 3]);
    });
});

describe("upsertBanner", () => {
    test("a new id appends", () => {
        const out = upsertBanner([b("gpu-lost", "lost")], b("build", "failed"));
        expect(out.map((e) => e.id)).toEqual(["gpu-lost", "build"]);
    });

    test("an existing id replaces in place, never stacking a duplicate", () => {
        const out = upsertBanner(
            [b("build", "failed once"), b("gpu-lost", "lost")],
            b("build", "failed twice"),
        );
        expect(out.map((e) => e.id)).toEqual(["build", "gpu-lost"]);
        expect(out[0].text).toBe("failed twice");
    });
});
