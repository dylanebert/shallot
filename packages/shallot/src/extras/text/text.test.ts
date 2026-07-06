import { describe, expect, test } from "bun:test";
import { font, TextPlugin, text } from "./";

// The registries are the CPU codec boundary the scene parse/format trait crosses — the glyph layout + GPU
// draw are validated in `bun bench`. These assert the interning is stable and round-trips through the
// public trait, no State or GPU needed.
const traits = TextPlugin.traits!.Text;
const parseContent = traits.parse!.content as (raw: string) => number;
const formatContent = traits.format!.content as (id: number) => string;
const parseFont = traits.parse!.font as (name: string) => number;

describe("kitchen text registry surface", () => {
    test("text() interns a string and the trait formats the id back", () => {
        const id = text("Hello");
        expect(formatContent(id)).toBe("Hello");
        expect(parseContent("Hello")).toBe(id); // scene parse interns through the same table
    });

    test("identical strings dedupe to one id, distinct strings don't", () => {
        expect(text("shared")).toBe(text("shared"));
        expect(text("first")).not.toBe(text("second"));
    });

    test("content id 0 is the empty string — the Text.content default", () => {
        expect(text("")).toBe(0);
        expect(formatContent(0)).toBe("");
    });

    test("font() dedupes by name, distinct names get distinct ids", () => {
        const inter = font("/inter.ttf", "inter");
        expect(font("/inter.ttf", "inter")).toBe(inter); // same name, same id
        expect(font("/pixel.ttf", "pixel")).not.toBe(inter);
    });

    test("the font trait parse resolves a name to its id, unknown names to the default 0", () => {
        const id = font("/mono.ttf", "mono");
        expect(parseFont("mono")).toBe(id);
        expect(parseFont("nonexistent")).toBe(0);
    });
});
