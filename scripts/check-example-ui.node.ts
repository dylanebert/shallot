import { expect, setDefaultTimeout, test } from "bun:test";
import { readdir, readFile } from "node:fs/promises";
import { join } from "node:path";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { PALETTES } from "../diagnostics/default-look/palette";

setDefaultTimeout(TIMEOUT.node);

async function sources(directory: string): Promise<string[]> {
    const result: string[] = [];
    for (const entry of await readdir(directory, { withFileTypes: true })) {
        if (["node_modules", "dist", "build", ".artifacts", "test-results"].includes(entry.name))
            continue;
        const path = join(directory, entry.name);
        if (entry.isDirectory()) result.push(...(await sources(path)));
        else if (
            /\.(css|html|svelte|[cm]?[jt]s)$/.test(entry.name) &&
            !/\.(test|node|e2e)\./.test(entry.name)
        )
            result.push(path);
    }
    return result;
}

function titleDeclaration(source: string, offset: number): boolean {
    const before = source.slice(0, offset);
    const tag = before.slice(before.lastIndexOf("<"));
    if (/^<h1\b[^>]*\bstyle\s*=/.test(tag)) return true;
    const open = before.lastIndexOf("{");
    const boundary = Math.max(before.lastIndexOf("}", open), before.lastIndexOf("{", open - 1));
    const selectors = before
        .slice(boundary + 1, open)
        .trim()
        .split(",");
    return (
        open > before.lastIndexOf("}") &&
        selectors.every((selector) => /(?:^|\s|>)h1\s*$/.test(selector))
    );
}

function violations(source: string): string[] {
    const ui = PALETTES["aperture-carbon"].ui;
    const failures: string[] = [];
    for (const match of source.matchAll(
        /\b(font(?:-family|-size|-weight)?|border(?:-top-left|-top-right|-bottom-left|-bottom-right)?-radius)\s*:\s*([^;}"'\n]+)/g,
    )) {
        const [, property, raw] = match;
        const value = raw!.trim();
        let valid = false;
        let size: string | undefined;
        if (property === "font-family") valid = value === ui.fontFamily;
        else if (property === "font-size") {
            size = value;
            valid = ui.fontSizes.includes(value);
        } else if (property === "font-weight") valid = value === ui.fontWeight;
        else if (property!.includes("radius")) valid = value === ui.radius || value === ui.circle;
        else {
            const font = value.match(/^(?:(\d+)\s+)?(\d+px)(?:\/[^\s]+)?\s+(.+)$/);
            size = font?.[2];
            valid =
                !!font &&
                (font[1] ?? "400") === ui.fontWeight &&
                ui.fontSizes.includes(font[2]!) &&
                font[3] === ui.fontFamily;
        }
        if (size === ui.fontSizes[1] && !titleDeclaration(source, match.index!)) valid = false;
        if (!valid) failures.push(`${property}: ${value}`);
    }
    return failures;
}

test("the UI scale reserves 24px for titles and rejects off-token CSS and inline declarations", () => {
    const ui = PALETTES["aperture-carbon"].ui;
    expect(ui.fontSizes).toEqual(["14px", "24px"]);
    expect(
        violations(
            `.panel { font: ${ui.fontWeight} ${ui.fontSizes[0]} ${ui.fontFamily}; border-radius: ${ui.radius}; }`,
        ),
    ).toEqual([]);
    expect(
        violations(
            '<span style="font-family: sans-serif; font-size: 18px; font-weight: 700; border-radius: 6px">',
        ),
    ).toHaveLength(4);
    expect(violations(`.reticle { border-radius: ${ui.circle}; }`)).toEqual([]);
    expect(violations(".panel { font: bold 48px serif; }")).toHaveLength(1);
    expect(violations("h1 { font-size: 24px; }")).toEqual([]);
    expect(violations('<h1 style="font-size: 24px">Name</h1>')).toEqual([]);
    expect(violations(".notice { font-size: 24px; }")).toHaveLength(1);
    expect(violations("h1, .notice { font-size: 24px; }")).toHaveLength(1);
    expect(violations('<p style="font-size: 24px">Status</p>')).toHaveLength(1);
    expect(violations("small { font-size: 12px; }")).toHaveLength(1);
});

for (const example of await readdir("examples")) {
    test(`${example} CSS and inline styles use the UI type and shape tokens`, async () => {
        const failures: string[] = [];
        for (const path of await sources(join("examples", example))) {
            const source = await readFile(path, "utf8");
            failures.push(...violations(source).map((failure) => `${path}: ${failure}`));
        }
        expect(failures).toEqual([]);
    });
}

test("embedded-scene host page uses the UI type and shape tokens", async () => {
    expect(violations(await readFile("examples/embedded-scene/src/style.css", "utf8"))).toEqual([]);
});
