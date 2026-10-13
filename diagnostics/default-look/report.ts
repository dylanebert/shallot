import {
    type ColorRole,
    colorHex,
    MATERIAL_ROLES,
    oklchFromLinear,
    PALETTES,
    type Rgba,
    srgbChannel,
} from "./palette";
import { SCENES } from "./scenes";

function displayColor(value: string | number): Rgba {
    const channels =
        typeof value === "number"
            ? [value >> 16, (value >> 8) & 255, value & 255, 1]
            : value.startsWith("#")
              ? [
                    Number.parseInt(value.slice(1, 3), 16),
                    Number.parseInt(value.slice(3, 5), 16),
                    Number.parseInt(value.slice(5, 7), 16),
                    1,
                ]
              : value.match(/[\d.]+/g)!.map(Number);
    const linear = channels.slice(0, 3).map((channel) => {
        const v = channel / 255;
        return v <= 0.04045 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4;
    });
    return [linear[0]!, linear[1]!, linear[2]!, channels[3]!];
}

function row(
    name: string,
    value: Rgba,
    role: ColorRole | undefined,
    surface: boolean,
    enforce: boolean,
): string {
    const [l, c, h] = oklchFromLinear(value);
    const roleInside =
        role === "background"
            ? l >= 0.16 && l <= 0.34 && c <= 0.012
            : role === "foreground"
              ? l >= 0.55 && l <= 0.96 && c >= 0.02 && c <= 0.06
              : role === "accent"
                ? l >= 0.55 && l <= 0.83 && c >= 0.07 && c <= 0.22
                : true;
    const safe = value
        .slice(0, 3)
        .every((channel) => srgbChannel(channel) >= 50 && srgbChannel(channel) <= 240);
    if (enforce && (!roleInside || (surface && !safe)))
        throw new Error(`${name} is outside its candidate bounds`);
    return `<tr><td>${name}</td><td>${role ?? "non-surface"}</td><td style="background:${colorHex(value)};width:3em"></td><td>${l.toFixed(5)}, ${c.toFixed(5)}, ${h.toFixed(2)}°</td><td>${colorHex(value)}</td><td>[${value.map((v) => v.toFixed(7)).join(", ")}]</td><td>${role ? (roleInside ? "inside" : "outside") : "exempt"}</td><td>${surface ? (safe ? "inside" : "outside") : "exempt (non-albedo)"}</td></tr>`;
}

export function reportHtml(): string {
    const names = Object.keys(PALETTES);
    const scenes = SCENES.map(
        (scene) =>
            `<section><h2>${scene.title}</h2><p>${scene.description}</p><div class="captures">${names.map((name) => `<figure><figcaption>${name}</figcaption><a href="${name}/${scene.id}.png"><img src="${name}/${scene.id}.png" alt="${scene.title}, ${name}"></a></figure>`).join("")}</div></section>`,
    ).join("");
    const tables = Object.entries(PALETTES)
        .map(([name, palette]) => {
            const enforce = name !== "current";
            const rows = Object.entries(palette.materials).map(([key, value]) =>
                row(
                    `materials.${key}`,
                    value as Rgba,
                    MATERIAL_ROLES[key as keyof typeof MATERIAL_ROLES],
                    true,
                    enforce,
                ),
            );
            rows.push(
                row("textColor", displayColor(palette.textColor), "foreground", false, enforce),
            );
            rows.push(
                row("ui.panel", displayColor(palette.ui.panel), "background", false, enforce),
            );
            rows.push(row("ui.ink", displayColor(palette.ui.ink), "foreground", false, enforce));
            rows.push(
                row("ui.border", displayColor(palette.ui.border), "foreground", false, enforce),
            );
            rows.push(row("clear", displayColor(palette.clearColor), undefined, false, false));
            rows.push(
                row("warm sun / point / spot", displayColor(0xffe8cc), undefined, false, false),
            );
            rows.push(
                row("cool constant ambient", displayColor(0xd8e4ff), undefined, false, false),
            );
            return `<h2>${name}</h2><table><thead><tr><th>Value</th><th>Role</th><th>Swatch</th><th>OKLCH (L, C, hue)</th><th>sRGB hex</th><th>Linear RGBA</th><th>Role range</th><th>sRGB 50–240</th></tr></thead><tbody>${rows.join("")}</tbody></table>`;
        })
        .join("");
    return `<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Shallot default-look candidates</title><style>body{background:#1c1917;color:#ebdbb2;font:16px system-ui;margin:24px}h1,h2{font-weight:500}.captures{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}figure{margin:0}img{width:100%}figcaption{margin-bottom:8px}table{border-collapse:collapse;font:13px monospace;width:100%}td,th{padding:8px;text-align:left;border:1px solid #665c54}p{max-width:1000px}</style><h1>Shallot default-look candidates</h1><p>Same scenes, landed renderer defaults, warm sun, cool constant ambient. Click a capture for its full 1280×720 image. Current is the unchanged examples baseline; its outside values are reported, not repaired.</p><p>Candidate background: L 0.16–0.34, C ≤ 0.012. Foreground: L 0.55–0.96, C 0.02–0.06. Accents: L 0.55–0.83, C 0.07–0.22. Lit surfaces must also have every sRGB channel in 50–240 (checked before rounding). Text and UI meet role ranges, not albedo bounds; clear and lights are exempt from both. Linear RGBA is the tuple StandardMaterial takes; for text, UI and lights it is a conversion for comparison, not their authoring API. Hex omits alpha; the tuple retains it.</p><p>Dev orange: grey blockout with orange 55°, blue 250°, yellow-green 100° and red-orange 25°. Tidal teal: cool maritime teal 185°, violet 290°, blue 225° and green 145°. Orchid garden: magenta 325°, green 145°, purple 300° and gold 80°. All candidates keep identical role lightness/chroma; only hue changes. Ground and steps stay neutral; tower/walls use the dark neutral; three accents mark primitives and the lift; the fourth fills the existing embedded material entry, unused by this scene set.</p>${scenes}<h1>Colour values</h1>${tables}</html>`;
}
