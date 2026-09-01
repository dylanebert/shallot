// Color-support detection: the portability ladder's tier selector (`shallot-tui.md` "Portability
// is a ladder in the encoder, not a second design"). Pure function over an injected env snapshot —
// no `process` read here, so it's testable with a plain object and carries no ambient state
// (`checks.md` "A test that exercises a function whose destination comes from ambient env must
// stub that env").

/**
 * The four output tiers, cheapest first. `plain` is non-interactive (no cursor addressing, no
 * SGR — a full grid dump every call, the mode a pipe or an agent reads as content); `glyph` is
 * interactive but colorless; `ansi256` and `truecolor` add SGR color at increasing fidelity and
 * byte cost.
 */
export type Tier = "plain" | "glyph" | "ansi256" | "truecolor";

export interface ColorEnvSnapshot {
    readonly isTTY: boolean;
    readonly env: Readonly<Record<string, string | undefined>>;
}

/**
 * Selects a tier from TTY-ness plus the standard color-support environment variables:
 * `NO_COLOR` (https://no-color.org) forces `glyph` on any TTY, `COLORTERM=truecolor|24bit` opts
 * into 24-bit color, and a `TERM` ending `-256color` opts into the 256-color SGR tier. A non-TTY
 * stream (a pipe, a file redirect) always gets `plain` — color escapes in a non-terminal sink are
 * noise, not content. An unrecognized or absent `TERM` on a real TTY defaults to `glyph`: every
 * terminal can print plain text and move its cursor, but not every terminal can be assumed to
 * parse color SGR correctly, so the safe default is the tier that always renders as legible text.
 */
export function detectTier({ isTTY, env }: ColorEnvSnapshot): Tier {
    if (!isTTY) return "plain";
    if (env.NO_COLOR !== undefined && env.NO_COLOR !== "") return "glyph";
    const colorterm = (env.COLORTERM ?? "").toLowerCase();
    if (colorterm === "truecolor" || colorterm === "24bit") return "truecolor";
    const term = env.TERM ?? "";
    if (term === "dumb") return "glyph";
    if (/-256color\b/.test(term)) return "ansi256";
    return "glyph";
}
