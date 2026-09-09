// Color-support detection: the portability ladder's tier selector — portability is a ladder in
// the encoder, not a second design. Pure function over an injected env snapshot — no `process`
// read here, so it's testable with a plain object and carries no ambient state.

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
 * Selects a tier from TTY-ness plus the standard color-support environment variables.
 *
 * `FORCE_COLOR`'s **enable** values (a set value other than `"0"`/`"false"`) are checked **first,
 * regardless of TTY-ness** (N6) — overriding a non-TTY sink (CI, a wrapped run, `| less -R`) into
 * emitting color is the primary reason the variable exists, so an enabling `FORCE_COLOR` set on a
 * pipe or file redirect is honored, not silently discarded by a TTY check that runs first. `"1"`
 * or `"2"` force `ansi256` (this ladder has no separate 16-color tier, so both collapse to the
 * lowest color tier it does have), `"3"` forces `truecolor`, and any other set value (a bare
 * `FORCE_COLOR=` or `"true"`) forces the lowest color tier on.
 *
 * `FORCE_COLOR`'s **disable** values (`"0"` or `"false"`) do not short-circuit the TTY gate the
 * same way: on a non-TTY sink `plain` already carries no color and no cursor addressing, so
 * "disable color" is a no-op there and the sink still reads `plain` — never `glyph`, which is the
 * *interactive* colorless tier and still emits `CLEAR_SCREEN`/`cursorTo` (`encoder.ts`), noise in
 * a pipe. The disable value does real work only on a real TTY, where it forces `glyph`.
 *
 * Absent `FORCE_COLOR`: a non-TTY stream always gets `plain` — color escapes in a non-terminal
 * sink are noise, not content. On a real TTY: `NO_COLOR` (https://no-color.org) forces `glyph`.
 * `TERM=dumb` is checked next, and wins over `COLORTERM` (N6) — `supports-color` treats a dumb
 * terminal as a hard "no color" regardless of what `COLORTERM` claims, so `COLORTERM=truecolor
 * TERM=dumb` reads `glyph` here too, not `truecolor`. Otherwise: `COLORTERM=truecolor|24bit` opts
 * into 24-bit color, a `TERM` ending `-256color` opts into the 256-color SGR tier, a `TERM` ending
 * `-direct` (the terminfo convention for direct-color terminals, e.g. `xterm-direct`) opts into
 * truecolor, and `xterm-kitty`/`xterm-ghostty` are recognized as truecolor-capable by `TERM` alone
 * — both are commonly reached over `ssh`, where `COLORTERM` isn't inherited across the connection
 * and would otherwise fall all the way to `glyph`. An unrecognized or absent `TERM` with none of
 * the above still defaults to `glyph`: every terminal can print plain text and move its cursor,
 * but not every terminal can be assumed to parse color SGR correctly, so the safe default is the
 * tier that always renders as legible text.
 */
export function detectTier({ isTTY, env }: ColorEnvSnapshot): Tier {
    const forceColor = env.FORCE_COLOR;
    const forceDisabled = forceColor === "0" || forceColor === "false";
    if (forceColor !== undefined && !forceDisabled) {
        if (forceColor === "3") return "truecolor";
        // "1", "2", or any other set value (bare/"true") — the lowest color tier this ladder has.
        return "ansi256";
    }

    if (!isTTY) return "plain";

    if (forceDisabled) return "glyph";

    if (env.NO_COLOR !== undefined && env.NO_COLOR !== "") return "glyph";
    const term = env.TERM ?? "";
    if (term === "dumb") return "glyph";
    const colorterm = (env.COLORTERM ?? "").toLowerCase();
    if (colorterm === "truecolor" || colorterm === "24bit") return "truecolor";
    if (/-256color\b/.test(term)) return "ansi256";
    if (/-direct\b/.test(term)) return "truecolor";
    if (term === "xterm-kitty" || term === "xterm-ghostty") return "truecolor";
    return "glyph";
}
