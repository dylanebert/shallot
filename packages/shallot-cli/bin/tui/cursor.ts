// Cursor addressing and the escape constants the encoder and screen session share. Sequences are
// plain ANSI/VT100 (CSI = `\x1b[`), no terminfo lookup — every target tier in the ladder
// (`color-support.ts`) is a real terminal, never a legacy device this package needs to special-case.

/** Cursor Position: absolute-move to `(row, col)`, both 0-indexed here, 1-indexed on the wire. */
export function cursorTo(row: number, col: number): string {
    return `\x1b[${row + 1};${col + 1}H`;
}

/** Home: absolute-move to (0, 0) — one byte shorter than `cursorTo(0, 0)` and equivalent. */
export const CURSOR_HOME = "\x1b[H";

/** Erase the whole screen (does not move the cursor — pair with `CURSOR_HOME` or a `cursorTo`). */
export const CLEAR_SCREEN = "\x1b[2J";

export const HIDE_CURSOR = "\x1b[?25l";
export const SHOW_CURSOR = "\x1b[?25h";

/** Switches to the terminal's alternate screen buffer, leaving scrollback and prior content intact. */
export const ENTER_ALT_SCREEN = "\x1b[?1049h";

/** Restores the primary screen buffer exactly as the terminal left it before `ENTER_ALT_SCREEN`. */
export const EXIT_ALT_SCREEN = "\x1b[?1049l";

/** Select Graphic Rendition reset — clears any fg/bg/attribute set by a prior SGR sequence. */
export const SGR_RESET = "\x1b[0m";
