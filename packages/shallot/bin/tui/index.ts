// Private terminal encoder turning a character-cell grid into
// terminal bytes. No GPU, no `@dylanebert/shallot` import (`scripts/check-tui-boundary.ts`
// enforces it), so this is fully testable on any seat with no adapter.

export { type ColorEnvSnapshot, detectTier, type Tier } from "./color-support";
export {
    CLEAR_SCREEN,
    CURSOR_HOME,
    cursorTo,
    ENTER_ALT_SCREEN,
    EXIT_ALT_SCREEN,
    HIDE_CURSOR,
    SGR_RESET,
    SHOW_CURSOR,
} from "./cursor";
export { type ChangedRun, diffRuns } from "./diff";
export { Encoder } from "./encoder";
export {
    onResize,
    type ResizeSource,
    type SizeSource,
    type TerminalSize,
    terminalSize,
} from "./resize";
export {
    ALT_SCREEN_ENTER,
    ALT_SCREEN_EXIT,
    DEFAULT_TEARDOWN_SIGNALS,
    installTeardown,
    type SignalSource,
    type TeardownOptions,
} from "./screen";
export { ansi256FromRgb, cube6, encodeRun, sameStyle, sgrPrefix } from "./sgr";
export { BLANK_CELL, type Cell, cellEqual, type Grid, makeGrid, type RGB, rgbEqual } from "./types";
