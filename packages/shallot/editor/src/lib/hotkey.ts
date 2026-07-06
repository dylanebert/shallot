// One formatter for the hotkey hint shown in toolbar tooltips, so every control reads the same — "Move [2]".
// The key declared on the control (tool.ts `ToolDef.key`, frame.ts) is the source of truth; a tooltip
// appends `hint(key)`, never a hand-written "(2)".

/** the trailing tooltip hint for a key — `hint("2")` → " [2]"; "" for no key, so it concatenates cleanly. */
export function hint(key?: string | null): string {
    return key ? ` [${key.toUpperCase()}]` : "";
}
