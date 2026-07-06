// Pure key routing: a keydown resolves to one named editor shortcut (or none). The `.svelte` handler is a
// thin dispatch over the result, so the chord table + the type-into-a-field guard are `bun test`-covered.
// Tool-mode keys (the number row) route through `toolForKey` (tool.ts) under this same `inField` guard —
// exported here as the one source of truth for "is the user typing". See testing.md "Editor tiers".

export type Shortcut = "save" | "undo" | "redo" | "toggle-sidebar" | "rename" | "delete" | "help";

// the minimal slice of a KeyboardEvent the resolver reads — a real event satisfies it structurally
interface KeyEventLike {
    key: string;
    ctrlKey: boolean;
    metaKey: boolean;
    shiftKey: boolean;
    target: EventTarget | null;
}

/** true when a text field has focus, so a bare key (Delete, a tool digit) defers to typing rather than
 * firing a shortcut. Modifier chords (save/undo) fire regardless and don't consult this. */
export function inField(e: KeyEventLike): boolean {
    const tag = (e.target as { tagName?: string } | null)?.tagName;
    return tag === "INPUT" || tag === "TEXTAREA";
}

/** map a keydown to its editor shortcut, or null when nothing matches. rename/delete defer to whatever
 * else a text field does while focused; save/undo/redo/collapse fire everywhere. */
export function resolveShortcut(e: KeyEventLike): Shortcut | null {
    const mod = e.ctrlKey || e.metaKey;
    // shift (and caps lock) report the shifted character — ctrl+shift+z arrives as "Z"
    const k = e.key.toLowerCase();

    if (mod && k === "s") return "save";
    if (mod && k === "z" && !e.shiftKey) return "undo";
    if ((mod && k === "z" && e.shiftKey) || (mod && k === "y")) return "redo";
    if (mod && e.key === "\\") return "toggle-sidebar";
    if (e.key === "F2") return inField(e) ? null : "rename";
    if ((e.key === "Delete" || e.key === "Backspace") && !mod) return inField(e) ? null : "delete";
    if (e.key === "?" && !mod) return inField(e) ? null : "help";
    return null;
}
