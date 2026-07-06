// Keyboard logic for the custom Select (a listbox combobox replacing the native `<select>`, whose
// popup renders at the OS / native-webview layer — unthemed, and positioned outside the editor's `fit`
// guard so it can open off-screen). The Select component owns the DOM + portal; this owns "what does a
// key do to the highlighted option", so the interaction is a pure transform unit-tested without a DOM.

export interface SelectStep {
    /** index to highlight next */
    active: number;
    /** commit the highlighted option (and close) */
    commit: boolean;
    /** close the menu */
    close: boolean;
}

/**
 * step an open menu by a key, or null for an unhandled key (the caller lets it through). Arrows wrap;
 * Home/End jump to the ends; Enter/Space commit the highlight; Escape/Tab close without committing.
 */
export function step(key: string, active: number, count: number): SelectStep | null {
    if (count === 0) return null;
    switch (key) {
        case "ArrowDown":
            return { active: (active + 1) % count, commit: false, close: false };
        case "ArrowUp":
            return { active: (active - 1 + count) % count, commit: false, close: false };
        case "Home":
            return { active: 0, commit: false, close: false };
        case "End":
            return { active: count - 1, commit: false, close: false };
        case "Enter":
        case " ":
            return { active, commit: true, close: true };
        case "Escape":
        case "Tab":
            return { active, commit: false, close: true };
        default:
            return null;
    }
}

/** the keys that open a closed Select — the native-select contract (arrows + Enter/Space). */
export function opensMenu(key: string): boolean {
    return key === "ArrowDown" || key === "ArrowUp" || key === "Enter" || key === " ";
}

/**
 * type-ahead: index of the next option whose label starts with `char` (case-insensitive), searching
 * forward from `from` and wrapping so repeated presses cycle matches. -1 when nothing matches.
 */
export function typeahead(labels: string[], char: string, from: number): number {
    const c = char.toLowerCase();
    const n = labels.length;
    for (let i = 1; i <= n; i++) {
        const idx = (from + i) % n;
        if (labels[idx]?.toLowerCase().startsWith(c)) return idx;
    }
    return -1;
}
