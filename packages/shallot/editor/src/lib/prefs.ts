// Editor preferences: the shape persisted to localStorage and the pure read/write/clamp helpers.
// App owns the localStorage get/set (the side effect); the parse-guard and panel bounds live here so
// they're `bun test`-covered and don't drift between the load site and the resize site.

export const PREFS_KEY = "shallot:editor";

export const PANEL_MIN = 200;
export const PANEL_MAX = 460;
export const PANEL_DEFAULT = 280;

// the docs reader floats as its own overlay and reads like a document, so it gets wider bounds than a
// sidebar — a comfortable measure for prose + code (DevDocs/MDN sit around here), the viewport still dominant.
export const DOCS_MIN = 340;
export const DOCS_MAX = 760;
export const DOCS_DEFAULT = 480;

/** clamp a sidebar width to the editor's panel bounds */
export function clampWidth(w: number): number {
    return Math.max(PANEL_MIN, Math.min(PANEL_MAX, w));
}

/** clamp the docs reader's width to its own (wider) bounds */
export function clampDocsWidth(w: number): number {
    return Math.max(DOCS_MIN, Math.min(DOCS_MAX, w));
}

export interface Prefs {
    /** the edit viewport's enabled-overlay bitmask (see `Overlay` in viewport.ts); overlays are edit-only chrome */
    overlays?: { edit?: number };
    lastScene?: Record<string, string>;
    outlinerWidth?: number;
    inspectorWidth?: number;
    docsWidth?: number;
    outlinerCollapsed?: boolean;
    inspectorCollapsed?: boolean;
    /** selected theme id (see THEMES in lib/theme.ts) */
    theme?: string;
}

/** parse persisted prefs, tolerating absent or corrupt storage (a first run, a hand-edit) — returns {} */
export function parsePrefs(raw: string | null): Prefs {
    try {
        return JSON.parse(raw ?? "{}");
    } catch {
        return {};
    }
}

export function serializePrefs(prefs: Prefs): string {
    return JSON.stringify(prefs);
}
