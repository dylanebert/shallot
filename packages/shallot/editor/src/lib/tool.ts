// The viewport's active transform tool — the classic scheme (Unity / PlayCanvas / Godot): one mode at a
// time, switched from the toolbar. Pure data here: the mode identity plus the toolbar's ordered list. App
// owns the active-tool `$state`; the toolbar control renders from `TOOLS`, and the manipulator gizmo (a
// later pass) switches on the value. Keeping the set here makes it `bun test`-checkable for drift against
// the toolbar, the way `Overlay` (viewport.ts) is the source of truth for the gizmo dropdown.
import { MousePointer2, Move3d, Rotate3d, Scale3d } from "lucide-static";

/** the manipulator a selection shows. Select = pick only; Move / Rotate / Scale show the matching gizmo. */
export const Tool = { Select: 0, Move: 1, Rotate: 2, Scale: 3 } as const;
export type Tool = (typeof Tool)[keyof typeof Tool];

export interface ToolDef {
    id: Tool;
    label: string;
    /** lucide-static SVG markup, rendered through `Icon` */
    icon: string;
    /** the hotkey — the single source of truth: `toolForKey` matches against it and the toolbar tooltip
     * renders its {@link hint}. Tools sit on the digits, not W/E/R, because the editor camera flies on
     * always-on WASD/QE (PlayCanvas's scheme). */
    key: string;
}

/** the toolbar order, left to right. */
export const TOOLS: ToolDef[] = [
    { id: Tool.Select, label: "Select", icon: MousePointer2, key: "1" },
    { id: Tool.Move, label: "Move", icon: Move3d, key: "2" },
    { id: Tool.Rotate, label: "Rotate", icon: Rotate3d, key: "3" },
    { id: Tool.Scale, label: "Scale", icon: Scale3d, key: "4" },
];

/** Move is the default — the most common edit, the floor a beginner reaches for first. */
export const DEFAULT_TOOL: Tool = Tool.Move;

/** the tool a key selects, or null — matched against each {@link ToolDef.key}, the one source of truth. */
export function toolForKey(key: string): Tool | null {
    return TOOLS.find((t) => t.key === key)?.id ?? null;
}
