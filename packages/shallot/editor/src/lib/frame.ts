// The transform gizmo's coordinate frame — the axes the Move / Rotate handles align to (Scale is always
// local, since a per-axis scale is local by nature). Pure data here, the way `tool.ts` holds the tool set:
// the frame identity plus the dropdown's ordered list. App owns the active-frame `$state`; the viewport-bar
// dropdown renders from `FRAMES`, and `Handles.getAxes` switches on the value. Two frames today (World /
// Local); the list is the extension point — Normal / View / Gimbal (Unity / Blender's set) drop in as one
// more entry plus the axes that compute them, no UI change.
import { Axis3d, Globe } from "lucide-static";

/** the frame the handles align to. World = scene axes; Local = the selection's own orientation. */
export const Frame = { World: 0, Local: 1 } as const;
export type Frame = (typeof Frame)[keyof typeof Frame];

export interface FrameDef {
    id: Frame;
    label: string;
    /** lucide-static SVG markup, rendered through `Icon` */
    icon: string;
}

/** the dropdown order, top to bottom. */
export const FRAMES: FrameDef[] = [
    { id: Frame.World, label: "World", icon: Globe },
    { id: Frame.Local, label: "Local", icon: Axis3d },
];

/** World is the default — handles align to the scene axes until you opt into the object's own frame. */
export const DEFAULT_FRAME: Frame = Frame.World;

/** the hotkey that cycles the frame (X — Unity's handle-orientation toggle). The single source of truth:
 * the keydown handler matches it and the toolbar tooltip renders its {@link hint}. */
export const FRAME_KEY = "x";

/** the next frame in the cycle after `f` (wraps), for the hotkey. */
export function nextFrame(f: Frame): Frame {
    const i = FRAMES.findIndex((d) => d.id === f);
    return FRAMES[(i + 1) % FRAMES.length].id;
}
