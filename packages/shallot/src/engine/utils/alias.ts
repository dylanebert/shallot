import { euler, quat } from "./math";

/**
 * a field's authoring alias: a bidirectional codec between its stored lanes and the representation a
 * user edits. Sits alongside the scalar string codecs (`parse`/`format` traits); this is the vector
 * one (lanes ↔ lanes). Declare it in a component's traits when the stored form isn't what should be
 * edited; the canonical case is a quaternion authored as euler angles ({@link eulerAlias}). A field
 * without an alias edits its lanes directly. The standard set is small on purpose: euler is the one
 * common vector alias; add a member when a field actually needs it.
 */
export interface Alias {
    /** axis labels for the editable inputs */
    axes: readonly string[];
    /** stored lanes (dotted keys, e.g. `rot.x`) → the per-axis values shown */
    read(parsed: Record<string, number>): number[];
    /** axis `i` edited to `value` → the stored-lane updates to apply (dotted keys) */
    write(axis: number, value: number, parsed: Record<string, number>): Record<string, number>;
}

/**
 * author a packed `Pair`/`Quad`'s lanes by name (`metallic`, `roughness`) for one-buffer storage with
 * friendly authoring: `material="metallic: 1; roughness: 0.2"`. An **identity** alias: `axes.length`
 * equals the field's lane count, each axis 1:1 with a lane, which the scene parser + serializer honor.
 * A non-identity alias ({@link eulerAlias}, 3 axes over a 4-lane quat) stays editor-only: the length
 * mismatch is the discriminator that keeps quaternion fields authored positionally.
 *
 * @example
 * traits: { Material: { aliases: { params: laneAlias("params", ["metallic", "roughness", "emissive", "occlusion"]) } } }
 */
export function laneAlias(base: string, names: readonly string[]): Alias {
    const key = (i: number): string => `${base}.${"xyzw"[i]}`;
    return {
        axes: names,
        read: (p) => names.map((_, i) => p[key(i)] ?? 0),
        write: (axis, value) => ({ [key(axis)]: value }),
    };
}

const EULER_AXES = ["x", "y", "z"] as const;

/**
 * author a quaternion stored at `base` (lanes `base.x/y/z/w`) as euler angles in degrees. The user
 * never sees the quaternion: `read` decodes quat→euler, `write` encodes the edited euler→quat.
 *
 * @example
 * traits: { Transform: { aliases: { rot: eulerAlias("rot") } } }
 */
export function eulerAlias(base: string): Alias {
    const lane = (p: Record<string, number>, k: string): number =>
        p[`${base}.${k}`] ?? (k === "w" ? 1 : 0);
    return {
        axes: EULER_AXES,
        read(p) {
            const e = euler(lane(p, "x"), lane(p, "y"), lane(p, "z"), lane(p, "w"));
            return [e.x, e.y, e.z];
        },
        write(axis, value, p) {
            const e = euler(lane(p, "x"), lane(p, "y"), lane(p, "z"), lane(p, "w"));
            const a = EULER_AXES[axis];
            if (a) e[a] = value;
            const q = quat(e.x, e.y, e.z);
            return {
                [`${base}.x`]: q.x,
                [`${base}.y`]: q.y,
                [`${base}.z`]: q.z,
                [`${base}.w`]: q.w,
            };
        },
    };
}
