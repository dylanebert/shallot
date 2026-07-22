/**
 * an input widget for a component field. Display-only: the stored value never changes; a
 * widget maps stored↔shown at the authoring boundary so ECS data stays pristine. Declare it in a
 * component's traits (`inputs`) when the default number field isn't the right control: a radians field
 * authored in degrees is an {@link angle}. The standard set is small on purpose; add a variant when a
 * field actually needs one.
 */
export type Input = { kind: "unit"; units: Unit[] };

/**
 * one entry in a {@link units} menu: how to show the stored value in this unit and read it back. `to`
 * and `from` must be inverse: an authoring host round-trips a value through them on every edit.
 */
export interface Unit {
    /** dropdown label, e.g. `deg` */
    label: string;
    /** stored value → shown value */
    to: (stored: number) => number;
    /** shown value → stored value */
    from: (shown: number) => number;
}

/** radians shown as-is: the identity unit, storage's own. */
export const radians: Unit = { label: "rad", to: (x) => x, from: (x) => x };

/** a radians field shown in degrees. */
export const degrees: Unit = {
    label: "deg",
    to: (r) => (r * 180) / Math.PI,
    from: (d) => (d * Math.PI) / 180,
};

/**
 * a number field with a unit dropdown. `list[0]` is the unit shown by default; storage is unchanged,
 * an authoring host converts through the selected unit's {@link Unit.to}/{@link Unit.from}.
 *
 * @example
 * traits: { Lens: { inputs: { fov: units([degrees, radians]) } } }
 */
export const units = (list: Unit[]): Input => ({ kind: "unit", units: list });

/** a radians field authored in degrees, with a `deg`/`rad` switch: the common angle case. */
export const angle: Input = units([degrees, radians]);
