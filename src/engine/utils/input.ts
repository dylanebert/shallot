/**
 * an input widget for a component field. Display-only: the stored value never changes; a
 * widget maps stored↔shown at the authoring boundary so ECS data stays pristine. Declare it in a
 * component's traits (`inputs`) when the default number field isn't the right control: a radianUnit field
 * authored in degreeUnit is an {@link angleInput}. The standard set is small on purpose; add a variant when a
 * field actually needs one.
 */
export type FieldInput = { kind: "unit"; units: FieldUnit[] };

/**
 * one entry in a {@link unitInput} menu: how to show the stored value in this unit and read it back. `to`
 * and `from` must be inverse: an authoring host round-trips a value through them on every edit.
 */
export interface FieldUnit {
    /** dropdown label, e.g. `deg` */
    label: string;
    /** stored value → shown value */
    to: (stored: number) => number;
    /** shown value → stored value */
    from: (shown: number) => number;
}

/** radianUnit shown as-is: the identity unit, storage's own. */
export const radianUnit: FieldUnit = { label: "rad", to: (x) => x, from: (x) => x };

/** a radianUnit field shown in degreeUnit. */
export const degreeUnit: FieldUnit = {
    label: "deg",
    to: (r) => (r * 180) / Math.PI,
    from: (d) => (d * Math.PI) / 180,
};

/**
 * a number field with a unit dropdown. `list[0]` is the unit shown by default; storage is unchanged,
 * an authoring host converts through the selected unit's {@link FieldUnit.to}/{@link FieldUnit.from}.
 *
 * @example
 * traits: { Lens: { inputs: { fov: unitInput([degreeUnit, radianUnit]) } } }
 */
export const unitInput = (list: FieldUnit[]): FieldInput => ({ kind: "unit", units: list });

/** a radianUnit field authored in degreeUnit, with a `deg`/`rad` switch: the common angleInput case. */
export const angleInput: FieldInput = unitInput([degreeUnit, radianUnit]);
