import { BodyType } from "../../core/physics";
import type { ScalarField } from "../../engine";
import { type FieldBinding, fieldMarked } from "./authoring";
import type { EntityId } from "./common/ids";
import type { FieldCandidates } from "./field-candidates";
import {
    BodyProperty,
    bodyDisable,
    bodyEnable,
    bodyGetProperty,
    bodyGetType,
    bodyIsValid,
    bodySetProperty,
    bodySetType,
} from "./world/body";
import type { WorldState } from "./world/world";

function value(binding: FieldBinding, eid: number, field: string): number {
    return (binding.storage[field] as ScalarField).get(eid);
}
function warnInvalid(warned: Set<number>, binding: FieldBinding, eid: number, field: string): void {
    const key = eid * 32 + binding.fieldIndices.get(field)!;
    if (warned.has(key)) return;
    warned.add(key);
    console.warn(`[physics] Body ${eid} ignored invalid ${field}; the solver value is unchanged`);
}
function applyBoolean(
    state: WorldState,
    binding: FieldBinding,
    eid: number,
    mask: number,
    field: string,
    body: number,
    property: number,
): void {
    if (!fieldMarked(binding, mask, field)) return;
    const next = value(binding, eid, field) !== 0;
    if (Number(next) !== bodyGetProperty(state, body, property))
        bodySetProperty(state, body, property, Number(next));
}
function applyFloat(
    state: WorldState,
    binding: FieldBinding,
    eid: number,
    mask: number,
    field: string,
    body: number,
    property: number,
    valid: "nonnegative" | "finite" | "any",
    warned: Set<number>,
): void {
    if (!fieldMarked(binding, mask, field)) return;
    const next = value(binding, eid, field);
    if (
        (valid === "nonnegative" && (!Number.isFinite(next) || next < 0)) ||
        (valid === "finite" && !Number.isFinite(next))
    ) {
        warnInvalid(warned, binding, eid, field);
        return;
    }
    if (!Object.is(next, bodyGetProperty(state, body, property)))
        bodySetProperty(state, body, property, next);
}

/** Applies marked live-definition fields by comparing them with Box3D's current body values. */
export function syncBodyFields(
    state: WorldState,
    candidates: FieldCandidates,
    binding: FieldBinding,
    bodies: ReadonlyMap<number, EntityId>,
    warned: Set<number>,
): void {
    for (let i = 0; i < candidates.count; i++) {
        const eid = candidates.eids[i];
        const mask = candidates.fieldMarks[eid];
        const id = bodies.get(eid);
        if (!id || !bodyIsValid(state, id)) continue;
        const body = id.index1 - 1;

        if (fieldMarked(binding, mask, "type")) {
            const next = value(binding, eid, "type");
            if (
                next !== BodyType.Static &&
                next !== BodyType.Kinematic &&
                next !== BodyType.Dynamic
            )
                warnInvalid(warned, binding, eid, "type");
            else if (next !== bodyGetType(state, body)) bodySetType(state, body, next);
        }
        applyFloat(
            state,
            binding,
            eid,
            mask,
            "linearDamping",
            body,
            BodyProperty.linearDamping,
            "nonnegative",
            warned,
        );
        applyFloat(
            state,
            binding,
            eid,
            mask,
            "angularDamping",
            body,
            BodyProperty.angularDamping,
            "nonnegative",
            warned,
        );
        applyFloat(
            state,
            binding,
            eid,
            mask,
            "gravityScale",
            body,
            BodyProperty.gravityScale,
            "finite",
            warned,
        );
        applyFloat(
            state,
            binding,
            eid,
            mask,
            "sleepThreshold",
            body,
            BodyProperty.sleepThreshold,
            "any",
            warned,
        );
        if (fieldMarked(binding, mask, "motionLocks")) {
            const next = value(binding, eid, "motionLocks");
            if ((next & ~0x3f) !== 0) warnInvalid(warned, binding, eid, "motionLocks");
            else if (next !== bodyGetProperty(state, body, BodyProperty.motionLocks))
                bodySetProperty(state, body, BodyProperty.motionLocks, next);
        }
        applyBoolean(state, binding, eid, mask, "enableSleep", body, BodyProperty.enableSleep);
        applyBoolean(state, binding, eid, mask, "isBullet", body, BodyProperty.isBullet);
        applyBoolean(
            state,
            binding,
            eid,
            mask,
            "allowFastRotation",
            body,
            BodyProperty.allowFastRotation,
        );
        applyBoolean(
            state,
            binding,
            eid,
            mask,
            "enableContactRecycling",
            body,
            BodyProperty.enableContactRecycling,
        );
        if (fieldMarked(binding, mask, "isEnabled")) {
            const next = value(binding, eid, "isEnabled") !== 0;
            if (next !== (bodyGetProperty(state, body, BodyProperty.isEnabled) !== 0)) {
                if (next) bodyEnable(state, body);
                else bodyDisable(state, body);
            }
        }
    }
}
