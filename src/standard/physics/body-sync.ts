import { BodyType } from "../../core/physics";
import type { ScalarField } from "../../engine";
import type { Body as SolverBody } from "./api";
import type { BodyBinding } from "./authoring";
import type { FieldCandidates } from "./field-candidates";

function value(binding: BodyBinding, eid: number, field: string): number {
    return (binding.storage[field] as ScalarField).get(eid);
}
function changed(binding: BodyBinding, mask: number, field: string): boolean {
    const index = binding.fieldIndices.get(field);
    return index !== undefined && (mask & (1 << index)) !== 0;
}
function warnInvalid(warned: Set<number>, binding: BodyBinding, eid: number, field: string): void {
    const key = eid * 32 + binding.fieldIndices.get(field)!;
    if (warned.has(key)) return;
    warned.add(key);
    console.warn(`[physics] Body ${eid} ignored invalid ${field}; the solver value is unchanged`);
}
function applyBoolean(
    binding: BodyBinding,
    eid: number,
    mask: number,
    field: string,
    body: SolverBody,
    getter: string,
    setter: string,
): void {
    if (!changed(binding, mask, field)) return;
    const next = value(binding, eid, field) !== 0;
    const methods = body as unknown as Record<string, (...args: boolean[]) => boolean | void>;
    if (next !== methods[getter].call(body)) methods[setter].call(body, next);
}
function applyFloat(
    binding: BodyBinding,
    eid: number,
    mask: number,
    field: string,
    body: SolverBody,
    getter: string,
    setter: string,
    valid: "nonnegative" | "finite" | "any",
    warned: Set<number>,
): void {
    if (!changed(binding, mask, field)) return;
    const next = value(binding, eid, field);
    if (
        (valid === "nonnegative" && (!Number.isFinite(next) || next < 0)) ||
        (valid === "finite" && !Number.isFinite(next))
    ) {
        warnInvalid(warned, binding, eid, field);
        return;
    }
    const methods = body as unknown as Record<string, (...args: number[]) => number | void>;
    if (!Object.is(next, methods[getter].call(body))) methods[setter].call(body, next);
}

/** Applies marked live-definition fields by comparing them with Box3D's current body values. */
export function syncBodyFields(
    candidates: FieldCandidates,
    binding: BodyBinding,
    bodies: ReadonlyMap<number, SolverBody>,
    warned: Set<number>,
): void {
    for (let i = 0; i < candidates.count; i++) {
        const eid = candidates.eids[i];
        const mask = candidates.fieldMarks[eid];
        const body = bodies.get(eid);
        if (!body) continue;

        if (changed(binding, mask, "type")) {
            const next = value(binding, eid, "type");
            if (
                next !== BodyType.Static &&
                next !== BodyType.Kinematic &&
                next !== BodyType.Dynamic
            )
                warnInvalid(warned, binding, eid, "type");
            else if (next !== body.getType()) body.setType(next);
        }
        applyFloat(
            binding,
            eid,
            mask,
            "linearDamping",
            body,
            "getLinearDamping",
            "setLinearDamping",
            "nonnegative",
            warned,
        );
        applyFloat(
            binding,
            eid,
            mask,
            "angularDamping",
            body,
            "getAngularDamping",
            "setAngularDamping",
            "nonnegative",
            warned,
        );
        applyFloat(
            binding,
            eid,
            mask,
            "gravityScale",
            body,
            "getGravityScale",
            "setGravityScale",
            "finite",
            warned,
        );
        applyFloat(
            binding,
            eid,
            mask,
            "sleepThreshold",
            body,
            "getSleepThreshold",
            "setSleepThreshold",
            "any",
            warned,
        );
        if (changed(binding, mask, "motionLocks")) {
            const next = value(binding, eid, "motionLocks");
            if ((next & ~0x3f) !== 0) warnInvalid(warned, binding, eid, "motionLocks");
            else if (next !== body.getMotionLocks()) body.setMotionLocks(next);
        }
        applyBoolean(binding, eid, mask, "enableSleep", body, "isSleepEnabled", "enableSleep");
        applyBoolean(binding, eid, mask, "isBullet", body, "isBullet", "setBullet");
        applyBoolean(
            binding,
            eid,
            mask,
            "allowFastRotation",
            body,
            "isFastRotationAllowed",
            "allowFastRotation",
        );
        applyBoolean(
            binding,
            eid,
            mask,
            "enableContactRecycling",
            body,
            "isContactRecyclingEnabled",
            "enableContactRecycling",
        );
        if (changed(binding, mask, "isEnabled")) {
            const next = value(binding, eid, "isEnabled") !== 0;
            if (next !== body.isEnabled()) {
                if (next) body.enable();
                else body.disable();
            }
        }
    }
}
