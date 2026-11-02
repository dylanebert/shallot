import type { World } from "@dylanebert/shallot";
import { manifoldDrag } from "../src/manifold-drag";
import { add, vector } from "../src/manifold-math";
import { ManifoldSample } from "../src/manifold-state";
export function manifoldDiagnosticGesture(world: World, step: number): void {
    const s = world.resource(ManifoldSample),
        translation = vector(0, 0, -20);
    if (step === 4) manifoldDrag(world, add(s.b.p, vector(0, 0, 10)), translation, true, true);
    else if (step >= 5 && step <= 12)
        manifoldDrag(
            world,
            add(s.base, vector((step - 4) * 0.125, (step - 4) * 0.0625, 10)),
            translation,
            false,
            true,
        );
    else if (step === 13) manifoldDrag(world, vector(), translation, false, false);
}
