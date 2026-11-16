import { VisualCorrection } from "../../core/rendering";
import { GlobalTransform } from "../../core/transform";
import { createApp, Time } from "../../engine";
import { ClientInterpolation, InterpolationPlugin, InterpolationTarget } from ".";
import { multiply } from "./interpolation";

export let controlSink: object;
export function control() {
    controlSink = {};
}

export default async function create() {
    const app = await createApp({ defaults: false, plugins: [InterpolationPlugin] });
    const world = app.world;
    const interpolation = new ClientInterpolation(world);
    const remotes: number[] = [];
    for (let i = 0; i < 16; i++) {
        const eid = world.create();
        world.add(eid, GlobalTransform);
        world.add(eid, InterpolationTarget);
        remotes.push(eid);
    }
    const predicted = world.create();
    world.add(predicted, GlobalTransform, { translation: [10, 0, 0] });
    world.add(predicted, VisualCorrection);
    const previous = [0, 0, 0.01, Math.sqrt(1 - 0.01 ** 2)];
    const inverse = [0, 0, 0, 1];
    const error = [0, 0, 0, 1];
    const rotation = [0, 0, 0, 1];
    let tick = 0;
    const pose = world.storage(GlobalTransform);
    const correction = world.storage(VisualCorrection);
    return {
        step() {
            for (let i = 0; i < remotes.length; i++) {
                pose.translation.set(remotes[i]!, tick + i, 0, 0);
                pose.rotation.set(remotes[i]!, 0, Math.sin(tick * 0.01), 0, Math.cos(tick * 0.01));
            }
            interpolation.confirm(tick++, remotes);
            // Repeated confirmation errors keep a real offset decaying throughout every window.
            correction.translation.x.set(predicted, -0.25);
            const q = correction.rotation.column;
            rotation[0] = q[predicted * 4]!;
            rotation[1] = q[predicted * 4 + 1]!;
            rotation[2] = q[predicted * 4 + 2]!;
            rotation[3] = q[predicted * 4 + 3]!;
            multiply(previous, inverse, error);
            multiply(rotation, error, rotation);
            correction.rotation.set(
                predicted,
                rotation[0]!,
                rotation[1]!,
                rotation[2]!,
                rotation[3]!,
            );
            world.step(Time.FIXED_DT);
            if (
                pose.translation.column[predicted * 4] !== 10 ||
                !(
                    correction.translation.column[predicted * 4]! > -0.25 &&
                    correction.translation.column[predicted * 4]! < 0
                )
            )
                throw new Error("allocation subject lost its corrected fixed pose or draw decay");
        },
        dispose: () => app.dispose(),
    };
}
