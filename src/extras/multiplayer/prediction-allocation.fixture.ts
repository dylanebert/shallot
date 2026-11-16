import { Player, PlayerInput, PlayerPlugin } from "../../../examples/first-person/src/player";
import { Body, BodyType, Shape, ShapeKind } from "../../core/physics";
import { GlobalTransform, Transform } from "../../core/transform";
import { createApp } from "../../engine";
import { Character } from "../../standard/physics";
import {
    ClientPrediction,
    PredictionMotion,
    PredictionPlugin,
    PredictionTarget,
    Replicate,
    ReplicationSender,
    seededLink,
} from ".";

export let controlSink: object;
export function control() {
    controlSink = {};
}
export default async function create() {
    const app = await createApp({ defaults: false, plugins: [PlayerPlugin, PredictionPlugin] });
    const world = app.world;
    const camera = world.create();
    world.add(camera, Transform);
    const avatar = world.create();
    world.add(avatar, Transform, { translation: [0, 2, 0] });
    world.add(avatar, Body, { type: BodyType.Kinematic });
    world.add(avatar, Shape, { kind: ShapeKind.Capsule });
    world.add(avatar, Character);
    world.add(avatar, Player, { camera });
    world.add(avatar, PlayerInput);
    const box = world.create();
    world.add(box, Transform, { translation: [3, 2, 0] });
    world.add(box, Body, { type: BodyType.Dynamic });
    world.add(box, Shape);
    world.add(box, PredictionTarget);
    const remote = world.create();
    world.add(remote, Transform, { translation: [6, 2, 0] });
    world.add(remote, Body, { type: BodyType.Dynamic });
    world.add(remote, Shape);
    for (const eid of world.query([Transform])) world.add(eid, Replicate);
    world.tick();
    const link = seededLink(1, { minDelay: 0, maxDelay: 0, loss: 0, duplication: 0 });
    const schema = [Body, GlobalTransform, PredictionMotion];
    const prediction = new ClientPrediction(
        world,
        schema,
        link.b,
        () => avatar,
        (w, tick, _records, input) => {
            if (!input) throw new Error("Missing replay input");
            w.storage(PlayerInput).move.set(avatar, input.values[0]!, tick % 2 ? 1 : -1);
        },
    );
    for (const [index, eid] of [...world.query([Replicate])].entries())
        prediction.receiver.entities.set(index + 1, eid);
    const sender = new ReplicationSender(world, schema, [new Map([[1, schema]])], 1, link.a);
    sender.record(1);
    sender.flush();
    world.step(0);
    if (!prediction.predicted(avatar) || prediction.predicted(remote))
        throw new Error("allocation subject lost display selection");
    prediction.accept(
        Array.from({ length: 13 }, (_, index) => ({
            tick: index + 1,
            kind: "input" as const,
            values: [0],
        })),
    );
    const saved = world.snapshot();
    for (let i = 0; i < 12; i++) world.tick();
    let ticks = 0;
    return {
        step() {
            if (ticks++ % 12 === 0) world.restore(saved);
            world.tick();
        },
        allocationFunctions: [world.snapshot, world.restore],
        dispose: () => app.dispose(),
    };
}
