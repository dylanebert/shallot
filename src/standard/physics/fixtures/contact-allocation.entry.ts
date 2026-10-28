import { Transform } from "../../../core/transform";
import { createApp } from "../../../engine";
import { contactPair } from "../contact-data.fixture";
import { getContactData, StandardPhysicsPlugin } from "../index";

export let controlSink: { frame: number } | undefined;
export function control() {
    controlSink = { frame: 0 };
}

export default async function create(input: string) {
    const app = await createApp({ defaults: false, plugins: [StandardPhysicsPlugin] });
    const world = app.world;
    const subjects = [1, 0, 2].map((scene) => {
        const [a, b] = contactPair(world, scene);
        world.add(a, Transform, { translation: [scene * 50, 0, 0] });
        world.storage(Transform).translation.x.set(b, scene * 50);
        return { a, b, children: scene === 2 ? 2 : 1 };
    });
    world.tick();
    for (const system of StandardPhysicsPlugin.systems!) world.removeSystem(system);
    world.addSystem({
        name: "read-contacts",
        group: "fixed",
        update(world) {
            for (const subject of subjects) {
                for (let child = 0; child < subject.children; child++) {
                    const data = getContactData(world, subject.a, subject.b, child);
                    if (!data?.manifoldCount)
                        throw new Error("allocation subject lost its contact");
                    if (!getContactData(world, subject.b, subject.a, child))
                        throw new Error("reverse contact missing");
                }
                if (getContactData(world, subject.a, subject.b, 99))
                    throw new Error("absent child produced contact data");
            }
        },
    });
    return {
        step() {
            world.tick();
            if (input === "allocating") control();
        },
        dispose() {
            app.dispose();
        },
    };
}
