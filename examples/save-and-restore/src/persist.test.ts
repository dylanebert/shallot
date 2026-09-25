import { resolve } from "node:path";
import {
    build,
    InputPlugin,
    pressKey,
    serialize,
    stringify,
    Time,
    Transform,
    TransformsPlugin,
} from "@dylanebert/shallot";
import { check } from "@dylanebert/shallot/harness/check";
import Persist from "./persist";

const SCENE = resolve(import.meta.dir, "../public/scenes/save-and-restore.scene");
const KEY = "shallot:save-and-restore";

check(
    "save-and-restore restores the authored scene through S and L",
    {
        claim: "save-and-restore saves the actual authored scene on S and L restores its node tree after edits",
    },
    async () => {
        const values = new Map<string, string>();
        const previousStorage = Object.getOwnPropertyDescriptor(globalThis, "localStorage");
        Object.defineProperty(globalThis, "localStorage", {
            configurable: true,
            value: {
                getItem: (key: string) => values.get(key) ?? null,
                setItem: (key: string, value: string) => values.set(key, String(value)),
            },
        });
        let app: Awaited<ReturnType<typeof build>> | undefined;
        try {
            app = await build({
                defaults: false,
                plugins: [InputPlugin, TransformsPlugin, Persist],
                scene: SCENE,
            });
            const { state } = app;
            const hero = [...state.identity.authored].find(
                (eid) => state.identity.id(eid) === "hero",
            );
            const ground = [...state.identity.authored].find(
                (eid) => state.identity.id(eid) === "ground",
            );
            if (hero === undefined || ground === undefined)
                throw new Error("actual scene is missing its authored hero or ground");

            pressKey(state, "KeyS");
            state.step(Time.FIXED_DT);
            const saved = values.get(KEY);
            if (!saved) throw new Error("S did not save the scene to localStorage");

            Transform.pos.x.set(hero, 12);
            state.destroy(ground);
            if (stringify(serialize(state)) === saved)
                throw new Error("authored world did not change before L");
            pressKey(state, "KeyL");
            state.step(Time.FIXED_DT);
            if (stringify(serialize(state)) !== saved)
                throw new Error("L did not restore the saved node tree");
        } finally {
            app?.dispose();
            if (previousStorage) Object.defineProperty(globalThis, "localStorage", previousStorage);
            else Reflect.deleteProperty(globalThis, "localStorage");
        }
    },
);
