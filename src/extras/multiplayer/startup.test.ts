import { expect, test } from "bun:test";
import { World } from "../../engine";
import { ReplicationReceiver, ReplicationSender } from "./index";
import { seededLink } from "./link";

test.each([undefined, "wss://game.example/"])(
    "applied state identifies its transport endpoint %s without an overlay",
    (endpoint) => {
        const a = new World(),
            b = new World();
        const link = seededLink(1, { minDelay: 0, maxDelay: 0, loss: 0, duplication: 0 });
        const sender = new ReplicationSender(a, [], [], 1, link.a);
        const receiver = new ReplicationReceiver(b, [], { ...link.b, endpoint });
        const previous = Object.getOwnPropertyDescriptor(globalThis, "window");
        const browser = new EventTarget();
        const observations: unknown[] = [];
        browser.addEventListener("shallot:startup", (event) =>
            observations.push((event as CustomEvent).detail),
        );
        try {
            Object.defineProperty(globalThis, "window", { configurable: true, value: browser });
            receiver.flush();
            expect(observations).toEqual([]);
            sender.record(0);
            sender.flush();
            receiver.flush();
            sender.flush();
            receiver.flush();
            expect(observations).toEqual([{ kind: "server-state", tick: 0, endpoint }]);
            sender.record(1);
            sender.flush();
            receiver.flush();
            expect(observations).toEqual([
                { kind: "server-state", tick: 0, endpoint },
                { kind: "server-state", tick: 1, endpoint },
            ]);
        } finally {
            if (previous) Object.defineProperty(globalThis, "window", previous);
            else Reflect.deleteProperty(globalThis, "window");
            a.dispose();
            b.dispose();
        }
    },
);
