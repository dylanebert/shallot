import { DiagnosticsStore, type System, type World } from "@dylanebert/shallot";
import { network } from "./runtime";

export function hud(container: HTMLElement, world: World) {
    const topology = network(world);
    const panel = document.createElement("section");
    panel.className = "ec controls hud";
    panel.ariaLabel = "Network simulator";
    const fields = document.createElement("div");
    fields.className = "fields";
    const mode = document.createElement("span");
    mode.textContent = topology.faults ? "In-tab server · thin client" : "WebSocket server";
    fields.append(mode);
    for (const [id, label, max, step, value, unit, change] of [
        [
            "latency",
            "Round trip",
            300,
            10,
            100,
            "ms",
            (value: number) => {
                topology.faults!.minDelay = value / 2;
                topology.faults!.maxDelay = value / 2;
            },
        ],
        [
            "loss",
            "Packet loss",
            10,
            1,
            2,
            "%",
            (value: number) => {
                topology.faults!.loss = value / 100;
            },
        ],
    ] as const) {
        const row = document.createElement("div");
        row.className = "row";
        const caption = document.createElement("label");
        caption.htmlFor = id;
        caption.textContent = label;
        const slider = document.createElement("div");
        slider.className = "slider";
        const input = document.createElement("input");
        input.id = id;
        input.type = "range";
        input.min = "0";
        input.max = String(max);
        input.step = String(step);
        input.value = String(value);
        input.disabled = !topology.faults;
        input.tabIndex = -1;
        input.onfocus = () => input.blur();
        const output = document.createElement("output");
        output.htmlFor.add(id);
        output.value = topology.faults ? `${value}${unit}` : "—";
        input.oninput = () => {
            output.value = `${input.value}${unit}`;
            change(Number(input.value));
        };
        slider.append(input, output);
        row.append(caption, slider);
        fields.append(row);
    }
    const measured = document.createElement("div");
    measured.className = "row";
    const caption = document.createElement("span");
    caption.textContent = "Measured RTT";
    const output = document.createElement("output");
    output.setAttribute("data-measured-rtt", "");
    measured.append(caption, output);
    fields.insertBefore(measured, fields.children[2]!);
    panel.append(fields);
    container.append(panel);
    let lastSample: number | undefined;
    let changed = performance.now();
    const update = () => {
        const store = world.resource(DiagnosticsStore);
        const sample = store.get("network/round_trip")?.measurement();
        const now = performance.now();
        if (sample?.time !== lastSample) {
            lastSample = sample?.time;
            changed = now;
        }
        const stale = (store.get("network/time_since_receive")?.value() ?? 0) > 1000;
        output.textContent =
            !sample || now - changed > 1000
                ? "no data"
                : `${store.get("network/round_trip")!.smoothed()!.toFixed(2)}ms${stale ? " · stale" : ""}`;
    };
    const system: System = { name: "NetworkHud", group: "draw", last: true, update };
    world.addSystem(system);
    update();
    return () => {
        world.removeSystem(system);
        panel.remove();
    };
}
