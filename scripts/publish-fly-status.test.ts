import { expect, test } from "bun:test";
import { flyMachineStartedAt } from "./publish-fly-status";

// Event rows captured from pinned flyctl in run 44212650002; no app identifiers are needed.
const events = `\u001b[1mEvent Logs\u001b[0m
 STATE    │ EVENT    │ SOURCE │ TIMESTAMP                │ INFO
 started  │ start    │ flyd   │ 2026-11-18T01:16:07.227Z │
 starting │ start    │ proxy  │ 2026-11-18T01:16:06.517Z │
 stopped  │ uncordon │ flyd   │ 2026-11-18T01:16:05.762Z │
 stopped  │ exit     │ flyd   │ 2026-11-18T01:16:03.111Z │ exit_code=130,oom_killed=false,requested_stop=true
 stopping │ stop     │ proxy  │ 2026-11-18T01:16:02.767Z │`;

test("pinned Machine status separates event columns with vertical rules, not whitespace", () => {
    expect(flyMachineStartedAt(events)).toBe(Date.parse("2026-11-18T01:16:07.227Z"));
    const older = " started │ start │ flyd │ 2026-11-18T01:09:25Z │";
    expect(flyMachineStartedAt(`${older}\n${events}`)).toBe(flyMachineStartedAt(events));
    expect(flyMachineStartedAt(`${events}\n${older}`)).toBe(flyMachineStartedAt(events));
});

test.each(["State: started", " started │ start │ flyd │ unknown │"])(
    "missing/invalid event timestamps name the telemetry failure: %s",
    (status) => {
        expect(() => flyMachineStartedAt(status)).toThrow(
            "started-event timestamps are missing or invalid",
        );
    },
);
