import { expect, test } from "bun:test";
import { flyCommands, flyConfig } from "./publish-fly";

const config = () => Bun.TOML.parse(flyConfig("game")) as any;

test("Fly operations use the fetched CLI and put confirmation before deletion", () => {
    const commands = flyCommands("/cache with spaces/flyctl", "game");
    expect(commands).toContain('Server logs: "/cache with spaces/flyctl" logs --app game');
    expect(commands).toContain(
        "Server rollback: check out an earlier commit, then bunx shallot publish",
    );
    expect(commands).toContain('Server deletion: "/cache with spaces/flyctl" apps destroy game');
    expect(commands).toContain("Confirm with the person before deleting the server.");
    expect(commands.indexOf("Confirm with the person")).toBeLessThan(
        commands.indexOf("Server deletion:"),
    );
});

test("Fly config names the app", () => {
    expect(config().app).toBe("game");
});

test.each([
    ["env", "PORT", "8080"],
    ["http_service", "internal_port", 8080],
    ["http_service", "force_https", true],
    ["http_service", "auto_stop_machines", "stop"],
    ["http_service", "auto_start_machines", true],
    ["http_service", "min_machines_running", 0],
] as const)("Fly config pins %s.%s to %s", (section, field, value) => {
    expect(config()[section][field]).toBe(value);
});

test("Fly config places its idle Machine in the spike's primary region", () => {
    expect(config().primary_region).toBe("iad");
});

test("Fly mounts one minimum-size volume at the server data directory", () => {
    expect(config().mounts).toEqual({
        source: "shallot_data",
        destination: "/data",
        ["initial_size"]: "1GB",
        ["scheduled_snapshots"]: false,
    });
    expect(config().env.SHALLOT_DATA_DIR).toBe(config().mounts.destination);
});

test("Fly delivers the shutdown signal handled by the Bun room host", () => {
    expect(config().kill_signal).toBe("SIGTERM");
});

test("Fly config counts WebSocket connections for proxy load and autostop", () => {
    expect(config().http_service.concurrency.type).toBe("connections");
    expect(config().http_service.concurrency.soft_limit).toBe(20);
});

test("Fly config pins one shared CPU", () => {
    expect(config().vm.map((vm: { size: string }) => vm.size)).toEqual(["shared-cpu-1x"]);
});

test("Fly config leaves stopped Machines stopped until proxy traffic starts them", () => {
    expect(config().restart.map((restart: { policy: string }) => restart.policy)).toEqual([
        "never",
    ]);
});

test("Fly config pins 256 MB", () => {
    expect(config().vm.map((vm: { memory: string }) => vm.memory)).toEqual(["256mb"]);
});
