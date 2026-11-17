import { expect, test } from "bun:test";
import { flyConfig } from "./publish-fly";

const config = () => Bun.TOML.parse(flyConfig("game")) as any;

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

test("Fly config pins one shared CPU", () => {
    expect(config().vm.map((vm: { size: string }) => vm.size)).toEqual(["shared-cpu-1x"]);
});

test("Fly config pins 256 MB", () => {
    expect(config().vm.map((vm: { memory: string }) => vm.memory)).toEqual(["256mb"]);
});
