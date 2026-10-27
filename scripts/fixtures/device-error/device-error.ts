import { setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { observedDeviceError } from "../../../src/engine/runtime";
import { gpuApps } from "../../gpu.fixture";

setDefaultTimeout(TIMEOUT.gpu);
const apps = gpuApps(import.meta.path, [{ defaults: false, plugins: [] }]);

test("raises an undeclared device error", () => {
    apps()[0].world.gpu.device.createBuffer({ size: 4, usage: 0 });
});

// Outlasts the device's report, which would otherwise land in this test.
if (process.env.SHALLOT_DEVICE_ERROR_FOLLOWED === "1")
    test("follows the error", () => observedDeviceError(apps()[0].world.gpu.device));
