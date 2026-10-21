import { config } from "../../playwright.config";

export default {
    ...config,
    testDir: ".",
    testIgnore: [],
    globalSetup: "../browser-setup.ts",
};
