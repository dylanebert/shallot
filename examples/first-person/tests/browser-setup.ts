import { chromium } from "playwright/test";
import { assertChromiumInstalled } from "./browser-install";

export default function browserSetup(): void {
    assertChromiumInstalled(chromium.executablePath());
}
