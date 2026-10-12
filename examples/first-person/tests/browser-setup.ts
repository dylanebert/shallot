import { chromium } from "playwright/test";
import { assertChromiumInstalled } from "../chromium";

export default function browserSetup(): void {
    assertChromiumInstalled(chromium.executablePath());
}
