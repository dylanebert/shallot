// The gate's one prerequisite: a display. It runs headed because that is the only launch that reaches
// real GPU hardware — measured 2026-09-08 on an Omarchy/Hyprland seat (RTX 4090, driver 610.57.04),
// headed system Chrome over a localhost origin reports `nvidia / lovelace` while headless reports
// `google / swiftshader`, or no adapter at all, under every flag set tried. With no display there is no
// headed launch, so this refuses the run by name instead of letting Playwright fall back to a software
// adapter that the drivers would then skip on — a skip reads as a green run that verified nothing.

export default function globalSetup(): void {
    if (process.platform === "linux" && !(process.env.DISPLAY || process.env.WAYLAND_DISPLAY)) {
        throw new Error(
            "sandbox gate: no display (neither DISPLAY nor WAYLAND_DISPLAY is set) — this gate launches a headed browser and refuses to run without one",
        );
    }
}
