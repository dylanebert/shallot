import { OrbitPlugin, run, SkyPlugin } from "@dylanebert/shallot";
import { installHarness } from "@dylanebert/shallot/harness";
import { OceanPlugin } from "@dylanebert/shallot-ocean";
import { CAPTURE, sunDirection } from "./conditions";

const sun = sunDirection(CAPTURE.sunAzimuthOffset, CAPTURE.sunElevation);
const scene = `<scene>
    <a ambient-light="color: 0x9db7ca; intensity: 0.25" />
    <a directional-light="direction: ${sun.join(" ")}; color: 0xfff3dc; intensity: 1.2" />
    <a sky="zenith: 0x5f8fbd; horizon: 0xb8c8d3; sun-glow: 0.7; cloud-coverage: 0.45; haze-density: 0.004" />
    <a camera sear backdrop="name: sky" orbit="distance: ${CAPTURE.camera.distance}; yaw: ${CAPTURE.camera.yaw}; pitch: ${CAPTURE.camera.pitch}" transform />
    <a part="mesh: oceanGrid; surface: ocean" transform />
</scene>`;

const { state, dispose } = await run({
    scene,
    plugins: [OrbitPlugin, SkyPlugin, OceanPlugin],
    pixelRatio: 1,
});

const harness = installHarness(state);
harness.run = async () => {
    while (state.time.elapsed < CAPTURE.time) {
        await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));
    }
    state.pause();
    const canvas = document.querySelector("canvas");
    const checks = [
        {
            name: "matched 1280x720 capture viewport",
            ok: innerWidth === CAPTURE.width && innerHeight === CAPTURE.height,
            detail: `${innerWidth}x${innerHeight}`,
        },
        { name: "matched ocean time reached", ok: state.time.elapsed >= CAPTURE.time },
        { name: "ocean canvas rendered", ok: Boolean(canvas?.width && canvas.height) },
    ];
    return { ok: checks.every((check) => check.ok), checks };
};

if (import.meta.hot) import.meta.hot.dispose(dispose);
