// The one public capture contract. It fixes the viewport, device scale, target surface, presentation
// boundary and tightly packed RGBA semantics, so every semantic check, artifact and human frame reads the
// same geometry. `captureFrame` runs IN THE PAGE; the driver fixes the viewport that makes the geometry
// hold and never re-implements the read.
//
// The route is the experiment's answer, not a preference. A WebGPU canvas hands its composited frame to
// `toDataURL` and to nothing else: at the fixed geometry on a real macOS adapter, `createImageBitmap(canvas)`
// and a 2D `drawImage(canvas)` both returned a correctly sized, completely blank frame (0 of 184,320
// expected tag pixels), while `toDataURL` returned the exact drawn region and repeated captures of one
// unchanged state were byte-identical. A blank frame at the right size is the defect this contract exists
// to prevent, so the cheaper-looking routes are not admissible.

/** the declared capture geometry and semantics. One contract, not a per-consumer option. */
export interface CaptureIdentity {
    width: number;
    height: number;
    deviceScale: number;
    surface: "final-canvas";
    encoding: "rgba8-tight";
}

/** the one declared capture contract every consumer reads. */
export const CAPTURE_CONTRACT: CaptureIdentity = {
    width: 1280,
    height: 720,
    deviceScale: 1,
    surface: "final-canvas",
    encoding: "rgba8-tight",
};

/** the identity string a verdict, artifact name or refusal reason carries. */
export function captureIdentityLabel(identity: CaptureIdentity): string {
    return `${identity.surface} ${identity.width}x${identity.height}@${identity.deviceScale} ${identity.encoding}`;
}

/** whether two capture identities are the same contract in every field. */
export function captureIdentityMatches(left: CaptureIdentity, right: CaptureIdentity): boolean {
    return (
        left.width === right.width &&
        left.height === right.height &&
        left.deviceScale === right.deviceScale &&
        left.surface === right.surface &&
        left.encoding === right.encoding
    );
}

/**
 * Refuse a surface whose geometry is not the declared contract. A changed viewport or device scale is a
 * different capture, so it refuses here rather than quietly producing pixels at another size.
 */
export function assertCaptureGeometry(
    width: number,
    height: number,
    contract: CaptureIdentity = CAPTURE_CONTRACT,
): void {
    if (width !== contract.width || height !== contract.height) {
        throw new Error(
            `capture refused: surface is ${width}x${height}, not the declared contract ${captureIdentityLabel(contract)}`,
        );
    }
}

/** one capture: tightly packed RGBA at the declared identity. */
export interface Capture {
    rgba: Uint8ClampedArray;
    width: number;
    height: number;
    identity: CaptureIdentity;
}

/**
 * Capture the final canvas as tightly packed RGBA at the declared contract, after the presentation
 * boundary. A WebGPU canvas holds its frame only until the task that rendered it ends, so the read happens
 * inside a frame callback queued after the engine's own.
 *
 * @example
 * ```
 * const shot = await captureFrame(document.querySelector("canvas")!);
 * const result = probePixels(shot.rgba, shot.width, shot.height, probe);
 * ```
 */
export async function captureFrame(
    canvas: HTMLCanvasElement,
    contract: CaptureIdentity = CAPTURE_CONTRACT,
): Promise<Capture> {
    assertCaptureGeometry(canvas.width, canvas.height, contract);
    const url = await new Promise<string>((done) =>
        requestAnimationFrame(() => done(canvas.toDataURL("image/png"))),
    );
    const bitmap = await createImageBitmap(await (await fetch(url)).blob());
    try {
        assertCaptureGeometry(bitmap.width, bitmap.height, contract);
        const surface = new OffscreenCanvas(bitmap.width, bitmap.height);
        const context = surface.getContext("2d");
        if (!context) throw new Error("capture refused: no 2d context for the capture surface");
        context.drawImage(bitmap, 0, 0);
        const image = context.getImageData(0, 0, surface.width, surface.height);
        return { rgba: image.data, width: image.width, height: image.height, identity: contract };
    } finally {
        bitmap.close();
    }
}

/**
 * Capture the final canvas as a PNG data URL, for a bounded failure artifact. Same surface and boundary as
 * {@link captureFrame}; only the encoding differs, so an artifact and a semantic check never disagree
 * about what was on screen.
 */
export async function captureArtifact(
    canvas: HTMLCanvasElement,
    contract: CaptureIdentity = CAPTURE_CONTRACT,
): Promise<string> {
    assertCaptureGeometry(canvas.width, canvas.height, contract);
    return new Promise<string>((done) =>
        requestAnimationFrame(() => done(canvas.toDataURL("image/png"))),
    );
}
