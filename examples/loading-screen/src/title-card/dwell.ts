export const MINIMUM_DWELL_MS = 1500;

export async function completeIntro(
    shownAt: number,
    intro: Promise<unknown>,
    now = () => performance.now(),
    wait = (ms: number) => new Promise<void>((resolve) => setTimeout(resolve, ms)),
): Promise<void> {
    await intro;
    await wait(Math.max(0, MINIMUM_DWELL_MS - (now() - shownAt)));
}
