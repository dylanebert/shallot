// One source of truth for placing a summoned surface (color picker, context menu, menu dropdown) so
// it never opens off-screen. Open off an anchor toward a preferred side; flip to the opposite side
// when that side would clip; clamp into the viewport so the panel is always fully visible. The math
// is a pure transform (`place`); consumers apply it through the `fit` action.

export interface Rect {
    left: number;
    top: number;
    right: number;
    bottom: number;
}

export interface PlaceOpts {
    /** main-axis side to open toward; flips to the opposite when it would clip (default "below") */
    side?: "below" | "above" | "right" | "left";
    /** cross-axis edge of the anchor to align the panel to (default "start") */
    align?: "start" | "end";
    /** distance from the anchor along the main axis (default 4) */
    gap?: number;
    /** minimum distance kept from every viewport edge (default 8) */
    margin?: number;
}

const clamp = (v: number, lo: number, hi: number) => (hi < lo ? lo : Math.min(Math.max(v, lo), hi));

// main axis: place toward the preferred side, flip to the side with more room when it clips, clamp on.
function main(
    near: number,
    far: number,
    size: number,
    vp: number,
    gap: number,
    margin: number,
    toFar: boolean,
): number {
    const afterFar = far + gap;
    const beforeNear = near - gap - size;
    const roomFar = vp - margin - afterFar;
    const roomNear = near - gap - margin;
    let start = toFar ? afterFar : beforeNear;
    const room = toFar ? roomFar : roomNear;
    if (room < size) {
        const other = toFar ? roomNear : roomFar;
        if (other > room) start = toFar ? beforeNear : afterFar;
    }
    return clamp(start, margin, vp - size - margin);
}

// cross axis: align the panel's near (or far) edge to the anchor's, clamp on.
function cross(
    near: number,
    far: number,
    size: number,
    vp: number,
    margin: number,
    end: boolean,
): number {
    return clamp(end ? far - size : near, margin, vp - size - margin);
}

export function place(
    panel: { width: number; height: number },
    anchor: Rect,
    viewport: { width: number; height: number },
    opts: PlaceOpts = {},
): { left: number; top: number } {
    const { side = "below", align = "start", gap = 4, margin = 8 } = opts;
    const end = align === "end";
    if (side === "below" || side === "above") {
        return {
            left: cross(anchor.left, anchor.right, panel.width, viewport.width, margin, end),
            top: main(
                anchor.top,
                anchor.bottom,
                panel.height,
                viewport.height,
                gap,
                margin,
                side === "below",
            ),
        };
    }
    return {
        left: main(
            anchor.left,
            anchor.right,
            panel.width,
            viewport.width,
            gap,
            margin,
            side === "right",
        ),
        top: cross(anchor.top, anchor.bottom, panel.height, viewport.height, margin, end),
    };
}

type FitParams = { anchor: Rect } & PlaceOpts;

// Svelte action: measure the rendered panel and pin it on-screen against its anchor, keeping it fitted
// as the viewport resizes or scrolls. The panel's CSS sets its size; this owns its position.
export function fit(node: HTMLElement, params: FitParams) {
    function apply() {
        const { anchor, ...opts } = params;
        // offsetWidth/Height is the layout box, immune to the entrance scale animation that
        // getBoundingClientRect would shrink on the first frame.
        const { left, top } = place(
            { width: node.offsetWidth, height: node.offsetHeight },
            anchor,
            { width: window.innerWidth, height: window.innerHeight },
            opts,
        );
        node.style.position = "fixed";
        node.style.left = `${left}px`;
        node.style.top = `${top}px`;
    }
    apply();
    window.addEventListener("resize", apply);
    window.addEventListener("scroll", apply, true);
    return {
        update(next: FitParams) {
            params = next;
            apply();
        },
        destroy() {
            window.removeEventListener("resize", apply);
            window.removeEventListener("scroll", apply, true);
        },
    };
}
