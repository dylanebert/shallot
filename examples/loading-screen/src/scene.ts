// Neutral + sRGB inverse for --bg2 (#1c1917): authored 0x383736 displays as #1b1917, one red level lower after 8-bit quantization.
const clearColor = "0x383736";

export const SCENE = `<scene>
    <a ambient-light="color: 0xd0dcec; intensity: 0.5" />
    <a directional-light="direction: -0.4 -1 -0.55; color: 0xfff4e0; intensity: 1.1" />
    <a camera="clear-color: ${clearColor}" standard-renderer orbit="distance: 5; yaw: 0.6; pitch: 0.25" transform />
    <a mesh-instance transform="translation: 0 0 0" color="rgba: 0.85 0.55 0.35 1" />
</scene>`;
