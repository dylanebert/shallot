/** an easing curve: maps normalized progress (0→1) to an eased value (0→1, may overshoot for back/elastic) */
export type Easing = (t: number) => number;

const linear: Easing = (t) => t;

const easeInQuad: Easing = (t) => t * t;
const easeOutQuad: Easing = (t) => t * (2 - t);
const easeInOutQuad: Easing = (t) => (t < 0.5 ? 2 * t * t : -1 + (4 - 2 * t) * t);

const easeInCubic: Easing = (t) => t * t * t;
const easeOutCubic: Easing = (t) => --t * t * t + 1;
const easeInOutCubic: Easing = (t) =>
    t < 0.5 ? 4 * t * t * t : (t - 1) * (2 * t - 2) * (2 * t - 2) + 1;

const easeInQuart: Easing = (t) => t * t * t * t;
const easeOutQuart: Easing = (t) => 1 - --t * t * t * t;
const easeInOutQuart: Easing = (t) => (t < 0.5 ? 8 * t * t * t * t : 1 - 8 * --t * t * t * t);

const easeInQuint: Easing = (t) => t * t * t * t * t;
const easeOutQuint: Easing = (t) => 1 + --t * t * t * t * t;
const easeInOutQuint: Easing = (t) =>
    t < 0.5 ? 16 * t * t * t * t * t : 1 + 16 * --t * t * t * t * t;

const easeInSine: Easing = (t) => 1 - Math.cos((t * Math.PI) / 2);
const easeOutSine: Easing = (t) => Math.sin((t * Math.PI) / 2);
const easeInOutSine: Easing = (t) => -(Math.cos(Math.PI * t) - 1) / 2;

const easeInExpo: Easing = (t) => (t === 0 ? 0 : Math.pow(2, 10 * t - 10));
const easeOutExpo: Easing = (t) => (t === 1 ? 1 : 1 - Math.pow(2, -10 * t));
const easeInOutExpo: Easing = (t) =>
    t === 0
        ? 0
        : t === 1
          ? 1
          : t < 0.5
            ? Math.pow(2, 20 * t - 10) / 2
            : (2 - Math.pow(2, -20 * t + 10)) / 2;

const easeInCirc: Easing = (t) => 1 - Math.sqrt(1 - t * t);
const easeOutCirc: Easing = (t) => Math.sqrt(1 - --t * t);
const easeInOutCirc: Easing = (t) =>
    t < 0.5
        ? (1 - Math.sqrt(1 - 4 * t * t)) / 2
        : (Math.sqrt(1 - (-2 * t + 2) * (-2 * t + 2)) + 1) / 2;

const easeInBack: Easing = (t) => {
    const c = 1.70158;
    return (c + 1) * t * t * t - c * t * t;
};
const easeOutBack: Easing = (t) => {
    const c = 1.70158;
    return 1 + (c + 1) * Math.pow(t - 1, 3) + c * Math.pow(t - 1, 2);
};
const easeInOutBack: Easing = (t) => {
    const c = 1.70158 * 1.525;
    return t < 0.5
        ? (Math.pow(2 * t, 2) * ((c + 1) * 2 * t - c)) / 2
        : (Math.pow(2 * t - 2, 2) * ((c + 1) * (t * 2 - 2) + c) + 2) / 2;
};

const easeInElastic: Easing = (t) =>
    t === 0
        ? 0
        : t === 1
          ? 1
          : -Math.pow(2, 10 * t - 10) * Math.sin((t * 10 - 10.75) * ((2 * Math.PI) / 3));
const easeOutElastic: Easing = (t) =>
    t === 0
        ? 0
        : t === 1
          ? 1
          : Math.pow(2, -10 * t) * Math.sin((t * 10 - 0.75) * ((2 * Math.PI) / 3)) + 1;
const easeInOutElastic: Easing = (t) =>
    t === 0
        ? 0
        : t === 1
          ? 1
          : t < 0.5
            ? -(Math.pow(2, 20 * t - 10) * Math.sin((20 * t - 11.125) * ((2 * Math.PI) / 4.5))) / 2
            : (Math.pow(2, -20 * t + 10) * Math.sin((20 * t - 11.125) * ((2 * Math.PI) / 4.5))) /
                  2 +
              1;

const easeOutBounce: Easing = (t) => {
    const n1 = 7.5625;
    const d1 = 2.75;
    if (t < 1 / d1) return n1 * t * t;
    if (t < 2 / d1) return n1 * (t -= 1.5 / d1) * t + 0.75;
    if (t < 2.5 / d1) return n1 * (t -= 2.25 / d1) * t + 0.9375;
    return n1 * (t -= 2.625 / d1) * t + 0.984375;
};
const easeInBounce: Easing = (t) => 1 - easeOutBounce(1 - t);
const easeInOutBounce: Easing = (t) =>
    t < 0.5 ? (1 - easeOutBounce(1 - 2 * t)) / 2 : (1 + easeOutBounce(2 * t - 1)) / 2;

/** the built-in easing curves, indexed by easing id (parallel to the kebab names `getEasingIndex` resolves) */
export const EASING_FUNCTIONS: readonly Easing[] = [
    linear,
    easeInQuad,
    easeOutQuad,
    easeInOutQuad,
    easeInCubic,
    easeOutCubic,
    easeInOutCubic,
    easeInQuart,
    easeOutQuart,
    easeInOutQuart,
    easeInQuint,
    easeOutQuint,
    easeInOutQuint,
    easeInSine,
    easeOutSine,
    easeInOutSine,
    easeInExpo,
    easeOutExpo,
    easeInOutExpo,
    easeInCirc,
    easeOutCirc,
    easeInOutCirc,
    easeInBack,
    easeOutBack,
    easeInOutBack,
    easeInElastic,
    easeOutElastic,
    easeInOutElastic,
    easeInBounce,
    easeOutBounce,
    easeInOutBounce,
] as const;

const EASING_INDEX: Record<string, number> = {
    linear: 0,
    "ease-in-quad": 1,
    "ease-out-quad": 2,
    "ease-in-out-quad": 3,
    "ease-in-cubic": 4,
    "ease-out-cubic": 5,
    "ease-in-out-cubic": 6,
    "ease-in-quart": 7,
    "ease-out-quart": 8,
    "ease-in-out-quart": 9,
    "ease-in-quint": 10,
    "ease-out-quint": 11,
    "ease-in-out-quint": 12,
    "ease-in-sine": 13,
    "ease-out-sine": 14,
    "ease-in-out-sine": 15,
    "ease-in-expo": 16,
    "ease-out-expo": 17,
    "ease-in-out-expo": 18,
    "ease-in-circ": 19,
    "ease-out-circ": 20,
    "ease-in-out-circ": 21,
    "ease-in-back": 22,
    "ease-out-back": 23,
    "ease-in-out-back": 24,
    "ease-in-elastic": 25,
    "ease-out-elastic": 26,
    "ease-in-out-elastic": 27,
    "ease-in-bounce": 28,
    "ease-out-bounce": 29,
    "ease-in-out-bounce": 30,
};

// reverse of EASING_INDEX, so the scene `easing` trait formats an index back to
// its kebab name and round-trips
const EASING_NAMES: string[] = Object.entries(EASING_INDEX).reduce((names, [name, index]) => {
    names[index] = name;
    return names;
}, [] as string[]);

/** the easing id for a kebab name (`ease-out-quad`); 0 (linear) for an unknown name */
export function getEasingIndex(name: string): number {
    return EASING_INDEX[name] ?? 0;
}

/** the kebab name for an easing id; `linear` for an unknown id */
export function getEasingName(index: number): string {
    return EASING_NAMES[index] ?? "linear";
}

/** the easing curve for an easing id; linear for an unknown id */
export function getEasing(index: number): Easing {
    return EASING_FUNCTIONS[index] ?? linear;
}
