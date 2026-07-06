import {
    type Banner,
    type BannerAction,
    pushToast,
    type Severity,
    type Toast,
    upsertBanner,
} from "./notify-core";

export type { Banner, BannerAction, Severity, Toast } from "./notify-core";

// the editor band, curated. Two transient/persistent surfaces; the issue *set* is the live
// `diagnose(doc)` derivation App already owns, not a stream this collects. The runtime/console
// firehose stays in the browser console — only an editor-vocabulary signal lands here.
const TOAST_MS = 4000;
const MAX_TOASTS = 3;

let _toasts = $state<Toast[]>([]);
let _banners = $state<Banner[]>([]);
let _nextId = 0;

export function getToasts(): Toast[] {
    return _toasts;
}

export function getBanners(): Banner[] {
    return _banners;
}

/** flash a transient message; auto-dismisses. action feedback (saved, export) + the runtime-error pointer. */
export function toast(severity: Severity, text: string): number {
    const id = _nextId++;
    _toasts = pushToast(_toasts, { id, severity, text }, MAX_TOASTS);
    setTimeout(() => dismissToast(id), TOAST_MS);
    return id;
}

export function dismissToast(id: number) {
    _toasts = _toasts.filter((t) => t.id !== id);
}

/**
 * raise or replace a persistent blocking banner, keyed by id; clear it with the same id when resolved.
 * Pass `actions` for a state the user resolves in place (a conflict's Reload / Keep mine), not just dismiss.
 */
export function banner(id: string, severity: Severity, text: string, actions?: BannerAction[]) {
    _banners = upsertBanner(_banners, { id, severity, text, actions });
}

export function clearBanner(id: string) {
    _banners = _banners.filter((b) => b.id !== id);
}
