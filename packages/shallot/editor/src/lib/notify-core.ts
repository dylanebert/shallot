export type Severity = "error" | "warning" | "info";

export interface Toast {
    id: number;
    severity: Severity;
    text: string;
}

// a labelled choice on a banner — the resolution affordances of a blocking state (an external-change
// conflict offering Reload vs Keep mine). A plain banner is dismiss-only; one with actions is resolvable.
export interface BannerAction {
    label: string;
    fn: () => void;
}

export interface Banner {
    id: string;
    severity: Severity;
    text: string;
    actions?: BannerAction[];
}

// a toast is the most recent signal, not a log — the browser console holds the history. Append the new
// toast and keep only the newest `max`, so a burst (repeated save/load failures) never towers up the
// viewport. Banners dedupe by id instead; a toast carries no key, so it caps by count.
export function pushToast(list: Toast[], entry: Toast, max: number): Toast[] {
    return [...list, entry].slice(-max);
}

// keyed by id: a re-raise (a second failed build) updates the live banner in place; the id is also the
// clear handle once the state resolves. Append only when the id is new, so the same condition never
// stacks duplicate banners.
export function upsertBanner(list: Banner[], entry: Banner): Banner[] {
    const i = list.findIndex((b) => b.id === entry.id);
    if (i === -1) return [...list, entry];
    const next = list.slice();
    next[i] = entry;
    return next;
}
