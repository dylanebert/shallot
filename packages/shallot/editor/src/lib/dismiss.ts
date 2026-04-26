export function dismissOnClickOutside(onDismiss: () => void, ...selectors: string[]): () => void {
    const dismiss = (e: Event) => {
        const target = e.target as HTMLElement;
        if (selectors.some((s) => target?.closest?.(s))) return;
        onDismiss();
    };
    const dismissKey = (e: KeyboardEvent) => {
        if (e.key === "Escape") onDismiss();
    };
    window.addEventListener("pointerdown", dismiss, true);
    window.addEventListener("contextmenu", dismiss, true);
    window.addEventListener("keydown", dismissKey);
    return () => {
        window.removeEventListener("pointerdown", dismiss, true);
        window.removeEventListener("contextmenu", dismiss, true);
        window.removeEventListener("keydown", dismissKey);
    };
}
