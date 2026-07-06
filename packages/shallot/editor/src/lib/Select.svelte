<script lang="ts">
    import { tick } from "svelte";
    import { fit, type Rect } from "./place";
    import { opensMenu, step, typeahead } from "./select";

    interface Option {
        label: string;
        value: number;
    }

    // a themed dropdown replacing the native `<select>`: the menu is portaled to `.editor` and pinned
    // on-screen via `fit`, so it can't open off-screen or paint unthemed the way the native popup does.
    let {
        options,
        value,
        onchange,
        title,
        variant = "field",
        placeholder,
    }: {
        options: Option[];
        value: number;
        onchange: (value: number) => void;
        title?: string;
        variant?: "field" | "unit";
        /** shown instead of the selected label — a multi-select with differing values passes `—`. */
        placeholder?: string;
    } = $props();

    let open = $state(false);
    let active = $state(0);
    let width = $state(0);
    let btnEl: HTMLButtonElement;
    let menuEl = $state.raw<HTMLElement | null>(null);
    let anchor = $state.raw<Rect>({ left: 0, top: 0, right: 0, bottom: 0 });

    const current = $derived(options.find((o) => o.value === value));
    const labels = $derived(options.map((o) => o.label));

    function openMenu() {
        const r = btnEl.getBoundingClientRect();
        anchor = { left: r.left, top: r.top, right: r.right, bottom: r.bottom };
        width = r.width;
        active = Math.max(0, options.findIndex((o) => o.value === value));
        open = true;
    }

    function close(refocus = true) {
        open = false;
        if (refocus) btnEl?.focus();
    }

    function commit(i: number) {
        const o = options[i];
        if (o && o.value !== value) onchange(o.value);
        close();
    }

    function onTriggerKey(e: KeyboardEvent) {
        if (opensMenu(e.key)) {
            e.preventDefault();
            openMenu();
        } else if (e.key.length === 1) {
            const idx = typeahead(labels, e.key, options.findIndex((o) => o.value === value));
            if (idx >= 0) commit(idx);
        }
    }

    function onMenuKey(e: KeyboardEvent) {
        const s = step(e.key, active, options.length);
        if (s) {
            e.preventDefault();
            active = s.active;
            if (s.commit) commit(active);
            else if (s.close) close();
        } else if (e.key.length === 1) {
            const idx = typeahead(labels, e.key, active);
            if (idx >= 0) active = idx;
        }
    }

    // outside-pointer close, keyed on element identity (not a selector) so a second Select sharing the
    // trigger/menu class still closes this one (Escape/Tab close from the menu's own keydown).
    $effect(() => {
        if (!open) return;
        tick().then(() => menuEl?.focus());
        function onDown(e: PointerEvent) {
            const t = e.target as Node;
            if (btnEl.contains(t) || menuEl?.contains(t)) return;
            close(false);
        }
        window.addEventListener("pointerdown", onDown, true);
        return () => window.removeEventListener("pointerdown", onDown, true);
    });
</script>

<button
    bind:this={btnEl}
    type="button"
    class="select-trigger {variant}"
    class:open
    {title}
    aria-haspopup="listbox"
    aria-expanded={open}
    onclick={() => (open ? close() : openMenu())}
    onkeydown={onTriggerKey}
>
    <span class="select-value">{placeholder ?? current?.label ?? ""}</span>
    <svg class="caret" viewBox="0 0 8 6" fill="none" stroke="currentColor" stroke-width="1.3">
        <path d="M0.75 1 L4 4.5 L7.25 1" />
    </svg>
</button>

{#if open}
    <div
        bind:this={menuEl}
        class="select-menu"
        role="listbox"
        tabindex="-1"
        style="min-width: {width}px"
        use:fit={{ anchor, align: "start" }}
        onkeydown={onMenuKey}
    >
        {#each options as opt, i (opt.value)}
            <button
                type="button"
                role="option"
                tabindex="-1"
                class="select-option"
                class:active={i === active}
                aria-selected={opt.value === value}
                onpointerenter={() => (active = i)}
                onclick={() => commit(i)}
            >
                <span class="so-label">{opt.label}</span>
                {#if opt.value === value}
                    <svg class="so-check" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M3 8.5 L6.5 12 L13 4" />
                    </svg>
                {/if}
            </button>
        {/each}
    </div>
{/if}

<style>
    /* the field variant matches the value inputs' box exactly (no layout shift vs the number field
       beside it); the unit variant is the borderless, transparent appendage on the right of a number
       field — both are flex trigger + caret, the menu is shared. */
    .select-trigger {
        box-sizing: border-box;
        display: flex;
        align-items: center;
        gap: 6px;
        cursor: pointer;
        outline: none;
        font-family: "JetBrains Mono", monospace;
    }

    .select-trigger.field {
        width: 100%;
        min-width: 0;
        height: 22px;
        padding: 0 6px 0 8px;
        border: 1px solid var(--border);
        border-radius: 4px;
        background: var(--bg);
        color: var(--text);
        font-size: 11px;
        transition: border-color 150ms var(--ease-out), background 150ms var(--ease-out),
            box-shadow 150ms var(--ease-out);
    }

    .select-trigger.field:hover {
        border-color: var(--surface-3);
    }

    .select-trigger.field.open,
    .select-trigger.field:focus-visible {
        border-color: var(--accent);
        background: var(--surface-4);
        box-shadow: 0 0 0 2px color-mix(in srgb, var(--accent) 15%, transparent);
    }

    .select-trigger.unit {
        flex: 0 0 auto;
        height: 22px;
        padding: 0 6px 0 5px;
        border: none;
        background: transparent;
        color: var(--text-muted);
        font-size: 10px;
        font-weight: 600;
    }

    .select-trigger.unit:hover,
    .select-trigger.unit.open {
        color: var(--text-secondary);
    }

    .select-value {
        flex: 1;
        min-width: 0;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
        text-align: left;
    }

    .select-trigger.unit .select-value {
        flex: 0 1 auto;
    }

    .caret {
        flex-shrink: 0;
        width: 8px;
        height: 6px;
        color: var(--text-muted);
        transition: transform 150ms var(--ease-out);
    }

    .select-trigger.open .caret {
        transform: rotate(180deg);
    }

    .select-menu {
        z-index: 100;
        max-height: 280px;
        overflow-y: auto;
        padding: 4px;
        background: var(--surface-3-solid);
        border: 1px solid var(--border);
        border-radius: 6px;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.35);
        outline: none;
        animation: select-appear 150ms var(--ease-out);
    }

    @keyframes select-appear {
        from {
            opacity: 0;
            transform: translateY(-4px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    .select-option {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 10px;
        width: 100%;
        height: 24px;
        padding: 0 8px;
        border: none;
        border-radius: 4px;
        background: transparent;
        color: var(--text-secondary);
        font-size: 11px;
        font-family: "JetBrains Mono", monospace;
        text-align: left;
        cursor: pointer;
    }

    .select-option.active {
        background: color-mix(in srgb, var(--accent) 14%, transparent);
        color: var(--text);
    }

    .select-option[aria-selected="true"] {
        color: var(--accent);
    }

    .so-label {
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }

    .so-check {
        flex-shrink: 0;
        width: 12px;
        height: 12px;
        color: var(--accent);
    }
</style>
