<script lang="ts">
    import { valueToNorm as _valueToNorm, normToValue as _normToValue } from "./presets";

    let {
        label,
        value = $bindable(),
        min = 0,
        max = 1,
        step = 0.01,
        scale,
        fmt,
        parse,
        defaultValue,
        onchange,
    }: {
        label: string;
        value: number;
        min?: number;
        max?: number;
        step?: number;
        scale?: "log";
        fmt?: (v: number) => string;
        parse?: (s: string) => number | undefined;
        defaultValue?: number;
        onchange?: (value: number) => void;
    } = $props();

    let dragging = $state(false);
    let editing = $state(false);
    let editText = $state("");
    let menuOpen = $state(false);
    let menuX = $state(0);
    let menuY = $state(0);

    function oncontextmenu(e: MouseEvent) {
        if (defaultValue === undefined) return;
        e.preventDefault();
        menuOpen = true;
        menuX = e.clientX;
        menuY = e.clientY;

        function close() {
            menuOpen = false;
            window.removeEventListener("pointerdown", close);
        }
        setTimeout(() => window.addEventListener("pointerdown", close), 0);
    }

    function resetToDefault() {
        if (defaultValue !== undefined) {
            value = defaultValue;
            onchange?.(value);
        }
        menuOpen = false;
    }

    function valueToNorm(v: number): number {
        return _valueToNorm(v, min, max, scale ?? "linear");
    }

    function normToValue(n: number): number {
        return _normToValue(n, min, max, scale ?? "linear", step);
    }

    function preciseString(v: number): string {
        const decimals = step < 1 ? Math.max(0, Math.ceil(-Math.log10(step))) : 0;
        return v.toFixed(decimals);
    }

    const display = $derived(fmt ? fmt(value) : preciseString(value));
    const fillPct = $derived(valueToNorm(value) * 100);

    function onpointerdown(e: PointerEvent) {
        if (editing) return;
        const el = e.currentTarget as HTMLElement;
        el.setPointerCapture(e.pointerId);
        dragging = true;
        let norm = valueToNorm(value);

        function onmove(ev: PointerEvent) {
            const fine = ev.shiftKey ? 0.1 : 1;
            norm += (ev.movementX / 300) * fine;
            norm = Math.max(0, Math.min(1, norm));
            value = Math.max(min, Math.min(max, normToValue(norm)));
            onchange?.(value);
        }

        function onup() {
            dragging = false;
            el.removeEventListener("pointermove", onmove);
            el.removeEventListener("pointerup", onup);
        }

        el.addEventListener("pointermove", onmove);
        el.addEventListener("pointerup", onup);
    }

    function focus(node: HTMLInputElement) {
        node.focus();
        node.select();
    }

    function startEdit() {
        editing = true;
        editText = fmt ? fmt(value) : preciseString(value);
    }

    function commitEdit() {
        let parsed: number | undefined;
        if (parse) parsed = parse(editText);
        if (parsed === undefined) {
            const text = editText.trim();
            const suffixes: [RegExp, number][] = [
                [/kHz$/i, 1000],
                [/Hz$/i, 1],
                [/ms$/i, 0.001],
                [/s$/i, 1],
                [/%$/i, 0.01],
            ];
            let num = NaN;
            let mul = 1;
            for (const [re, m] of suffixes) {
                if (re.test(text)) {
                    num = parseFloat(text.replace(re, ""));
                    mul = m;
                    break;
                }
            }
            if (isNaN(num)) num = parseFloat(text);
            else num *= mul;
            if (!isNaN(num)) parsed = num;
        }
        if (parsed !== undefined) {
            value = Math.max(min, Math.min(max, parsed));
            onchange?.(value);
        }
        editing = false;
    }

    function cancelEdit() {
        editing = false;
    }

    function onEditKeydown(e: KeyboardEvent) {
        if (e.key === "Enter") {
            e.preventDefault();
            commitEdit();
        } else if (e.key === "Escape") {
            e.preventDefault();
            cancelEdit();
        }
    }
</script>

<!-- svelte-ignore a11y_no_static_element_interactions -->
<div class="knob" class:dragging {oncontextmenu}>
    <span class="knob-label">{label}</span>
    <div class="knob-right">
        {#if editing}
            <input
                class="knob-input"
                type="text"
                bind:value={editText}
                onblur={commitEdit}
                onkeydown={onEditKeydown}
                use:focus
            />
        {:else}
            <span
                class="knob-value"
                {onpointerdown}
                ondblclick={startEdit}
                role="slider"
                tabindex="-1"
                aria-valuenow={value}
                aria-valuemin={min}
                aria-valuemax={max}
            >{display}</span>
        {/if}
        <div class="knob-fill" style:width="{fillPct}%"></div>
    </div>
</div>

{#if menuOpen}
    <div class="ctx-menu" style:left="{menuX}px" style:top="{menuY}px">
        <button class="ctx-item" onpointerdown={(e) => { e.stopPropagation(); resetToDefault(); }}>Reset</button>
    </div>
{/if}

<style>
    .knob {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 12px;
        padding: 4px 0;
        user-select: none;
    }

    .knob-label {
        color: var(--text-muted);
        font-size: 12px;
        cursor: default;
        flex-shrink: 0;
    }

    .knob-right {
        position: relative;
        display: flex;
        flex-direction: column;
        align-items: flex-end;
        min-width: 56px;
    }

    .knob-value {
        font-family: "JetBrains Mono", monospace;
        font-size: 12px;
        color: var(--text-secondary);
        cursor: ew-resize;
        text-align: right;
        width: 100%;
        border: 1px solid transparent;
        padding: 1px 4px;
    }

    .dragging .knob-value {
        color: var(--accent);
    }

    .knob-fill {
        height: 2px;
        background: var(--accent);
        opacity: 0.3;
        border-radius: 1px;
        margin-top: 2px;
        align-self: flex-start;
        transition: width 60ms ease-out;
    }

    .ctx-menu {
        position: fixed;
        z-index: 100;
        background: var(--surface-2-solid, #1f1e1d);
        border: 1px solid var(--border);
        border-radius: 4px;
        padding: 2px;
        min-width: 80px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
    }

    .ctx-item {
        display: block;
        width: 100%;
        padding: 4px 10px;
        font-size: 11px;
        font-family: "JetBrains Mono", monospace;
        color: var(--text-secondary);
        background: none;
        border: none;
        border-radius: 3px;
        cursor: pointer;
        text-align: left;
    }

    .ctx-item:hover {
        background: var(--surface-3);
        color: var(--text);
    }

    .knob-input {
        font-family: "JetBrains Mono", monospace;
        font-size: 12px;
        color: var(--accent);
        background: var(--surface-2);
        border: 1px solid var(--accent);
        border-radius: 3px;
        padding: 1px 4px;
        width: 100%;
        text-align: right;
        outline: none;
    }
</style>
