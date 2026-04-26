<script lang="ts">
    import Knob from "./Knob.svelte";
    import SegmentedToggle from "./SegmentedToggle.svelte";
    import { sample, getSample, whenLoaded } from "@dylanebert/shallot";
    import { sampleKnobs, loopOptions } from "./presets";
    import type { SampleBuffer } from "./App.svelte";

    let {
        slotId,
        values = $bindable(),
        buffers = $bindable(),
        onchange,
    }: {
        slotId: string;
        values: Record<string, number>;
        buffers: SampleBuffer[];
        onchange: (key: string, value: number) => void;
    } = $props();

    $effect.pre(() => {
        for (const knob of sampleKnobs) {
            const key = `${slotId}.${knob.field}`;
            if (values[key] === undefined) values[key] = knob.default;
        }
        if (values[`${slotId}.loop`] === undefined) values[`${slotId}.loop`] = 0;
        if (values[`${slotId}.bufferId`] === undefined && buffers.length > 0) {
            values[`${slotId}.bufferId`] = buffers[0].id;
            onchange(`${slotId}.bufferId`, buffers[0].id);
        }
    });

    let bufferId = $derived(values[`${slotId}.bufferId`]);
    let loopMode = $derived(values[`${slotId}.loop`] ?? 0);
    let selected = $derived(buffers.find((b) => b.id === bufferId) ?? null);

    let fileInput: HTMLInputElement;
    let dropActive = $state(false);
    let uploading = $state(false);
    let uploadError = $state<string | null>(null);
    let pickerOpen = $state(false);

    async function ingest(file: File) {
        uploadError = null;
        uploading = true;
        try {
            const id = sample(file, file.name);
            await whenLoaded(id);
            const decoded = getSample(id);
            if (!decoded?.data) throw new Error("decode failed");
            const exists = buffers.findIndex((b) => b.id === id);
            const entry = { id, name: file.name, data: decoded.data, sampleRate: decoded.sampleRate };
            if (exists >= 0) buffers[exists] = entry;
            else buffers = [...buffers, entry];
            values[`${slotId}.bufferId`] = id;
            onchange(`${slotId}.bufferId`, id);
        } catch (e) {
            uploadError = e instanceof Error ? e.message : "decode failed";
        } finally {
            uploading = false;
        }
    }

    async function onFileChange(e: Event) {
        const input = e.target as HTMLInputElement;
        const file = input.files?.[0];
        if (!file) return;
        await ingest(file);
        input.value = "";
    }

    async function onDrop(e: DragEvent) {
        e.preventDefault();
        dropActive = false;
        const file = e.dataTransfer?.files?.[0];
        if (file) await ingest(file);
    }

    function onDragOver(e: DragEvent) {
        e.preventDefault();
        dropActive = true;
    }

    function onDragLeave(e: DragEvent) {
        e.preventDefault();
        dropActive = false;
    }

    function setLoop(idx: number) {
        values[`${slotId}.loop`] = idx;
        onchange(`${slotId}.loop`, idx);
    }

    function selectBuffer(id: number) {
        values[`${slotId}.bufferId`] = id;
        onchange(`${slotId}.bufferId`, id);
        pickerOpen = false;
    }

    function fmtDuration(b: SampleBuffer): string {
        const s = b.data.length / b.sampleRate;
        return s < 1 ? `${(s * 1000).toFixed(0)}ms` : `${s.toFixed(2)}s`;
    }

    const PEAKS = 96;
    function buildPeaks(data: Float32Array): number[] {
        const step = Math.max(1, Math.floor(data.length / PEAKS));
        const out: number[] = [];
        for (let i = 0; i < PEAKS; i++) {
            let lo = 0;
            let hi = 0;
            const start = i * step;
            const end = Math.min(start + step, data.length);
            for (let j = start; j < end; j++) {
                const v = data[j];
                if (v < lo) lo = v;
                if (v > hi) hi = v;
            }
            out.push(Math.max(Math.abs(lo), Math.abs(hi)));
        }
        const max = Math.max(...out, 1e-6);
        return out.map((v) => v / max);
    }

    let peaks = $derived(selected ? buildPeaks(selected.data) : null);
</script>

<div class="editor">
    <div
        class="preview"
        class:drop-active={dropActive}
        class:empty={!selected}
        ondragover={onDragOver}
        ondragleave={onDragLeave}
        ondrop={onDrop}
        onclick={() => fileInput.click()}
        onkeydown={(e) => { if (e.key === "Enter") fileInput.click(); }}
        role="button"
        tabindex="0"
    >
        {#if selected && peaks}
            <svg class="wave" viewBox="0 0 {PEAKS} 100" preserveAspectRatio="none">
                {#each peaks as p, i}
                    {@const h = Math.max(2, p * 88)}
                    <rect
                        x={i + 0.15}
                        y={50 - h / 2}
                        width="0.7"
                        height={h}
                        fill="currentColor"
                    />
                {/each}
            </svg>
            <div class="preview-meta">
                <span class="preview-name">{selected.name}</span>
                <span class="preview-stats">{fmtDuration(selected)} · {(selected.sampleRate / 1000).toFixed(1)}kHz</span>
            </div>
        {:else}
            <div class="empty-content">
                {#if uploading}
                    <span class="empty-title">decoding…</span>
                {:else}
                    <span class="empty-title">{dropActive ? "release to load" : "drop a sample"}</span>
                    <span class="empty-sub">wav · mp3 · ogg · click or drop</span>
                {/if}
            </div>
        {/if}
    </div>

    {#if uploadError}
        <div class="error">{uploadError}</div>
    {/if}

    <input
        bind:this={fileInput}
        type="file"
        accept="audio/*"
        onchange={onFileChange}
        style="display: none"
    />

    <div class="buffer-row">
        <span class="row-label">Buffer</span>
        <div class="picker">
            <button class="picker-btn" onclick={() => pickerOpen = !pickerOpen} disabled={buffers.length === 0}>
                <span class="picker-name">{selected ? selected.name : "—"}</span>
                <svg class="picker-chev" width="8" height="5" viewBox="0 0 8 5"><path d="M0 0l4 5 4-5z" fill="currentColor"/></svg>
            </button>
            {#if pickerOpen}
                <div class="picker-menu">
                    {#each buffers as b}
                        <button
                            class="picker-option"
                            class:active={b.id === bufferId}
                            onclick={() => selectBuffer(b.id)}
                        >
                            <span class="opt-name">{b.name}</span>
                            <span class="opt-meta">{fmtDuration(b)}</span>
                        </button>
                    {/each}
                </div>
            {/if}
        </div>
        <button class="upload-btn" onclick={() => fileInput.click()} title="Upload sample">↑</button>
    </div>

    <SegmentedToggle options={loopOptions} selected={loopMode} onchange={setLoop} />

    {#each sampleKnobs as knob}
        {@const key = `${slotId}.${knob.field}`}
        <Knob
            label={knob.label}
            bind:value={values[key]}
            min={knob.min}
            max={knob.max}
            step={knob.step}
            scale={knob.scale}
            fmt={knob.fmt}
            parse={knob.parse}
            defaultValue={knob.default}
            onchange={() => onchange(key, values[key])}
        />
    {/each}
</div>

<style>
    .editor {
        display: flex;
        flex-direction: column;
        gap: 8px;
    }

    .preview {
        position: relative;
        height: 110px;
        border: 1px dashed var(--border);
        border-radius: 6px;
        background: var(--surface-1);
        cursor: pointer;
        overflow: hidden;
        transition: border-color 150ms var(--ease-out), background 150ms var(--ease-out);
        color: #c9a578;
    }

    .preview:hover { border-color: var(--text-muted); }
    .preview.drop-active { border-color: var(--accent); border-style: solid; background: var(--surface-2); }
    .preview.empty { display: flex; align-items: center; justify-content: center; }

    .wave {
        position: absolute;
        inset: 0;
        width: 100%;
        height: 100%;
    }

    .preview-meta {
        position: absolute;
        left: 8px;
        right: 8px;
        bottom: 6px;
        display: flex;
        justify-content: space-between;
        align-items: baseline;
        font-family: "JetBrains Mono", monospace;
        font-size: 10px;
        color: var(--text-secondary);
        background: linear-gradient(transparent, rgba(14, 13, 12, 0.85) 60%);
        padding: 18px 4px 2px;
        margin: -18px -4px -2px;
    }

    .preview-name {
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
        max-width: 60%;
    }

    .preview-stats {
        color: var(--text-muted);
        flex-shrink: 0;
    }

    .empty-content {
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 4px;
        font-family: "JetBrains Mono", monospace;
        pointer-events: none;
    }

    .empty-title {
        font-size: 12px;
        color: var(--text-secondary);
        letter-spacing: 0.02em;
    }

    .empty-sub {
        font-size: 10px;
        color: var(--text-muted);
        letter-spacing: 0.06em;
        text-transform: uppercase;
    }

    .error {
        font-family: "JetBrains Mono", monospace;
        font-size: 11px;
        color: #d47860;
        padding: 4px 8px;
        background: rgba(212, 120, 96, 0.08);
        border-radius: 4px;
    }

    .buffer-row {
        display: grid;
        grid-template-columns: 60px 1fr 28px;
        align-items: center;
        gap: 8px;
    }

    .row-label {
        font-size: 11px;
        color: var(--text-muted);
        font-family: "JetBrains Mono", monospace;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }

    .picker {
        position: relative;
    }

    .picker-btn {
        width: 100%;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 6px;
        padding: 6px 10px;
        font-family: "JetBrains Mono", monospace;
        font-size: 12px;
        color: var(--text-secondary);
        background: var(--surface-2);
        border: 1px solid var(--border);
        border-radius: 4px;
        cursor: pointer;
        transition: border-color 150ms var(--ease-out);
    }

    .picker-btn:hover:not(:disabled) {
        border-color: var(--text-muted);
    }

    .picker-btn:disabled {
        cursor: default;
        color: var(--text-muted);
        opacity: 0.5;
    }

    .picker-name {
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
        flex: 1;
        min-width: 0;
        text-align: left;
    }

    .picker-chev {
        flex-shrink: 0;
        color: var(--text-muted);
    }

    .picker-menu {
        position: absolute;
        top: calc(100% + 4px);
        left: 0;
        right: 0;
        background: var(--surface-2-solid);
        border: 1px solid var(--border);
        border-radius: 4px;
        padding: 2px;
        z-index: 10;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
        max-height: 200px;
        overflow-y: auto;
    }

    .picker-option {
        display: flex;
        justify-content: space-between;
        align-items: baseline;
        gap: 8px;
        width: 100%;
        padding: 5px 8px;
        font-family: "JetBrains Mono", monospace;
        font-size: 11px;
        color: var(--text-secondary);
        background: none;
        border: none;
        border-radius: 3px;
        cursor: pointer;
        text-align: left;
    }

    .picker-option:hover {
        background: var(--surface-3);
        color: var(--text);
    }

    .picker-option.active {
        background: var(--surface-3);
        color: var(--accent);
    }

    .opt-name {
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
        flex: 1;
        min-width: 0;
    }

    .opt-meta {
        color: var(--text-muted);
        font-size: 10px;
        flex-shrink: 0;
    }

    .upload-btn {
        width: 28px;
        height: 28px;
        background: var(--surface-2);
        border: 1px solid var(--border);
        border-radius: 4px;
        color: var(--text-muted);
        font-size: 12px;
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 0;
        line-height: 1;
        transition: color 150ms var(--ease-out), border-color 150ms var(--ease-out);
    }

    .upload-btn:hover {
        color: var(--accent);
        border-color: var(--accent);
    }

    .upload-btn:active {
        transform: scale(0.95);
    }
</style>
