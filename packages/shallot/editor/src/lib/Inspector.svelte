<script lang="ts">
    import type { Document, Node, Command } from "@dylanebert/shallot/editor";
    import { parseFields, formatFields, type Diagnostic, type State } from "@dylanebert/shallot";
    import { getComponent, getComponents, dependencies, schema, readFields, type Derived, type Traits } from "@dylanebert/shallot/ecs/core";
    import { getMeta, Plus, ChevronRight, TriangleAlert, groupComponents, type ComponentGroup } from "./components";
    import Icon from "./Icon.svelte";
    import { dismissOnClickOutside } from "./dismiss";

    type FieldEntry =
        | { type: "float"; key: string; label: string; value: number }
        | { type: "vec3"; base: string; label: string; x: number; y: number; z: number }
        | { type: "vec2"; base: string; label: string; x: number; y: number }
        | { type: "color"; key: string; label: string; value: number }
        | { type: "enum"; key: string; label: string; value: number; options: Record<string, number> }
        | { type: "other"; label: string; value: string };

    type ComponentSection = {
        name: string;
        fields: FieldEntry[];
        parsed: Record<string, number | string> | null;
        registered: boolean;
        diagnosticMessage?: string;
    };

    let { doc, version, diagnostics, ecs, nodeMap, onchange, onsync, onremove, onadd, onreorder }: {
        doc: Document;
        version: number;
        diagnostics: Diagnostic[];
        ecs: State | null;
        nodeMap: Map<Node, number>;
        onchange: () => void;
        onsync: (node: Node, attrName: string, fields: Record<string, number | string>) => void;
        onremove: (node: Node, attrName: string) => void;
        onadd: (node: Node, attrName: string) => void;
        onreorder: () => void;
    } = $props();

    let contextMenu: { x: number; y: number; node: Node; name: string } | null = $state.raw(null);

    function removeComponent(node: Node, name: string) {
        doc.removeAttr(node, name);
        onremove(node, name);
        onchange();
    }

    function showContextMenu(e: MouseEvent, node: Node, name: string) {
        contextMenu = { x: e.clientX, y: e.clientY, node, name };
    }

    function handleContextRemove() {
        if (!contextMenu || !selected) return;
        const { name } = contextMenu;
        contextMenu = null;
        removeComponent(selected, name);
    }

    $effect(() => {
        if (!contextMenu) return;
        return dismissOnClickOutside(() => { contextMenu = null; }, ".context-menu");
    });

    let addOpen = $state(false);
    let addFilter = $state("");
    let addBtnEl = $state<HTMLButtonElement>();
    let pickerStyle = $state("");

    let availableComponents = $derived.by(() => {
        void version;
        if (!selected) return [];
        const existing = new Set(selected.attrs.map((a) => a.name));
        const eid = ecs ? nodeMap?.get(selected) : undefined;
        return getComponents()
            .map((r) => r.name)
            .filter((n) => {
                if (existing.has(n)) return false;
                if (eid !== undefined) {
                    const reg = getComponent(n);
                    if (reg && ecs!.hasComponent(eid, reg.component as never)) return false;
                }
                return true;
            });
    });

    let matches = $derived(
        addFilter
            ? availableComponents.filter((n) => n.includes(addFilter.toLowerCase()))
            : availableComponents
    );

    let groups = $derived.by((): ComponentGroup[] => {
        return groupComponents(matches);
    });

    let focusIdx = $state(-1);

    function handlePickerKey(e: KeyboardEvent) {
        const total = matches.length;
        if (!total) return;
        if (e.key === "ArrowDown") {
            e.preventDefault();
            focusIdx = focusIdx < total - 1 ? focusIdx + 1 : 0;
        } else if (e.key === "ArrowUp") {
            e.preventDefault();
            focusIdx = focusIdx > 0 ? focusIdx - 1 : total - 1;
        } else if (e.key === "Enter" && focusIdx >= 0 && focusIdx < total) {
            e.preventDefault();
            addComponent(matches[focusIdx]);
        }
    }

    $effect(() => {
        void addFilter;
        focusIdx = -1;
    });

    function openAddComponent() {
        const rect = addBtnEl!.getBoundingClientRect();
        const spaceAbove = rect.top;
        const spaceBelow = window.innerHeight - rect.bottom;
        const lr = `position:fixed;left:${rect.left}px;right:${window.innerWidth - rect.right}px`;
        pickerStyle = spaceAbove > spaceBelow
            ? `${lr};bottom:${window.innerHeight - rect.top + 4}px;--slide:4px`
            : `${lr};top:${rect.bottom + 4}px;--slide:-4px`;
        addOpen = true;
        addFilter = "";
        focusIdx = -1;
    }

    function addComponent(name: string) {
        if (!selected) return;
        const reg = getComponent(name);
        const defaults = reg?.traits?.defaults?.() ?? {};
        const value = formatFields(name, defaults);

        const existing = new Set(selected.attrs.map((a) => a.name));
        const depCommands: Command[] = [];
        for (const dep of dependencies(name)) {
            if (!existing.has(dep)) {
                const depReg = getComponent(dep);
                const depDefaults = depReg?.traits?.defaults?.() ?? {};
                depCommands.push({ type: "addAttr", node: selected, name: dep, value: formatFields(dep, depDefaults) });
            }
        }

        if (depCommands.length === 0) {
            doc.addAttr(selected, name, value);
        } else {
            doc.compound([
                { type: "addAttr", node: selected, name, value },
                ...depCommands,
            ]);
        }

        onadd(selected, name);
        for (const cmd of depCommands) {
            if (cmd.type === "addAttr") onadd(selected, cmd.name);
        }

        onchange();
        collapseState.set(name, true);
        collapseState = new Map(collapseState);
        addOpen = false;
    }

    function focus(node: HTMLElement) {
        node.focus();
    }

    $effect(() => {
        if (!addOpen) return;
        return dismissOnClickOutside(() => { addOpen = false; }, ".add-component-picker", ".add-component-btn");
    });

    let collapseState: Map<string, boolean> = $state.raw(new Map());

    function isExpanded(name: string): boolean {
        return collapseState.get(name) ?? false;
    }

    let animate = $state(false);

    function toggleSection(name: string) {
        animate = true;
        const current = isExpanded(name);
        collapseState.set(name, !current);
        collapseState = new Map(collapseState);
    }

    let selected = $derived(
        version >= 0 && doc.selection.size === 1 ? doc.selection.values().next().value! : null
    );

    $effect(() => {
        void selected;
        animate = false;
    });

    let sections = $derived.by(() => {
        void version;
        if (!selected) return [];
        const eid = ecs ? nodeMap?.get(selected) : undefined;

        const result: ComponentSection[] = [];
        const seen = new Set<string>();
        for (const attr of selected.attrs) {
            seen.add(attr.name);
            result.push(eid !== undefined ? analyzeFromECS(attr.name, eid) : analyzeAttr(attr));
        }

        if (eid !== undefined) {
            for (const { name, component } of getComponents()) {
                if (seen.has(name)) continue;
                if (!ecs!.hasComponent(eid, component as never)) continue;
                result.push(analyzeFromECS(name, eid));
            }
        }
        return result;
    });

    function toKebab(s: string): string {
        return s.replace(/([a-z])([A-Z])/g, "$1-$2").toLowerCase();
    }

    function round(v: number): number {
        return Math.round(v * 1000) / 1000;
    }

    function isCustomField(key: string, traits: { format?: Record<string, unknown>; parse?: Record<string, unknown> } | undefined): boolean {
        return !!(traits?.format?.[key] || traits?.parse?.[key]);
    }

    function getDerived(t: Traits | undefined): Record<string, Derived> | undefined {
        return t?.annotations?.derived as Record<string, Derived> | undefined;
    }

    function buildFields(name: string, parsed: Record<string, number | string>): FieldEntry[] {
        const reg = getComponent(name);
        if (!reg) return [];
        const s = schema(name);
        if (!s) return [];
        const traits = reg.traits;
        const fields: FieldEntry[] = [];

        for (const info of s.fields) {
            switch (info.kind) {
                case "vec4": {
                    const [xK, yK, zK, wK] = info.fields!;
                    fields.push({ type: "other", label: toKebab(info.name), value: `${round(parsed[xK] as number)} ${round(parsed[yK] as number)} ${round(parsed[zK] as number)} ${round(parsed[wK] as number)}` });
                    break;
                }
                case "vec3": {
                    const [xK, yK, zK] = info.fields!;
                    if (isCustomField(xK, traits)) {
                        fields.push({ type: "other", label: toKebab(info.name), value: `${parsed[xK]} ${parsed[yK]} ${parsed[zK]}` });
                    } else {
                        fields.push({ type: "vec3", base: info.name, label: toKebab(info.name), x: round(parsed[xK] as number), y: round(parsed[yK] as number), z: round(parsed[zK] as number) });
                    }
                    break;
                }
                case "vec2": {
                    const [xK, yK] = info.fields!;
                    if (isCustomField(xK, traits)) {
                        fields.push({ type: "other", label: toKebab(info.name), value: `${parsed[xK]} ${parsed[yK]}` });
                    } else {
                        fields.push({ type: "vec2", base: info.name, label: toKebab(info.name), x: round(parsed[xK] as number), y: round(parsed[yK] as number) });
                    }
                    break;
                }
                case "enum": {
                    const val = parsed[info.name];
                    if (typeof val === "number" && info.options) {
                        fields.push({ type: "enum", key: info.name, label: toKebab(info.name), value: val, options: info.options });
                    }
                    break;
                }
                case "color": {
                    const val = parsed[info.name];
                    if (typeof val === "number") {
                        fields.push({ type: "color", key: info.name, label: toKebab(info.name), value: val });
                    }
                    break;
                }
                case "float": {
                    const val = parsed[info.name];
                    if (typeof val === "number") {
                        fields.push({ type: "float", key: info.name, label: toKebab(info.name), value: round(val) });
                    } else {
                        const formatFn = traits?.format?.[info.name] as ((v: number) => string) | undefined;
                        const display = formatFn && typeof val === "number" ? formatFn(val) ?? String(val) : String(val);
                        fields.push({ type: "other", label: toKebab(info.name), value: display });
                    }
                    break;
                }
                case "string": {
                    const val = parsed[info.name];
                    fields.push({ type: "other", label: toKebab(info.name), value: String(val ?? "") });
                    break;
                }
            }
        }

        for (const virt of s.derived) {
            const p = parsed as Record<string, number>;
            if (virt.kind === "vec3" && virt.fields) {
                const [xK, yK, zK] = virt.fields;
                fields.push({
                    type: "vec3",
                    base: "derived:" + virt.name,
                    label: toKebab(virt.name),
                    x: round(getDerived(traits)![xK].get(p)),
                    y: round(getDerived(traits)![yK].get(p)),
                    z: round(getDerived(traits)![zK].get(p)),
                });
            } else {
                fields.push({
                    type: "float",
                    key: "derived:" + virt.name,
                    label: toKebab(virt.name),
                    value: round(getDerived(traits)![virt.name].get(p)),
                });
            }
        }

        return fields;
    }

    function analyzeAttr(attr: { name: string; value: string }): ComponentSection {
        const reg = getComponent(attr.name);
        if (!reg) {
            const diag = selected ? diagnostics.find((d) => d.node === selected && d.attr === attr.name) : undefined;
            return { name: attr.name, fields: [], parsed: null, registered: false, diagnosticMessage: diag?.message };
        }

        let parsed: Record<string, number | string>;
        try {
            const defaults = reg.traits?.defaults?.() ?? {};
            if (attr.value) {
                parsed = { ...defaults, ...parseFields(attr.name, attr.value) };
            } else {
                parsed = defaults;
            }
        } catch {
            return { name: attr.name, fields: [], parsed: null, registered: true };
        }

        return { name: attr.name, fields: buildFields(attr.name, parsed), parsed, registered: true };
    }

    function analyzeFromECS(name: string, eid: number): ComponentSection {
        const reg = getComponent(name);
        if (!reg) return { name, fields: [], parsed: null, registered: false };

        const defaults = reg.traits?.defaults?.() ?? {};
        const fields = readFields(reg.component, eid);
        const parsed = { ...defaults, ...fields };

        return { name, fields: buildFields(name, parsed), parsed, registered: true };
    }

    function ensureAttr(node: Node, attrName: string, parsed: Record<string, number | string>) {
        if (!node.attrs.find((a) => a.name === attrName)) {
            doc.addAttr(node, attrName, formatFields(attrName, parsed));
            onadd(node, attrName);
        }
    }

    function updateField(node: Node, attrName: string, parsed: Record<string, number | string>, fieldKey: string, value: number) {
        if (fieldKey.startsWith("derived:")) {
            const realKey = fieldKey.slice(8);
            const reg = getComponent(attrName);
            const virt = getDerived(reg?.traits)?.[realKey];
            if (!virt) return;
            const realUpdates = virt.set(value, parsed as Record<string, number>);
            const updated = { ...parsed, ...realUpdates };
            const str = formatFields(attrName, updated);
            ensureAttr(node, attrName, parsed);
            doc.setAttr(node, attrName, str);
            onchange();
            onsync(node, attrName, { [realKey]: value });
            return;
        }
        const updated = { ...parsed, [fieldKey]: value };
        const str = formatFields(attrName, updated);
        ensureAttr(node, attrName, parsed);
        doc.setAttr(node, attrName, str);
        onchange();
        onsync(node, attrName, { [fieldKey]: value });
    }

    function selectOnFocus(e: FocusEvent) {
        (e.target as HTMLInputElement).select();
    }

    function blurOnEnter(e: KeyboardEvent) {
        if (e.key === "Enter") (e.target as HTMLInputElement).blur();
    }

    let dragSectionIdx: number | null = $state(null);
    let dropSectionIdx: number | null = $state(null);
    let sectionDragging = $state(false);
    let sectionDragStartY = 0;
    let inspectorEl: HTMLElement;

    let dragPointerId: number | null = null;

    function handleSectionDragStart(e: PointerEvent, idx: number) {
        if (e.button !== 0) return;
        dragSectionIdx = idx;
        dragPointerId = e.pointerId;
        sectionDragStartY = e.clientY;
        sectionDragging = false;
    }

    function handleSectionDragMove(e: PointerEvent) {
        if (dragSectionIdx === null) return;
        if (!sectionDragging) {
            if (Math.abs(e.clientY - sectionDragStartY) < 4) return;
            sectionDragging = true;
            if (dragPointerId !== null) inspectorEl.setPointerCapture(dragPointerId);
        }
        const headers = Array.from(inspectorEl.querySelectorAll(".section-header")) as HTMLElement[];
        let target: number | null = null;
        for (let i = 0; i < headers.length; i++) {
            const rect = headers[i].getBoundingClientRect();
            const mid = rect.top + rect.height / 2;
            if (e.clientY < mid) { target = i; break; }
        }
        if (target === null) target = headers.length;
        if (target === dragSectionIdx || target === dragSectionIdx + 1) {
            dropSectionIdx = null;
        } else {
            dropSectionIdx = target;
        }
    }

    function handleSectionDragEnd() {
        if (dragSectionIdx === null) return;
        if (sectionDragging && dropSectionIdx !== null && selected) {
            let to = dropSectionIdx;
            if (to > dragSectionIdx) to--;
            doc.reorderAttr(selected, dragSectionIdx, to);
            onreorder();
        }
        dragSectionIdx = null;
        dragPointerId = null;
        dropSectionIdx = null;
        sectionDragging = false;
    }

    function dragLabel(e: PointerEvent, node: Node, attrName: string, fieldKey: string) {
        const reg = getComponent(attrName);
        const defaults = reg?.traits?.defaults?.() ?? {};
        const attr = node.attrs.find((a) => a.name === attrName);
        let startValue: string;
        let fresh: Record<string, number | string>;
        if (attr) {
            startValue = attr.value;
            fresh = startValue ? { ...defaults, ...parseFields(attrName, startValue) } : defaults;
        } else {
            const eid = nodeMap?.get(node);
            if (eid === undefined || !reg) return;
            fresh = { ...defaults, ...readFields(reg.component, eid) };
            ensureAttr(node, attrName, fresh);
            startValue = node.attrs.find((a) => a.name === attrName)!.value;
        }
        const label = e.currentTarget as HTMLElement;
        const input = (label.closest(".vec-field") ?? label.closest(".field-row"))?.querySelector("input") as HTMLInputElement | null;
        label.setPointerCapture(e.pointerId);
        const isDerived = fieldKey.startsWith("derived:");
        const realKey = isDerived ? fieldKey.slice(8) : fieldKey;
        const virt = isDerived ? getDerived(reg?.traits)?.[realKey] : undefined;
        let acc = virt ? virt.get(fresh as Record<string, number>) : ((fresh[fieldKey] as number) ?? 0);
        const live = { ...fresh };
        function onMove(ev: PointerEvent) {
            acc += ev.movementX * 0.01;
            const rounded = Math.round(acc * 1000) / 1000;
            if (virt) {
                const realUpdates = virt.set(rounded, live as Record<string, number>);
                Object.assign(live, realUpdates);
            } else {
                live[fieldKey] = rounded;
            }
            if (input) input.value = String(rounded);
            onsync(node, attrName, { [realKey]: rounded });
        }
        function onUp() {
            label.removeEventListener("pointermove", onMove);
            label.removeEventListener("pointerup", onUp);
            const finalValue = formatFields(attrName, live);
            if (finalValue !== startValue) {
                doc.setAttr(node, attrName, finalValue, startValue);
                onchange();
            }
        }
        label.addEventListener("pointermove", onMove);
        label.addEventListener("pointerup", onUp);
    }

    function hexToCss(packed: number): string {
        return "#" + ((packed >>> 0) & 0xffffff).toString(16).padStart(6, "0");
    }

    function hexToRgb(packed: number): [number, number, number] {
        const n = packed >>> 0;
        return [(n >> 16) & 0xff, (n >> 8) & 0xff, n & 0xff];
    }

    function rgbToHex(r: number, g: number, b: number): number {
        return ((r & 0xff) << 16) | ((g & 0xff) << 8) | (b & 0xff);
    }

    function rgbToHsv(r: number, g: number, b: number): [number, number, number] {
        r /= 255; g /= 255; b /= 255;
        const max = Math.max(r, g, b), min = Math.min(r, g, b);
        const d = max - min;
        const s = max === 0 ? 0 : d / max;
        let h = 0;
        if (d !== 0) {
            if (max === r) h = ((g - b) / d + 6) % 6;
            else if (max === g) h = (b - r) / d + 2;
            else h = (r - g) / d + 4;
            h /= 6;
        }
        return [h, s, max];
    }

    function hsvToRgb(h: number, s: number, v: number): [number, number, number] {
        const i = Math.floor(h * 6);
        const f = h * 6 - i;
        const p = v * (1 - s), q = v * (1 - f * s), t = v * (1 - (1 - f) * s);
        let r: number, g: number, b: number;
        switch (i % 6) {
            case 0: r = v; g = t; b = p; break;
            case 1: r = q; g = v; b = p; break;
            case 2: r = p; g = v; b = t; break;
            case 3: r = p; g = q; b = v; break;
            case 4: r = t; g = p; b = v; break;
            default: r = v; g = p; b = q;
        }
        return [Math.round(r * 255), Math.round(g * 255), Math.round(b * 255)];
    }

    function hexToHsv(packed: number): [number, number, number] {
        return rgbToHsv(...hexToRgb(packed));
    }

    function hsvToHex(h: number, s: number, v: number): number {
        return rgbToHex(...hsvToRgb(h, s, v));
    }

    let pickerCtx: {
        node: Node;
        startValue: string;
        live: Record<string, number | string>;
        dragging: boolean;
    } | null = null;

    let colorPicker: { key: string; attrName: string; h: number; s: number; v: number; x: number; y: number } | null = $state(null);

    function syncColorLive() {
        const ctx = pickerCtx;
        const cp = colorPicker;
        if (!cp || !ctx) return;
        const hex = hsvToHex(cp.h, cp.s, cp.v);
        ctx.live[cp.key] = hex;
        onsync(ctx.node, cp.attrName, { [cp.key]: hex });
    }

    function openColorPicker(e: MouseEvent, node: Node, attrName: string, parsed: Record<string, number | string>, key: string, value: number) {
        if (colorPicker) commitColorGesture();
        const rect = (e.currentTarget as HTMLElement).getBoundingClientRect();
        const [h, s, v] = hexToHsv(value);
        ensureAttr(node, attrName, parsed);
        const attr = node.attrs.find((a) => a.name === attrName);
        pickerCtx = { node, startValue: attr?.value ?? "", live: { ...parsed }, dragging: false };
        colorPicker = { key, attrName, h, s, v, x: rect.left, y: rect.bottom + 4 };
    }

    function commitColorGesture() {
        const ctx = pickerCtx;
        const cp = colorPicker;
        if (!ctx || !cp) return;
        ctx.dragging = false;
        const finalValue = formatFields(cp.attrName, ctx.live);
        if (finalValue !== ctx.startValue) {
            doc.setAttr(ctx.node, cp.attrName, finalValue, ctx.startValue);
            onchange();
            ctx.startValue = finalValue;
            ctx.live = { ...ctx.live };
        }
    }

    function closeColorPicker() {
        commitColorGesture();
        pickerCtx = null;
        colorPicker = null;
    }

    function handleSvPointer(e: PointerEvent) {
        if (!colorPicker) return;
        const rect = (e.currentTarget as HTMLElement).getBoundingClientRect();
        colorPicker.s = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
        colorPicker.v = Math.max(0, Math.min(1, 1 - (e.clientY - rect.top) / rect.height));
        syncColorLive();
    }

    function handleSvDown(e: PointerEvent) {
        (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
        if (pickerCtx) pickerCtx.dragging = true;
        handleSvPointer(e);
    }

    function handleHuePointer(e: PointerEvent) {
        if (!colorPicker) return;
        const rect = (e.currentTarget as HTMLElement).getBoundingClientRect();
        colorPicker.h = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
        syncColorLive();
    }

    function handleHueDown(e: PointerEvent) {
        (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
        if (pickerCtx) pickerCtx.dragging = true;
        handleHuePointer(e);
    }

    $effect(() => {
        void version;
        const ctx = pickerCtx;
        const cp = colorPicker;
        if (!ctx || !cp) return;
        if (ctx.dragging) return;
        const attr = ctx.node.attrs.find((a) => a.name === cp.attrName);
        if (!attr) return;
        if (attr.value !== ctx.startValue) {
            ctx.startValue = attr.value;
            const reg = getComponent(cp.attrName);
            const defaults = reg?.traits?.defaults?.() ?? {};
            const parsed = attr.value ? { ...defaults, ...parseFields(cp.attrName, attr.value) } : defaults;
            ctx.live = parsed;
            const hex = parsed[cp.key];
            if (typeof hex === "number") {
                const [h, s, v] = hexToHsv(hex);
                cp.h = h;
                cp.s = s;
                cp.v = v;
            }
        }
    });

    $effect(() => {
        if (!colorPicker) return;
        return dismissOnClickOutside(closeColorPicker, ".color-picker-popover", ".color-swatch");
    });
</script>

<div
    class="inspector"
    class:animate
    class:section-dragging={sectionDragging}
    role="group"
    ondragstart={(e) => e.preventDefault()}
    bind:this={inspectorEl}
    onpointermove={handleSectionDragMove}
    onpointerup={handleSectionDragEnd}
>
    {#if selected}
        <div class="section id-section">
            <div class="node-id">{version >= 0 ? (selected.id ?? "entity") : ""}</div>
        </div>
        {#each sections as section, sIdx}
            {@const meta = getMeta(section.name)}
            {@const expanded = isExpanded(section.name)}
            {#if sectionDragging && dropSectionIdx === sIdx}
                <div class="section-drop-indicator"></div>
            {/if}
            {#if !section.registered}
                <div class="section unregistered" class:drag-source-section={sectionDragging && dragSectionIdx === sIdx} style="--section-color: var(--text-muted)">
                    <div
                        class="section-header"
                        role="group"
                        onpointerdown={(e) => handleSectionDragStart(e, sIdx)}
                        oncontextmenu={(e) => { e.preventDefault(); selected && showContextMenu(e, selected, section.name); }}
                    >
                        <Icon icon={TriangleAlert} size={13} strokeWidth={1.5} class="section-icon" />
                        <span class="section-label">{section.name}</span>
                    </div>
                    <div class="missing-hint">{section.diagnosticMessage ?? "Plugin not loaded"}</div>
                </div>
            {:else}
            {@const hasFields = section.fields.length > 0}
            <div class="section" class:drag-source-section={sectionDragging && dragSectionIdx === sIdx} style="--section-color: {meta.color}">
                <button
                    class="section-header"
                    class:flag={!hasFields}
                    onclick={() => hasFields && toggleSection(section.name)}
                    onpointerdown={(e) => handleSectionDragStart(e, sIdx)}
                    oncontextmenu={(e) => { e.preventDefault(); selected && showContextMenu(e, selected, section.name); }}
                >
                    <Icon icon={meta.icon} size={14} strokeWidth={1.5} class="section-icon" />
                    <span class="section-label">{section.name}</span>
                    {#if hasFields}
                        <Icon icon={ChevronRight} size={12} strokeWidth={2} class="section-chevron {expanded ? 'expanded' : ''}" />
                    {/if}
                </button>
                <div class="section-body" class:expanded={hasFields && expanded}>
                    <div class="section-body-inner">
                        <div class="section-fields">
                        {#if section.fields.length > 0}
                            {#each section.fields as field}
                                {#if field.type === "float"}
                                    <div class="field-row">
                                        <span
                                            role="slider"
                                            tabindex="-1"
                                            aria-valuenow={field.value}
                                            class="field-label drag-label"
                                            onpointerdown={(e) => selected && dragLabel(e, selected, section.name, field.key)}
                                        >{field.label}</span>
                                        <input
                                            class="field-input"
                                            type="number"
                                            step="0.1"
                                            value={field.value}
                                            onfocus={selectOnFocus}
                                            onkeydown={blurOnEnter}
                                            onchange={(e) => {
                                                const v = parseFloat((e.target as HTMLInputElement).value);
                                                if (!Number.isNaN(v) && selected && section.parsed) {
                                                    updateField(selected, section.name, section.parsed, field.key, v);
                                                }
                                            }}
                                        />
                                    </div>
                                {:else if field.type === "vec3"}
                                    <div class="field-row">
                                        <span class="field-label">{field.label}</span>
                                        <div class="vec-inputs">
                                            {#each [["X", field.x], ["Y", field.y], ["Z", field.z]] as [suffix, val]}
                                                <div class="vec-field">
                                                    <span
                                                        role="slider"
                                                        tabindex="-1"
                                                        aria-valuenow={Number(val)}
                                                        class="vec-label drag-label"
                                                        onpointerdown={(e) => selected && dragLabel(e, selected, section.name, field.base + suffix)}
                                                    >{suffix}</span>
                                                    <input
                                                        class="field-input vec-input"
                                                        type="number"
                                                        step="0.1"
                                                        value={val}
                                                        onfocus={selectOnFocus}
                                                        onkeydown={blurOnEnter}
                                                        onchange={(e) => {
                                                            const v = parseFloat((e.target as HTMLInputElement).value);
                                                            if (!Number.isNaN(v) && selected && section.parsed) {
                                                                updateField(selected, section.name, section.parsed, field.base + suffix, v);
                                                            }
                                                        }}
                                                    />
                                                </div>
                                            {/each}
                                        </div>
                                    </div>
                                {:else if field.type === "vec2"}
                                    <div class="field-row">
                                        <span class="field-label">{field.label}</span>
                                        <div class="vec-inputs">
                                            {#each [["X", field.x], ["Y", field.y]] as [suffix, val]}
                                                <div class="vec-field">
                                                    <span
                                                        role="slider"
                                                        tabindex="-1"
                                                        aria-valuenow={Number(val)}
                                                        class="vec-label drag-label"
                                                        onpointerdown={(e) => selected && dragLabel(e, selected, section.name, field.base + suffix)}
                                                    >{suffix}</span>
                                                    <input
                                                        class="field-input vec-input"
                                                        type="number"
                                                        step="0.1"
                                                        value={val}
                                                        onfocus={selectOnFocus}
                                                        onkeydown={blurOnEnter}
                                                        onchange={(e) => {
                                                            const v = parseFloat((e.target as HTMLInputElement).value);
                                                            if (!Number.isNaN(v) && selected && section.parsed) {
                                                                updateField(selected, section.name, section.parsed, field.base + suffix, v);
                                                            }
                                                        }}
                                                    />
                                                </div>
                                            {/each}
                                        </div>
                                    </div>
                                {:else if field.type === "enum"}
                                    <div class="field-row">
                                        <span class="field-label">{field.label}</span>
                                        <select
                                            class="field-input field-select"
                                            value={String(field.value)}
                                            onchange={(e) => {
                                                const v = parseInt((e.target as HTMLSelectElement).value);
                                                if (!Number.isNaN(v) && selected && section.parsed) {
                                                    updateField(selected, section.name, section.parsed, field.key, v);
                                                }
                                            }}
                                        >
                                            {#each Object.entries(field.options) as [name, val]}
                                                <option value={String(val)} selected={val === field.value}>{toKebab(name)}</option>
                                            {/each}
                                        </select>
                                    </div>
                                {:else if field.type === "color"}
                                    {@const liveHex = colorPicker && colorPicker.key === field.key && colorPicker.attrName === section.name ? hsvToHex(colorPicker.h, colorPicker.s, colorPicker.v) : field.value}
                                    <div class="field-row">
                                        <span class="field-label">{field.label}</span>
                                        <button
                                            class="color-swatch"
                                            style="background: {hexToCss(liveHex)}"
                                            title={field.label}
                                            onclick={(e) => selected && section.parsed && openColorPicker(e, selected, section.name, section.parsed, field.key, field.value)}
                                        ></button>
                                        <span class="field-value">{hexToCss(liveHex)}</span>
                                    </div>
                                {:else}
                                    <div class="field-row">
                                        <span class="field-label">{field.label}</span>
                                        <span class="field-value">{field.value}</span>
                                    </div>
                                {/if}
                            {/each}
                        {/if}
                        </div>
                    </div>
                </div>
            </div>
            {/if}
        {/each}
        {#if sectionDragging && dropSectionIdx === sections.length}
            <div class="section-drop-indicator"></div>
        {/if}
        {#if availableComponents.length > 0}
            <div class="add-component">
                <button class="add-component-btn" bind:this={addBtnEl} onclick={openAddComponent}>
                    <Icon icon={Plus} size={14} strokeWidth={1.5} />
                    <span>Add Component</span>
                </button>
                {#if addOpen}
                    <!-- svelte-ignore a11y_no_static_element_interactions -->
                    <div class="add-component-picker" style={pickerStyle} onkeydown={handlePickerKey}>
                        <input
                            class="add-component-search"
                            type="text"
                            placeholder="Search..."
                            bind:value={addFilter}
                            use:focus
                        />
                        <div class="add-component-list">
                            {#if groups.length > 0}
                                {#each groups as group, gIdx}
                                    <div class="add-group" style="--group-delay: {gIdx * 30}ms">
                                        <div class="add-group-header" onmouseenter={() => { focusIdx = -1; }}>
                                            <span class="add-group-dot" style="background: {group.category.color}"></span>
                                            <span class="add-group-label">{group.category.label}</span>
                                        </div>
                                        {#each group.items as name}
                                            {@const meta = getMeta(name)}
                                            {@const flatIdx = matches.indexOf(name)}
                                            <button
                                                class="add-component-item"
                                                class:focused={focusIdx === flatIdx}
                                                onmousedown={() => addComponent(name)}
                                                onmouseenter={() => { focusIdx = flatIdx; }}
                                            >
                                                <Icon icon={meta.icon} size={13} strokeWidth={1.5} class="add-component-icon" style="color: {meta.color}" />
                                                <span>{name}</span>
                                            </button>
                                        {/each}
                                    </div>
                                {/each}
                            {:else}
                                <div class="add-component-empty">No components</div>
                            {/if}
                        </div>
                    </div>
                {/if}
            </div>
        {/if}
    {:else if version >= 0 && doc.selection.size > 1}
        <div class="hint">{doc.selection.size} nodes selected</div>
    {:else}
        <div class="hint">No selection</div>
    {/if}


</div>

{#if contextMenu}
    <!-- svelte-ignore a11y_no_static_element_interactions -->
    <div
        class="context-menu"
        style="left: {contextMenu.x}px; top: {contextMenu.y}px"
    >
        <button class="context-item" onmousedown={handleContextRemove}>
            Remove
        </button>
    </div>
{/if}

{#if colorPicker}
    {@const hueColor = `hsl(${colorPicker.h * 360}, 100%, 50%)`}
    {@const currentCss = hexToCss(hsvToHex(colorPicker.h, colorPicker.s, colorPicker.v))}
    <!-- svelte-ignore a11y_no_static_element_interactions -->
    <div
        class="color-picker-popover"
        style="left: {colorPicker.x}px; top: {colorPicker.y}px"
    >
        <div
            class="cp-sv"
            style="background: {hueColor}"
            onpointerdown={handleSvDown}
            onpointermove={(e) => (e.currentTarget as HTMLElement).hasPointerCapture(e.pointerId) && handleSvPointer(e)}
            onpointerup={commitColorGesture}
        >
            <div class="cp-sv-white"></div>
            <div class="cp-sv-black"></div>
            <div
                class="cp-sv-cursor"
                style="left: {colorPicker.s * 100}%; top: {(1 - colorPicker.v) * 100}%"
            ></div>
        </div>
        <div
            class="cp-hue"
            onpointerdown={handleHueDown}
            onpointermove={(e) => (e.currentTarget as HTMLElement).hasPointerCapture(e.pointerId) && handleHuePointer(e)}
            onpointerup={commitColorGesture}
        >
            <div
                class="cp-hue-cursor"
                style="left: {colorPicker.h * 100}%"
            ></div>
        </div>
        <div class="cp-footer">
            <div class="cp-preview" style="background: {currentCss}"></div>
            <input
                class="field-input cp-hex-input"
                type="text"
                value={currentCss}
                onfocus={selectOnFocus}
                onkeydown={blurOnEnter}
                onchange={(e) => {
                    const raw = (e.target as HTMLInputElement).value.replace(/^#/, "");
                    const v = parseInt(raw, 16);
                    if (!Number.isNaN(v) && colorPicker) {
                        const [h, s, sv] = hexToHsv(v);
                        colorPicker.h = h;
                        colorPicker.s = s;
                        colorPicker.v = sv;
                        syncColorLive();
                        commitColorGesture();
                    }
                }}
            />
        </div>
    </div>
{/if}

<style>
    .inspector {
        position: relative;
        flex: 1;
    }

    .section {
        margin-bottom: 2px;
    }

    .id-section {
        padding: 8px 12px;
    }

    .node-id {
        font-size: 13px;
        font-weight: 600;
        color: var(--text);
    }

    .section-header {
        display: flex;
        align-items: center;
        gap: 8px;
        width: 100%;
        height: 28px;
        padding: 0 12px;
        border: none;
        border-left: 2px solid color-mix(in srgb, var(--section-color) 60%, transparent);
        border-radius: 0;
        background: var(--surface-2-solid);
        color: var(--text);
        cursor: pointer;
        transition: background 150ms var(--ease-out), border-left-color 150ms var(--ease-out);
    }

    .section-header:hover {
        background: var(--surface-3);
        border-left-color: var(--section-color);
    }

    .section-header:active {
        background: rgba(212, 149, 96, 0.08);
        transform: scale(0.98);
    }

    .section-header:focus-visible {
        outline: 1px solid var(--accent);
        outline-offset: -1px;
    }

    .section-header :global(.section-icon) {
        flex-shrink: 0;
        color: var(--section-color);
        opacity: 0.9;
    }

    .section-label {
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: var(--text-muted);
        flex: 1;
        text-align: left;
    }

    .section-header.flag {
        cursor: default;
    }

    .unregistered {
        opacity: 0.5;
    }

    .unregistered .section-header {
        cursor: default;
    }

    .unregistered .section-header:active {
        transform: none;
    }

    .missing-hint {
        padding: 4px 12px 6px 36px;
        font-size: 10px;
        color: var(--text-muted);
        letter-spacing: 0.01em;
    }

    .section-header :global(.section-chevron) {
        flex-shrink: 0;
        color: var(--text-muted);
    }

    .section-header :global(.section-chevron.expanded) {
        transform: rotate(90deg);
    }

    .section-body {
        display: grid;
        grid-template-rows: 0fr;
    }

    .section-body.expanded {
        grid-template-rows: 1fr;
    }

    .inspector.animate :global(.section-chevron) {
        transition: transform 180ms var(--ease-out);
    }

    .inspector.animate .section-body {
        transition: grid-template-rows 180ms var(--ease-out);
    }

    .section-body-inner {
        overflow: hidden;
    }

    .section-fields {
        padding: 8px 12px;
    }

    .field-row {
        display: flex;
        align-items: center;
        gap: 8px;
        min-height: 24px;
        margin-bottom: 4px;
    }

    .field-label {
        width: 72px;
        flex-shrink: 0;
        font-size: 11px;
        color: var(--text-secondary);
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
        user-select: none;
        pointer-events: none;
    }

    .drag-label {
        cursor: ew-resize;
        pointer-events: auto;
    }

    .field-input {
        flex: 1;
        min-width: 0;
        height: 22px;
        padding: 0 8px;
        border: 1px solid var(--border);
        border-radius: 4px;
        background: var(--bg);
        color: var(--text);
        font-size: 11px;
        font-family: "JetBrains Mono", monospace;
        outline: none;
        transition: border-color 150ms var(--ease-out), background 150ms var(--ease-out), box-shadow 150ms var(--ease-out);
    }

    .field-input:hover:not(:focus) {
        border-color: var(--surface-3);
    }

    .field-input:focus {
        border-color: var(--accent);
        background: var(--surface-4);
        box-shadow: 0 0 0 2px rgba(212, 149, 96, 0.15);
    }

    .field-input::selection {
        background: rgba(212, 149, 96, 0.3);
    }

    .field-input::-webkit-inner-spin-button,
    .field-input::-webkit-outer-spin-button {
        -webkit-appearance: none;
        appearance: none;
        margin: 0;
    }

    .field-input[type="number"] {
        -moz-appearance: textfield;
        appearance: textfield;
    }

    .field-select {
        cursor: pointer;
        -webkit-appearance: none;
        appearance: none;
        background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='8' height='5' viewBox='0 0 8 5'%3E%3Cpath d='M0 0l4 5 4-5z' fill='%23888'/%3E%3C/svg%3E");
        background-repeat: no-repeat;
        background-position: right 6px center;
        padding-right: 20px;
    }

    .field-select option {
        background: #2a2928;
        color: #e8e4df;
    }

    .vec-inputs {
        flex: 1;
        display: flex;
        gap: 4px;
        min-width: 0;
    }

    .vec-field {
        flex: 1;
        display: flex;
        align-items: center;
        min-width: 0;
    }

    .vec-label {
        font-size: 11px;
        font-weight: 600;
        color: var(--text-muted);
        width: 14px;
        flex-shrink: 0;
        text-align: center;
    }

    .vec-label.drag-label {
        cursor: ew-resize;
        user-select: none;
    }

    .vec-input {
        flex: 1;
    }

    .field-value {
        flex: 1;
        font-size: 11px;
        color: var(--text-muted);
        font-family: "JetBrains Mono", monospace;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }

    .hint {
        padding: 16px 12px;
        font-size: 12px;
        color: var(--text-muted);
    }

    .add-component {
        padding: 10px 12px 24px;
    }

    .add-component-btn {
        display: flex;
        align-items: center;
        gap: 6px;
        width: 100%;
        height: 28px;
        padding: 0 8px;
        border: 1px dashed var(--border);
        border-radius: 4px;
        background: transparent;
        color: var(--text-muted);
        font-size: 11px;
        font-family: inherit;
        cursor: pointer;
        transition: all 150ms var(--ease-out);
    }

    .add-component-btn:hover {
        border-color: var(--accent);
        color: var(--accent);
        background: rgba(212, 149, 96, 0.04);
    }

    .add-component-btn:active {
        transform: scale(0.97);
    }

    .add-component-btn :global(svg) {
        flex-shrink: 0;
    }

    .add-component-picker {
        background: rgba(38, 37, 36, 0.85);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid var(--border);
        border-radius: 6px;
        z-index: 100;
        overflow: hidden;
        animation: dropdown-appear 150ms var(--ease-out);
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3), 0 1px 4px rgba(0, 0, 0, 0.2);
    }

    @keyframes dropdown-appear {
        from { opacity: 0; transform: translateY(var(--slide, 4px)); }
        to { opacity: 1; transform: translateY(0); }
    }

    .add-component-search {
        width: 100%;
        height: 30px;
        padding: 0 8px;
        border: none;
        border-bottom: 1px solid var(--border);
        background: transparent;
        color: var(--text);
        font-size: 11px;
        font-family: inherit;
        outline: none;
    }

    .add-component-search::placeholder {
        color: var(--text-muted);
    }

    .add-component-list {
        max-height: 320px;
        overflow-y: auto;
        padding: 4px;
    }

    .add-component-list::-webkit-scrollbar {
        width: 4px;
    }

    .add-component-list::-webkit-scrollbar-thumb {
        background: transparent;
        border-radius: 2px;
        transition: background 200ms var(--ease-out);
    }

    .add-component-list:hover::-webkit-scrollbar-thumb {
        background: var(--border);
    }

    .add-group {
        animation: group-appear 160ms var(--ease-out) var(--group-delay, 0ms) both;
    }

    .add-group + .add-group {
        margin-top: 2px;
        padding-top: 2px;
        border-top: 1px solid var(--border);
    }

    @keyframes group-appear {
        from { opacity: 0; transform: translateY(4px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .add-group-header {
        display: flex;
        align-items: center;
        gap: 6px;
        padding: 4px 8px 2px;
    }

    .add-group-dot {
        width: 5px;
        height: 5px;
        border-radius: 50%;
        flex-shrink: 0;
    }

    .add-group-label {
        font-size: 9px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: var(--text-muted);
    }

    .add-component-item {
        display: flex;
        align-items: center;
        gap: 8px;
        width: 100%;
        padding: 4px 8px;
        border: none;
        border-radius: 4px;
        background: transparent;
        color: var(--text-secondary);
        font-size: 11px;
        font-family: inherit;
        text-align: left;
        cursor: pointer;
        transition: background 120ms var(--ease-out), color 120ms var(--ease-out);
    }

    .add-component-item:hover,
    .add-component-item.focused {
        background: var(--surface-2);
        color: var(--text);
    }

    .add-component-item :global(.add-component-icon) {
        flex-shrink: 0;
    }

    .add-component-empty {
        padding: 12px 8px;
        font-size: 11px;
        color: var(--text-muted);
        text-align: center;
    }

    .inspector.section-dragging {
        cursor: grabbing;
    }

    .drag-source-section {
        opacity: 0.4;
    }

    .section-drop-indicator {
        height: 2px;
        margin: 0 8px;
        background: var(--accent);
        border-radius: 1px;
    }

    .context-menu {
        position: fixed;
        min-width: 140px;
        padding: 4px;
        background: rgba(38, 37, 36, 0.85);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid var(--border);
        border-radius: 6px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3), 0 1px 4px rgba(0, 0, 0, 0.2);
        z-index: 100;
    }

    .context-item {
        display: flex;
        align-items: center;
        width: 100%;
        padding: 4px 8px;
        border: none;
        border-radius: 4px;
        background: transparent;
        color: var(--text-secondary);
        font-size: 12px;
        font-family: inherit;
        text-align: left;
        cursor: pointer;
        white-space: nowrap;
        transition: background 150ms var(--ease-out), color 150ms var(--ease-out);
    }

    .context-item:hover {
        background: var(--surface-2);
        color: var(--text);
    }

    .context-item:active {
        background: rgba(212, 149, 96, 0.08);
        transform: scale(0.96);
    }

    .context-item:focus-visible {
        outline: 1px solid var(--accent);
        outline-offset: -1px;
    }

    .color-swatch {
        width: 22px;
        height: 22px;
        flex-shrink: 0;
        border: 1px solid var(--border);
        border-radius: 4px;
        cursor: pointer;
        padding: 0;
        transition: border-color 150ms var(--ease-out);
    }

    .color-swatch:hover {
        border-color: var(--accent);
    }


    .color-picker-popover {
        position: fixed;
        width: 200px;
        padding: 8px;
        background: rgba(38, 37, 36, 0.85);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid var(--border);
        border-radius: 6px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3), 0 1px 4px rgba(0, 0, 0, 0.2);
        z-index: 200;
        animation: dropdown-appear 150ms var(--ease-out);
    }

    .cp-sv {
        position: relative;
        width: 100%;
        height: 150px;
        border-radius: 4px;
        cursor: crosshair;
        overflow: hidden;
    }

    .cp-sv-white {
        position: absolute;
        inset: 0;
        background: linear-gradient(to right, white, transparent);
    }

    .cp-sv-black {
        position: absolute;
        inset: 0;
        background: linear-gradient(to top, black, transparent);
    }

    .cp-sv-cursor {
        position: absolute;
        width: 10px;
        height: 10px;
        border: 2px solid white;
        border-radius: 50%;
        box-shadow: 0 0 2px rgba(0, 0, 0, 0.6);
        transform: translate(-50%, -50%);
        pointer-events: none;
    }

    .cp-hue {
        position: relative;
        width: 100%;
        height: 12px;
        margin-top: 8px;
        border-radius: 6px;
        background: linear-gradient(to right, #f00, #ff0, #0f0, #0ff, #00f, #f0f, #f00);
        cursor: pointer;
    }

    .cp-hue-cursor {
        position: absolute;
        top: 50%;
        width: 10px;
        height: 10px;
        border: 2px solid white;
        border-radius: 50%;
        box-shadow: 0 0 2px rgba(0, 0, 0, 0.6);
        transform: translate(-50%, -50%);
        pointer-events: none;
    }

    .cp-footer {
        display: flex;
        align-items: center;
        gap: 6px;
        margin-top: 8px;
    }

    .cp-preview {
        width: 22px;
        height: 22px;
        flex-shrink: 0;
        border: 1px solid var(--border);
        border-radius: 4px;
    }

    .cp-hex-input {
        flex: 1;
        font-family: "JetBrains Mono", monospace;
    }

</style>
