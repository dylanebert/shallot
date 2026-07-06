<script lang="ts">
    import type { Document, Node, Command } from "@dylanebert/shallot/editor";
    import { type Diagnostic, type State, type Unit } from "@dylanebert/shallot";
    import { formatFields, parseFields } from "@dylanebert/shallot/scene/core";
    import { dependencies, entries, exclusions, isSingleton, kebab, readFields } from "@dylanebert/shallot/ecs/core";
    import { getMeta, nodeLabel, Plus, ChevronRight, TriangleAlert, groupComponents, type ComponentGroup } from "./components";
    import { multiSections, lookup, dotted, wide, type FieldMap } from "./sections";
    import { fieldDocs, plain } from "./fielddocs";
    import { CircleHelp } from "lucide-static";
    import Icon from "./Icon.svelte";
    import SectionLabel from "./SectionLabel.svelte";
    import Select from "./Select.svelte";
    import { dismissOnClickOutside } from "./dismiss";
    import { fit, type Rect } from "./place";

    const point = (x: number, y: number): Rect => ({ left: x, top: y, right: x, bottom: y });

    let { doc, version, diagnostics, ecs, nodeMap, docsFor, ondocs, onchange, onsync, onremove, onadd, onreorder }: {
        doc: Document;
        version: number;
        diagnostics: Diagnostic[];
        ecs: State | null;
        nodeMap: Map<Node, number>;
        /** does a component have a doc page? — gates the section's docs affordance (App passes docFor) */
        docsFor: (component: string) => { slug: string; anchor?: string } | null;
        /** open the docs reader at a component's reference (the context-sensitive help seam) */
        ondocs: (component: string) => void;
        onchange: () => void;
        onsync: (node: Node, attrName: string, fields: Record<string, number | string>) => void;
        onremove: (node: Node, attrName: string) => void;
        onadd: (node: Node, attrName: string) => void;
        onreorder: () => void;
    } = $props();

    let contextMenu: { x: number; y: number; name: string } | null = $state.raw(null);

    // whether a node carries a component, by scene attr or live ECS — the multi-select intersection test
    // for which components are shared (removable from all) vs addable (some node lacks it).
    function hasComponent(node: Node, name: string): boolean {
        if (node.attrs.some((a) => a.name === name)) return true;
        const eid = ecs ? nodeMap?.get(node) : undefined;
        if (eid === undefined) return false;
        const reg = lookup(name);
        return !!(reg && ecs!.has(eid, reg.component as never));
    }

    // remove a component from every selected node that has it, as one undo.
    function removeComponent(name: string) {
        const commands: Command[] = [];
        const removed: Node[] = [];
        for (const node of selectedNodes) {
            const index = node.attrs.findIndex((a) => a.name === name);
            if (index < 0) continue;
            commands.push({ type: "removeAttr", node, name, prev: node.attrs[index].value, index });
            removed.push(node);
        }
        if (commands.length === 0) return;
        if (commands.length === 1) doc.removeAttr(removed[0], name);
        else doc.compound(commands);
        for (const node of removed) onremove(node, name);
        onchange();
    }

    function showContextMenu(e: MouseEvent, name: string) {
        contextMenu = { x: e.clientX, y: e.clientY, name };
    }

    function handleContextRemove() {
        if (!contextMenu) return;
        const { name } = contextMenu;
        contextMenu = null;
        removeComponent(name);
    }

    $effect(() => {
        if (!contextMenu) return;
        return dismissOnClickOutside(() => { contextMenu = null; }, ".context-menu");
    });

    let addOpen = $state(false);
    let addFilter = $state("");
    let addBtnEl: HTMLButtonElement | undefined = $state();
    let pickerStyle = $state("");
    let addUp = $state(false);

    // a component is addable to the selection when at least one selected node lacks it (so Add Component
    // fills the gaps); single-select reduces to "the node doesn't already have it". Derived decorations
    // (glTF's Textured/Skin) are system-owned, never authorable, so they don't offer.
    let availableComponents = $derived.by(() => {
        void version;
        if (selectedNodes.length === 0) return [];
        return [...entries()]
            .filter((r) => !r.traits?.derived)
            .map((r) => r.name)
            .filter((n) => selectedNodes.some((node) => !hasComponent(node, n)));
    });

    let matches = $derived(
        addFilter
            ? availableComponents.filter((n) => n.includes(addFilter.toLowerCase()))
            : availableComponents
    );

    let groups = $derived.by((): ComponentGroup[] => {
        return groupComponents(matches);
    });

    // the flattened visual order (grouped by category) — keyboard nav and the focused-summary
    // index into this, not the raw registration-order `matches`, so the cursor walks rows top to bottom.
    let ordered = $derived(groups.flatMap((g) => g.items));

    let focusIdx = $state(-1);

    // the focused row's docs, shown in the picker footer — tracks the one cursor (mouse hover or
    // keyboard arrow), so it reads the same for both input modes. The summary is author prose
    // (fielddocs); the chips are structured traits read live from the registry, never prose.
    let focusedInfo = $derived.by(() => {
        if (focusIdx < 0 || focusIdx >= ordered.length) return null;
        const name = ordered[focusIdx];
        const raw = fieldDocs[name]?.summary;
        const summary = raw ? plain(raw) : null;
        const requires = dependencies(name);
        const excludes = exclusions(name);
        const singleton = isSingleton(name);
        if (!summary && !singleton && !requires.length && !excludes.length) return null;
        return { summary, requires, excludes, singleton };
    });

    function handlePickerKey(e: KeyboardEvent) {
        const total = ordered.length;
        if (!total) return;
        if (e.key === "ArrowDown") {
            e.preventDefault();
            focusIdx = focusIdx < total - 1 ? focusIdx + 1 : 0;
        } else if (e.key === "ArrowUp") {
            e.preventDefault();
            focusIdx = focusIdx > 0 ? focusIdx - 1 : total - 1;
        } else if (e.key === "Enter" && focusIdx >= 0 && focusIdx < total) {
            e.preventDefault();
            addComponent(ordered[focusIdx]);
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
        addUp = spaceAbove > spaceBelow;
        pickerStyle = addUp
            ? `${lr};bottom:${window.innerHeight - rect.top + 4}px;--slide:4px`
            : `${lr};top:${rect.bottom + 4}px;--slide:-4px`;
        addOpen = true;
        addFilter = "";
        focusIdx = -1;
    }

    // add a component (and its missing dependencies) to every selected node that lacks it, as one undo.
    function addComponent(name: string) {
        if (selectedNodes.length === 0) return;
        const reg = lookup(name);
        const defaults = reg?.traits?.defaults?.() ?? {};
        const value = formatFields(name, defaults);

        const commands: Command[] = [];
        const added: { node: Node; name: string }[] = [];
        for (const node of selectedNodes) {
            if (hasComponent(node, name)) continue;
            const have = new Set(node.attrs.map((a) => a.name));
            commands.push({ type: "addAttr", node, name, value });
            added.push({ node, name });
            for (const dep of dependencies(name)) {
                if (have.has(dep) || hasComponent(node, dep)) continue;
                have.add(dep);
                const depReg = lookup(dep);
                const depDefaults = depReg?.traits?.defaults?.() ?? {};
                commands.push({ type: "addAttr", node, name: dep, value: formatFields(dep, depDefaults) });
                added.push({ node, name: dep });
            }
        }

        if (commands.length === 0) {
            addOpen = false;
            return;
        }
        if (commands.length === 1) {
            const c = commands[0] as Extract<Command, { type: "addAttr" }>;
            doc.addAttr(c.node, c.name, c.value);
        } else {
            doc.compound(commands);
        }

        for (const a of added) onadd(a.node, a.name);

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

    // the selection as an ordered list (insertion order, so the last is the active node) and `selected`
    // = that active node. The inspector edits the whole selection: `sectionViews` shows the components
    // shared across it (mixed fields flagged), and each edit fans out to every selected node as one undo.
    let selectedNodes = $derived.by((): Node[] => {
        void version;
        return [...doc.selection];
    });
    let selected = $derived(selectedNodes.at(-1) ?? null);

    $effect(() => {
        void selected;
        animate = false;
    });

    let sectionViews = $derived.by(() => {
        void version;
        if (selectedNodes.length === 0) return [];
        const eids = selectedNodes.map((n) => (ecs ? nodeMap?.get(n) : undefined));
        return multiSections(selectedNodes, eids, ecs, diagnostics);
    });

    function ensureAttr(node: Node, attrName: string, parsed: FieldMap) {
        if (!node.attrs.find((a) => a.name === attrName)) {
            doc.addAttr(node, attrName, formatFields(attrName, parsed));
            onadd(node, attrName);
        }
    }

    // each node's current field values for a component — from its scene attr, else the live ECS. The
    // per-node read is what lets a multi-edit set one field while preserving every entity's other lanes.
    function parsedOf(node: Node, attrName: string): FieldMap | null {
        const reg = lookup(attrName);
        if (!reg) return null;
        const defaults = reg.traits?.defaults?.() ?? {};
        const attr = node.attrs.find((a) => a.name === attrName);
        if (attr) return attr.value ? { ...defaults, ...parseFields(attrName, attr.value) } : defaults;
        const eid = nodeMap?.get(node);
        if (eid === undefined) return null;
        return { ...defaults, ...readFields(reg.component, eid) };
    }

    // write one field's absolute value to a node inside an open gesture (doc.begin already called). An
    // `alias:` key (euler→quat) maps back to the stored lanes from the node's OWN current values.
    function writeField(node: Node, attrName: string, fieldKey: string, value: number) {
        const parsed = parsedOf(node, attrName);
        if (!parsed) return;
        ensureAttr(node, attrName, parsed);
        if (fieldKey.startsWith("alias:")) {
            const [, base, axis] = fieldKey.split(":");
            const alias = lookup(attrName)?.traits?.aliases?.[base];
            if (!alias) return;
            const realUpdates = alias.write(Number(axis), value, dotted(parsed));
            doc.setAttr(node, attrName, formatFields(attrName, { ...parsed, ...realUpdates }));
            onsync(node, attrName, realUpdates);
        } else {
            doc.setAttr(node, attrName, formatFields(attrName, { ...parsed, [fieldKey]: value }));
            onsync(node, attrName, { [fieldKey]: value });
        }
    }

    // an absolute field edit (typed input, enum, unit), fanned out to every selected node as one undo.
    function updateField(attrName: string, fieldKey: string, value: number) {
        doc.begin();
        for (const node of selectedNodes) writeField(node, attrName, fieldKey, value);
        doc.commit();
        onchange();
    }

    function selectOnFocus(e: FocusEvent) {
        (e.target as HTMLInputElement).select();
    }

    function blurOnEnter(e: KeyboardEvent) {
        if (e.key === "Enter") (e.target as HTMLInputElement).blur();
    }

    // selected unit per unit-field, editor-only display state (storage stays in its base unit). Keyed by
    // component + field, so the choice is a per-field view preference shared across entities; default 0.
    let unitSel: Record<string, number> = $state({});
    const unitKey = (comp: string, key: string) => `${comp}:${key}`;

    function round(v: number): number {
        return Math.round(v * 1000) / 1000;
    }

    let dragSectionIdx: number | null = $state(null);
    let dropSectionIdx: number | null = $state(null);
    let sectionDragging = $state(false);
    let sectionDragStartY = 0;
    let inspectorEl: HTMLElement;

    let dragPointerId: number | null = null;

    function handleSectionDragStart(e: PointerEvent, idx: number) {
        if (e.button !== 0) return;
        if (selectedNodes.length !== 1) return; // reorder targets one node's attrs; ambiguous for many
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

    function dragLabel(e: PointerEvent, attrName: string, fieldKey: string, unit?: Unit) {
        // the scrub drags one field absolute and writes it to every selected node (writeField), so a
        // multi-select scrub converges them and commits as one undo. acc seeds from the active node and
        // holds the shown value (a unit field drags in its shown unit — degrees feel like degrees).
        const targets = [...selectedNodes];
        const active = targets.at(-1);
        if (!active) return;
        const reg = lookup(attrName);
        const fresh = parsedOf(active, attrName);
        if (!fresh) return;
        const label = e.currentTarget as HTMLElement;
        const input = label.closest(".input-cell")?.querySelector("input") as HTMLInputElement | null;
        label.setPointerCapture(e.pointerId);
        const aliasKey = fieldKey.startsWith("alias:");
        const [, base, axis] = aliasKey ? fieldKey.split(":") : [];
        const alias = aliasKey ? reg?.traits?.aliases?.[base] : undefined;
        const stored0 = (fresh[fieldKey] as number) ?? 0;
        let acc = alias ? alias.read(dotted(fresh))[Number(axis)] ?? 0 : unit ? unit.to(stored0) : stored0;
        function onMove(ev: PointerEvent) {
            acc += ev.movementX * 0.01;
            const rounded = round(acc);
            // alias takes the shown axis value; unit stores the base-unit value; raw stores the value
            const value = alias ? rounded : unit ? unit.from(rounded) : rounded;
            // record the scrub into the gesture: auto-prev at first touch (pristine, before readback
            // mirrors the live ECS back to attr.value), coalesced to one undoable entry on commit
            for (const node of targets) writeField(node, attrName, fieldKey, value);
            if (input) input.value = String(rounded);
        }
        function onUp() {
            label.removeEventListener("pointermove", onMove);
            label.removeEventListener("pointerup", onUp);
            label.removeEventListener("pointercancel", onUp);
            doc.commit();
            onchange();
        }
        doc.begin();
        label.addEventListener("pointermove", onMove);
        label.addEventListener("pointerup", onUp);
        // a cancelled pointer must still close the gesture — a left-open one would buffer later edits
        label.addEventListener("pointercancel", onUp);
    }

    // layer-1 field help: hovering the title row's "?" reveals a styled popover with the field's
    // description + type + default, after a short rest so a passing cursor doesn't flash it.
    let hover: { component: string; field: string; top: number; left: number } | null = $state(null);
    let hoverTimer: ReturnType<typeof setTimeout> | null = null;
    const hoverDoc = $derived.by(() => {
        const h = hover;
        return h ? fieldDocs[h.component]?.fields[h.field] : undefined;
    });

    function enterField(e: PointerEvent, component: string, field: string) {
        const el = e.currentTarget as HTMLElement;
        if (hoverTimer) clearTimeout(hoverTimer);
        hoverTimer = setTimeout(() => {
            const r = el.getBoundingClientRect();
            hover = { component, field, top: r.top, left: r.left };
        }, 400);
    }

    function leaveField() {
        if (hoverTimer) {
            clearTimeout(hoverTimer);
            hoverTimer = null;
        }
        hover = null;
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
        live: FieldMap;
        dragging: boolean;
    } | null = null;

    let colorPicker: { key: string; attrName: string; h: number; s: number; v: number; anchor: Rect } | null = $state(null);

    // every selected node the picker writes, with its pristine pre-gesture attr value as the undo prev —
    // captured at open before any live mirror (onsync → readback) moves attr.value, so the commit's
    // prev→next is correct even though the value drifted live. One picker edit is one undo across all.
    let colorTargets: { node: Node; start: string }[] = [];

    function syncColorLive() {
        const cp = colorPicker;
        if (!cp || !pickerCtx) return;
        const hex = hsvToHex(cp.h, cp.s, cp.v);
        pickerCtx.live[cp.key] = hex;
        for (const t of colorTargets) onsync(t.node, cp.attrName, { [cp.key]: hex });
    }

    function openColorPicker(e: MouseEvent, attrName: string, key: string, value: number) {
        if (colorPicker) commitColorGesture();
        const active = selectedNodes.at(-1);
        if (!active) return;
        const r = (e.currentTarget as HTMLElement).getBoundingClientRect();
        const [h, s, v] = hexToHsv(value);
        colorTargets = [];
        for (const node of selectedNodes) {
            const parsed = parsedOf(node, attrName);
            if (!parsed) continue;
            ensureAttr(node, attrName, parsed);
            colorTargets.push({ node, start: node.attrs.find((a) => a.name === attrName)?.value ?? "" });
        }
        const activeParsed = parsedOf(active, attrName) ?? {};
        pickerCtx = {
            node: active,
            startValue: active.attrs.find((a) => a.name === attrName)?.value ?? "",
            live: { ...activeParsed },
            dragging: false,
        };
        colorPicker = { key, attrName, h, s, v, anchor: { left: r.left, top: r.top, right: r.right, bottom: r.bottom } };
    }

    function commitColorGesture() {
        const ctx = pickerCtx;
        const cp = colorPicker;
        if (!ctx || !cp) return;
        ctx.dragging = false;
        const hex = hsvToHex(cp.h, cp.s, cp.v);
        // build each node's final from its captured start (readback-independent) + the new hex; one undo
        const commands: Command[] = [];
        for (const t of colorTargets) {
            const reg = lookup(cp.attrName);
            const defaults = reg?.traits?.defaults?.() ?? {};
            const base = t.start ? { ...defaults, ...parseFields(cp.attrName, t.start) } : defaults;
            const final = formatFields(cp.attrName, { ...base, [cp.key]: hex });
            if (final !== t.start) {
                commands.push({ type: "setAttr", node: t.node, name: cp.attrName, prev: t.start, next: final });
                t.start = final;
            }
        }
        if (commands.length === 0) return;
        if (commands.length === 1) {
            const c = commands[0] as Extract<Command, { type: "setAttr" }>;
            doc.setAttr(c.node, c.name, c.next, c.prev);
        } else {
            doc.compound(commands);
        }
        onchange();
        ctx.startValue = ctx.node.attrs.find((a) => a.name === cp.attrName)?.value ?? ctx.startValue;
        ctx.live = { ...ctx.live, [cp.key]: hex };
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
            const reg = lookup(cp.attrName);
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
    <div class="inspector-head" class:empty={!selected}>
        {#if selectedNodes.length > 1}{version >= 0 ? `${selectedNodes.length} selected` : ""}{:else if selected}{version >= 0 ? nodeLabel(selected) : ""}{:else}No selection{/if}
    </div>
    {#if selected}
        <SectionLabel label="Components" />
        {#each sectionViews as section, sIdx}
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
                        oncontextmenu={(e) => { e.preventDefault(); showContextMenu(e, section.name); }}
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
                    oncontextmenu={(e) => { e.preventDefault(); showContextMenu(e, section.name); }}
                >
                    <Icon icon={meta.icon} size={14} strokeWidth={1.5} class="section-icon" />
                    <span class="section-label">{section.name}</span>
                    {#if hasFields}
                        <Icon icon={ChevronRight} size={12} strokeWidth={2} class="section-chevron {expanded ? 'expanded' : ''}" />
                    {/if}
                </button>
                {#if docsFor(section.name)}
                    <button
                        class="section-docs"
                        class:with-fields={hasFields}
                        title="Open docs"
                        aria-label="{section.name} docs"
                        onclick={() => ondocs(section.name)}
                    ><Icon icon={CircleHelp} size={12} strokeWidth={2} /></button>
                {/if}
                <div class="section-body" class:expanded={hasFields && expanded}>
                    <div class="section-body-inner">
                        <div class="section-fields">
                        {#if section.fields.length > 0}
                            {#each section.fields as field}
                                <div class="field-group" class:wide={wide(field)}>
                                    <div class="field-title">
                                        <span class="field-name">{field.label}</span>
                                        {#if fieldDocs[section.name]?.fields[field.label]}
                                            <button
                                                class="field-help"
                                                aria-label="{field.label} info"
                                                onpointerenter={(e) => enterField(e, section.name, field.label)}
                                                onpointerleave={leaveField}
                                                onclick={() => ondocs(section.name)}
                                            ><Icon icon={CircleHelp} size={11} strokeWidth={2} /></button>
                                        {/if}
                                    </div>
                                    {#if field.type === "float"}
                                        <div class="input-cell">
                                            <span
                                                class="cell-handle axis-label"
                                                role="slider"
                                                tabindex="-1"
                                                aria-valuenow={field.value}
                                                onpointerdown={(e) => dragLabel(e, section.name, field.key)}
                                            >x</span>
                                            <input
                                                class="field-input"
                                                type="number"
                                                step="any"
                                                value={field.mixed ? "" : field.value}
                                                placeholder={field.mixed ? "—" : undefined}
                                                onfocus={selectOnFocus}
                                                onkeydown={blurOnEnter}
                                                onchange={(e) => {
                                                    const v = parseFloat((e.target as HTMLInputElement).value);
                                                    if (!Number.isNaN(v)) updateField(section.name, field.key, v);
                                                }}
                                            />
                                        </div>
                                    {:else if field.type === "vec"}
                                        <div class="vec-cells">
                                            {#each field.axes as axis}
                                                <div class="input-cell">
                                                    <span
                                                        class="cell-handle axis-label"
                                                        role="slider"
                                                        tabindex="-1"
                                                        aria-valuenow={axis.value}
                                                        onpointerdown={(e) => dragLabel(e, section.name, axis.key)}
                                                    >{axis.label}</span>
                                                    <input
                                                        class="field-input vec-input"
                                                        type="number"
                                                        step="any"
                                                        value={axis.mixed ? "" : axis.value}
                                                        placeholder={axis.mixed ? "—" : undefined}
                                                        onfocus={selectOnFocus}
                                                        onkeydown={blurOnEnter}
                                                        onchange={(e) => {
                                                            const v = parseFloat((e.target as HTMLInputElement).value);
                                                            if (!Number.isNaN(v)) updateField(section.name, axis.key, v);
                                                        }}
                                                    />
                                                </div>
                                            {/each}
                                        </div>
                                    {:else if field.type === "enum"}
                                        <Select
                                            title={field.label}
                                            value={field.value}
                                            placeholder={field.mixed ? "—" : undefined}
                                            options={Object.entries(field.options).map(([name, val]) => ({ label: kebab(name), value: val }))}
                                            onchange={(v) => updateField(section.name, field.key, v)}
                                        />
                                    {:else if field.type === "color"}
                                        {@const liveHex = colorPicker && colorPicker.key === field.key && colorPicker.attrName === section.name ? hsvToHex(colorPicker.h, colorPicker.s, colorPicker.v) : field.value}
                                        <div class="color-control">
                                            <button
                                                class="color-swatch"
                                                style="background: {hexToCss(liveHex)}"
                                                aria-label={field.label}
                                                onclick={(e) => openColorPicker(e, section.name, field.key, field.value)}
                                            ></button>
                                            <span class="field-value">{field.mixed ? "—" : hexToCss(liveHex)}</span>
                                        </div>
                                    {:else if field.type === "unit"}
                                        {@const sel = unitSel[unitKey(section.name, field.key)] ?? 0}
                                        {@const u = field.units[sel] ?? field.units[0]}
                                        {@const shown = round(u.to(field.value))}
                                        <div class="input-cell">
                                            <span
                                                class="cell-handle axis-label"
                                                role="slider"
                                                tabindex="-1"
                                                aria-valuenow={shown}
                                                onpointerdown={(e) => dragLabel(e, section.name, field.key, u)}
                                            >x</span>
                                            <div class="unit-attach">
                                                <input
                                                    class="field-input"
                                                    type="number"
                                                    step="any"
                                                    value={field.mixed ? "" : shown}
                                                    placeholder={field.mixed ? "—" : undefined}
                                                    onfocus={selectOnFocus}
                                                    onkeydown={blurOnEnter}
                                                    onchange={(e) => {
                                                        const entered = parseFloat((e.target as HTMLInputElement).value);
                                                        if (!Number.isNaN(entered)) updateField(section.name, field.key, u.from(entered));
                                                    }}
                                                />
                                                <Select
                                                    variant="unit"
                                                    title="{field.label} unit"
                                                    value={sel}
                                                    options={field.units.map((uu, i) => ({ label: uu.label, value: i }))}
                                                    onchange={(v) => {
                                                        unitSel[unitKey(section.name, field.key)] = v;
                                                    }}
                                                />
                                            </div>
                                        </div>
                                    {:else}
                                        <span class="field-value">{field.value}</span>
                                    {/if}
                                </div>
                            {/each}
                        {/if}
                        </div>
                    </div>
                </div>
            </div>
            {/if}
        {/each}
        {#if sectionDragging && dropSectionIdx === sectionViews.length}
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
                    <div class="add-component-picker" class:up={addUp} style={pickerStyle} onkeydown={handlePickerKey}>
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
                                            {@const flatIdx = ordered.indexOf(name)}
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
                        {#if focusedInfo}
                            <div class="add-component-summary">
                                {#if focusedInfo.summary}
                                    <div class="acs-text">{focusedInfo.summary}</div>
                                {/if}
                                {#if focusedInfo.singleton || focusedInfo.requires.length || focusedInfo.excludes.length}
                                    <div class="acs-tags">
                                        {#if focusedInfo.singleton}
                                            <span class="acs-tag">one per scene</span>
                                        {/if}
                                        {#each focusedInfo.requires as req}
                                            <span class="acs-tag">requires {req}</span>
                                        {/each}
                                        {#each focusedInfo.excludes as exc}
                                            <span class="acs-tag acs-tag-warn">excludes {exc}</span>
                                        {/each}
                                    </div>
                                {/if}
                            </div>
                        {/if}
                    </div>
                {/if}
            </div>
        {/if}
    {/if}


</div>

{#if contextMenu}
    <!-- svelte-ignore a11y_no_static_element_interactions -->
    <div class="context-menu" use:fit={{ anchor: point(contextMenu.x, contextMenu.y) }}>
        <button class="context-item" onmousedown={handleContextRemove}>
            Remove
        </button>
    </div>
{/if}

{#if colorPicker}
    {@const hueColor = `hsl(${colorPicker.h * 360}, 100%, 50%)`}
    {@const currentCss = hexToCss(hsvToHex(colorPicker.h, colorPicker.s, colorPicker.v))}
    <!-- svelte-ignore a11y_no_static_element_interactions -->
    <div class="color-picker-popover" use:fit={{ anchor: colorPicker.anchor }}>
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

{#if hover && hoverDoc}
    <div class="field-popover" style="top: {hover.top}px; right: calc(100vw - {hover.left}px + 8px)">
        <div class="fp-head">
            <span class="fp-name">{hover.field}</span>
            <span class="fp-type">{hoverDoc.type}</span>
        </div>
        {#if hoverDoc.description}
            <div class="fp-desc">{plain(hoverDoc.description)}</div>
        {/if}
        <div class="fp-default">default {hoverDoc.default}</div>
    </div>
{/if}

<style>
    .inspector {
        position: relative;
        flex: 1;
    }

    .section {
        position: relative;
        margin-bottom: 2px;
    }

    /* the component's docs affordance — quiet until the section is hovered (editor-ui.md gate 2), sitting
       just left of the chevron so it never overlaps it. A sibling of the header button, not nested. */
    .section-docs {
        position: absolute;
        top: 0;
        right: 10px;
        height: 28px;
        display: flex;
        align-items: center;
        justify-content: center;
        width: 18px;
        padding: 0;
        border: none;
        background: transparent;
        color: var(--text-muted);
        opacity: 0;
        cursor: pointer;
        transition: opacity 150ms var(--ease-out), color 150ms var(--ease-out);
    }

    .section-docs.with-fields {
        right: 30px;
    }

    .section:hover .section-docs,
    .section-docs:focus-visible {
        opacity: 0.65;
    }

    .section-docs:hover {
        opacity: 1;
        color: var(--accent);
    }

    .inspector-head {
        height: var(--header-h);
        line-height: var(--header-h);
        padding: 0 12px;
        font-size: 13px;
        font-weight: 600;
        color: var(--text);
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        position: sticky;
        top: 0;
        background: var(--surface-1-solid);
        z-index: 1;
    }

    .inspector-head.empty {
        color: var(--text-muted);
        font-weight: 500;
    }

    .section-header {
        display: flex;
        align-items: center;
        gap: 8px;
        width: 100%;
        height: 28px;
        padding: 0 12px;
        border: none;
        border-left: 2px solid var(--section-color);
        border-radius: 0;
        background: var(--surface-2-solid);
        color: var(--text);
        cursor: pointer;
        transition: background 150ms var(--ease-out);
    }

    .section-header:hover {
        background: var(--surface-3);
    }

    .section-header:active {
        background: color-mix(in srgb, var(--accent) 8%, transparent);
        transform: scale(0.98);
    }

    .section-header:focus-visible {
        outline: 1px solid var(--accent);
        outline-offset: -1px;
    }

    .section-header :global(.section-icon) {
        flex-shrink: 0;
        color: var(--section-color);
    }

    .section-label {
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: var(--text-secondary);
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
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 8px 10px;
    }

    /* the standard field element: a minimal title row over its value row. Scalars pack two per grid
       row; multi-component fields (vec) span the full width. */
    .field-group {
        display: flex;
        flex-direction: column;
        gap: 3px;
        min-width: 0;
    }

    .field-group.wide {
        grid-column: 1 / -1;
    }

    .field-title {
        display: flex;
        align-items: center;
        gap: 4px;
        min-width: 0;
        height: 16px;
    }

    .field-name {
        min-width: 0;
        font-size: 11px;
        color: var(--text-muted);
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
        user-select: none;
    }

    /* the subtle "?": hover reveals the field's popover, a click opens the component's docs reference */
    .field-help {
        flex-shrink: 0;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        padding: 0;
        border: none;
        background: none;
        color: var(--text-muted);
        opacity: 0.7;
        cursor: help;
        transition: opacity 150ms var(--ease-out), color 150ms var(--ease-out);
    }

    .field-help:hover {
        opacity: 1;
        color: var(--accent);
    }

    .field-input {
        box-sizing: border-box;
        width: 100%;
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
        box-shadow: 0 0 0 2px color-mix(in srgb, var(--accent) 15%, transparent);
    }

    .field-input::selection {
        background: color-mix(in srgb, var(--accent) 30%, transparent);
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

    /* a value cell: a leading scrub handle (a type icon for scalars, an axis letter for vec components)
       then the editable input. The handle scrubs; the input is a plain text field. */
    .input-cell {
        display: flex;
        align-items: center;
        gap: 4px;
        min-width: 0;
    }

    .input-cell .field-input {
        flex: 1;
        width: auto;
    }

    .vec-cells {
        display: flex;
        flex-wrap: wrap;
        gap: 4px 6px;
    }

    .vec-cells .input-cell {
        flex: 1 1 auto;
        min-width: 56px;
    }

    .cell-handle {
        flex-shrink: 0;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 14px;
        color: var(--text-muted);
        cursor: ew-resize;
        user-select: none;
    }

    .cell-handle:hover {
        color: var(--text-secondary);
    }

    .axis-label {
        font-size: 10px;
        font-weight: 600;
    }

    .color-control {
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* a unit field: the number input keeps its own field box; the unit dropdown attaches flush to its
       right (zero gap), outside the field's border — so the dropdown stays the field's appendage, not
       part of it. */
    .unit-attach {
        flex: 1;
        display: flex;
        align-items: center;
        min-width: 0;
    }

    .field-value {
        font-size: 11px;
        color: var(--text-muted);
        font-family: "JetBrains Mono", monospace;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }

    /* layer-1 field help, anchored to the left of the hovered "?" */
    .field-popover {
        position: fixed;
        z-index: 100;
        max-width: 240px;
        padding: 8px 10px;
        background: var(--surface-2-solid);
        border: 1px solid var(--border);
        border-radius: 6px;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.35);
        pointer-events: none;
        display: flex;
        flex-direction: column;
        gap: 5px;
    }

    .fp-head {
        display: flex;
        align-items: baseline;
        justify-content: space-between;
        gap: 10px;
    }

    .fp-name {
        font-size: 12px;
        font-weight: 600;
        color: var(--text);
    }

    .fp-type {
        font-size: 10px;
        font-family: "JetBrains Mono", monospace;
        color: var(--text-muted);
        background: var(--surface-3-solid);
        padding: 1px 5px;
        border-radius: 3px;
        white-space: nowrap;
    }

    .fp-desc {
        font-size: 11px;
        line-height: 1.4;
        color: var(--text-secondary);
    }

    .fp-default {
        font-size: 10px;
        font-family: "JetBrains Mono", monospace;
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
        background: color-mix(in srgb, var(--accent) 4%, transparent);
    }

    .add-component-btn:active {
        transform: scale(0.97);
    }

    .add-component-btn :global(svg) {
        flex-shrink: 0;
    }

    .add-component-picker {
        display: flex;
        flex-direction: column;
        background: var(--surface-3-solid);
        border: 1px solid var(--border);
        border-radius: 6px;
        z-index: 100;
        overflow: hidden;
        animation: dropdown-appear 150ms var(--ease-out);
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3), 0 1px 4px rgba(0, 0, 0, 0.2);
    }

    /* opening upward: pin the list to the button, let the variable-height summary
       grow off the far (top) edge so hovering rows never shifts the list */
    .add-component-picker.up .add-component-summary {
        order: -1;
        border-top: none;
        border-bottom: 1px solid var(--border);
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

    .add-component-summary {
        display: flex;
        flex-direction: column;
        gap: 6px;
        padding: 8px 10px;
        border-top: 1px solid var(--border);
    }

    .acs-text {
        font-size: 11px;
        line-height: 1.4;
        color: var(--text-secondary);
    }

    .acs-tags {
        display: flex;
        flex-wrap: wrap;
        gap: 4px;
    }

    .acs-tag {
        font-size: 10px;
        padding: 1px 6px;
        border-radius: 3px;
        background: var(--surface-2);
        color: var(--text-muted);
        white-space: nowrap;
    }

    .acs-tag-warn {
        color: var(--accent);
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
        background: var(--surface-3-solid);
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
        background: color-mix(in srgb, var(--accent) 8%, transparent);
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
        background: var(--surface-3-solid);
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
