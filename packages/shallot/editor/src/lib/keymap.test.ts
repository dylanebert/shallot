import { describe, expect, test } from "bun:test";
import { resolveShortcut } from "./keymap";

// build a minimal keydown; `tag` stands in for the focused element's tagName
function key(
    over: Partial<{
        key: string;
        ctrlKey: boolean;
        metaKey: boolean;
        shiftKey: boolean;
        tag: string;
    }>,
) {
    return {
        key: over.key ?? "",
        ctrlKey: over.ctrlKey ?? false,
        metaKey: over.metaKey ?? false,
        shiftKey: over.shiftKey ?? false,
        target: (over.tag ? { tagName: over.tag } : null) as unknown as EventTarget | null,
    };
}

describe("resolveShortcut", () => {
    test("ctrl/cmd + s saves", () => {
        expect(resolveShortcut(key({ key: "s", ctrlKey: true }))).toBe("save");
        expect(resolveShortcut(key({ key: "s", metaKey: true }))).toBe("save");
    });

    test("ctrl/cmd + z undoes", () => {
        expect(resolveShortcut(key({ key: "z", ctrlKey: true }))).toBe("undo");
    });

    test("ctrl/cmd + shift + z and ctrl/cmd + y both redo", () => {
        // a real shift chord reports the shifted character — e.key is "Z", never "z"
        expect(resolveShortcut(key({ key: "Z", ctrlKey: true, shiftKey: true }))).toBe("redo");
        expect(resolveShortcut(key({ key: "Z", metaKey: true, shiftKey: true }))).toBe("redo");
        expect(resolveShortcut(key({ key: "y", metaKey: true }))).toBe("redo");
    });

    test("ctrl/cmd + backslash toggles the sidebar", () => {
        expect(resolveShortcut(key({ key: "\\", ctrlKey: true }))).toBe("toggle-sidebar");
    });

    test("F2 renames, but never while a field is focused", () => {
        expect(resolveShortcut(key({ key: "F2" }))).toBe("rename");
        expect(resolveShortcut(key({ key: "F2", tag: "INPUT" }))).toBeNull();
        expect(resolveShortcut(key({ key: "F2", tag: "TEXTAREA" }))).toBeNull();
    });

    test("Delete/Backspace deletes, but not while typing or with a modifier held", () => {
        expect(resolveShortcut(key({ key: "Delete" }))).toBe("delete");
        expect(resolveShortcut(key({ key: "Backspace" }))).toBe("delete");
        expect(resolveShortcut(key({ key: "Delete", tag: "INPUT" }))).toBeNull();
        expect(resolveShortcut(key({ key: "Backspace", ctrlKey: true }))).toBeNull();
    });

    test("? opens help, but never while a field is focused", () => {
        expect(resolveShortcut(key({ key: "?" }))).toBe("help");
        expect(resolveShortcut(key({ key: "?", tag: "INPUT" }))).toBeNull();
    });

    test("an unmapped key resolves to nothing", () => {
        expect(resolveShortcut(key({ key: "a" }))).toBeNull();
        expect(resolveShortcut(key({ key: "Enter" }))).toBeNull();
    });
});
