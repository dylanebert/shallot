import { mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";

/** Keep provider homes private while desktop openers use the invoking session. */
export function browserSessionPath(root: string, session = process.env): string {
    const directory = join(root, "browser-bin");
    mkdirSync(directory);
    const quote = (value: string) => `'${value.replaceAll("'", "'\\''")}'`;
    const names = new Set([
        "HOME",
        "PATH",
        "DISPLAY",
        "WAYLAND_DISPLAY",
        "DBUS_SESSION_BUS_ADDRESS",
        "XAUTHORITY",
        ...Object.keys(session).filter((name) => name.startsWith("XDG_")),
        "XDG_CONFIG_HOME",
        "XDG_CACHE_HOME",
        "XDG_DATA_HOME",
        "XDG_RUNTIME_DIR",
    ]);
    const restore = [...names]
        .map((name) =>
            session[name] === undefined
                ? `unset ${name}`
                : `export ${name}=${quote(session[name]!)}`,
        )
        .join("\n");
    for (const name of ["xdg-open", "open", "sensible-browser"]) {
        const opener = Bun.which(name, { PATH: session.PATH });
        if (!opener) continue;
        writeFileSync(
            join(directory, name),
            `#!/bin/sh\n${restore}\nexec ${quote(opener)} "$@"\n`,
            {
                mode: 0o700,
            },
        );
    }
    return `${directory}:${session.PATH ?? ""}`;
}
