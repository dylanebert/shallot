#!/bin/sh
set -eu

subjects=$(
    find examples scripts -type f \( -name 'vite.config.ts' -o -name 'playwright.config.ts' \) \
        -exec dirname {} \; | sort -u
)

if [ -z "$subjects" ]; then
    echo "No browser subjects found" >&2
    exit 1
fi

printf '%s\n' "$subjects" | while IFS= read -r subject; do
    vite_config="$subject/vite.config.ts"
    playwright_config="$subject/playwright.config.ts"

    if [ ! -f "$vite_config" ]; then
        echo "Missing Vite config: $vite_config" >&2
        exit 1
    fi
    if [ ! -f "$playwright_config" ]; then
        echo "Missing browser config: $playwright_config" >&2
        exit 1
    fi

    echo "Browser subject: $subject"
    bunx playwright test -c "$subject"
done
