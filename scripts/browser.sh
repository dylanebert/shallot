#!/bin/sh
set -eu

subjects=$(
    find examples scripts -type f \( -name 'vite.config.ts' -o -name 'playwright.config.ts' \) \
        -exec dirname {} \; | sort -u
)
e2e_files=$(
    find . -type d \( -name node_modules -o -name .git \) -prune \
        -o -type f -name '*.e2e.ts' -print
)

printf '%s\n' "$e2e_files" | while IFS= read -r e2e_file; do
    [ -n "$e2e_file" ] || continue
    found=false
    for subject in $subjects; do
        case "$e2e_file" in
            "./$subject/"*) found=true; break ;;
        esac
    done
    if [ "$found" = false ]; then
        echo "Orphan browser test: ${e2e_file#./}" >&2
        exit 1
    fi
done

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
