#!/bin/sh
set -eu

check_only=false
if [ "$#" -eq 1 ] && [ "$1" = "--check-only" ]; then
    check_only=true
elif [ "$#" -ne 0 ]; then
    echo "Usage: sh scripts/browser.sh [--check-only]" >&2
    exit 2
fi

subjects=$(
    find examples scripts src -type f \( -name 'vite.config.ts' -o -name 'playwright.config.ts' \) \
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
    if [ ! -f "$subject/vite.config.ts" ]; then
        echo "Missing Vite config: $subject/vite.config.ts" >&2
        exit 1
    fi
    if [ ! -f "$subject/playwright.config.ts" ]; then
        echo "Missing browser config: $subject/playwright.config.ts" >&2
        exit 1
    fi
done

if [ "$check_only" = true ]; then
    exit 0
fi

printf '%s\n' "$subjects" | while IFS= read -r subject; do
    # A subject with no browser check yet, as a fresh minimal project, has nothing for Playwright to run.
    if [ -z "$(find "$subject" -name node_modules -prune -o -type f -name '*.e2e.ts' -print)" ]; then
        echo "No browser checks: $subject"
        continue
    fi

    timeout=$(bun -e 'import { TIMEOUT } from "./src/project/testing.js"; console.log(TIMEOUT.browser)')
    echo "Browser subject: $subject"
    report=$(mktemp)
    status=0
    PLAYWRIGHT_JUNIT_OUTPUT_FILE="$report" bunx playwright test -c "$subject" --global-timeout="$timeout" --workers=1 --reporter=list,junit || status=$?
    bun scripts/report-slow-tests.ts "$report" "" "$subject"
    rm -f "$report"
    [ "$status" -eq 0 ] || exit "$status"
done
