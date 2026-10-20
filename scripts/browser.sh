#!/bin/sh
set -eu

usage() {
    echo "Usage: sh scripts/browser.sh [--check-only | --shard i/n]" >&2
    exit 2
}

check_only=false
shard_index=1
shard_count=1
if [ "$#" -eq 1 ] && [ "$1" = "--check-only" ]; then
    check_only=true
elif [ "$#" -eq 2 ] && [ "$1" = "--shard" ]; then
    shard_index=${2%/*}
    shard_count=${2#*/}
    case "$shard_index" in ''|0*|*[!0-9]*) usage ;; esac
    case "$shard_count" in ''|0*|*[!0-9]*) usage ;; esac
    [ "$2" = "$shard_index/$shard_count" ] || usage
    [ "$shard_index" -le "$shard_count" ] || usage
elif [ "$#" -ne 0 ]; then
    usage
fi

subjects=$(
    find examples scripts src -type f \( -name 'vite.config.ts' -o -name 'playwright.config.ts' \) \
        -exec dirname {} \; | sort -u
)
# Tracked and untracked files, never gitignored scratch such as .artifacts/.
e2e_files=$(git ls-files --cached --others --exclude-standard -- '*.e2e.ts' | sed 's|^|./|')

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

subject_count=$(printf '%s\n' "$subjects" | wc -l)
printf '%s\n' "$subjects" | awk -v shard="$shard_index" -v count="$shard_count" -v total="$subject_count" \
    'NR > int((shard - 1) * total / count) && NR <= int(shard * total / count)' | while IFS= read -r subject; do
    # A subject with no browser check yet, as a fresh minimal project, has nothing for Playwright to run.
    if [ -z "$(find "$subject" -name node_modules -prune -o -type f -name '*.e2e.ts' -print)" ]; then
        echo "No browser checks: $subject"
        continue
    fi

    timeout=$(bun -e 'import { TIMEOUT } from "./src/project/testing.js"; console.log(TIMEOUT.browser)')
    echo "Browser subject: $subject"
    report=$(mktemp)
    json_report=$(mktemp)
    status=0
    PLAYWRIGHT_JUNIT_OUTPUT_FILE="$report" PLAYWRIGHT_JSON_OUTPUT_FILE="$json_report" bunx playwright test -c "$subject" --global-timeout="$timeout" --workers=1 --output="$subject/test-results" --reporter=list,junit,json || status=$?
    bun scripts/report-slow-tests.ts "$report" "" "$subject" "$json_report"
    rm -f "$report" "$json_report"
    [ "$status" -eq 0 ] || exit "$status"
done
