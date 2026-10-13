#!/usr/bin/env bash
set -euo pipefail

setup_command='bash scripts/native-linux-setup.sh'
bun_install='curl -fsSL https://bun.sh/install | bash'
rustup_install="curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"

check() {
    local missing=() missing_bun=() missing_rust=() missing_system=()
    for command in bun bunx; do
        command -v "$command" >/dev/null 2>&1 || { missing+=("$command"); missing_bun+=("$command"); }
    done
    for command in cargo rustc; do
        command -v "$command" >/dev/null 2>&1 || { missing+=("$command"); missing_rust+=("$command"); }
    done
    for command in cc pkg-config cmake ninja Xvfb tar python3; do
        command -v "$command" >/dev/null 2>&1 || { missing+=("$command"); missing_system+=("$command"); }
    done
    if ! pkg-config --exists x11 2>/dev/null; then
        missing+=("X11 development headers (libx11-dev)")
        missing_system+=("X11 development headers (libx11-dev)")
    fi
    local -a icd_candidates=()
    if [[ -n "${VK_ICD_FILENAMES:-}" ]]; then
        IFS=: read -r -a icd_candidates <<< "$VK_ICD_FILENAMES"
    else
        icd_candidates=(/usr/share/vulkan/icd.d/*lvp*.json /usr/share/vulkan/icd.d/*swrast*.json)
    fi
    local icd=""
    for candidate in "${icd_candidates[@]}"; do
        if [[ -f "$candidate" && "${candidate##*/}" =~ (lvp|swrast) ]]; then
            icd="$candidate"
            break
        fi
    done
    if [[ -z "$icd" ]]; then
        missing+=("Mesa software Vulkan ICD (mesa-vulkan-drivers)")
        missing_system+=("Mesa software Vulkan ICD (mesa-vulkan-drivers)")
    fi
    if ((${#missing[@]})); then
        printf 'Missing Linux native smoke prerequisites: %s.\n' "$(IFS=', '; echo "${missing[*]}")" >&2
        if ((${#missing_system[@]})); then
            printf 'Install system packages with `%s`.\n' "$setup_command" >&2
        fi
        if ((${#missing_bun[@]})); then
            printf 'Install Bun with `%s`.\n' "$bun_install" >&2
        fi
        if ((${#missing_rust[@]})); then
            printf 'Install Rust with `%s`.\n' "$rustup_install" >&2
        fi
        if [[ -z "$icd" ]]; then
            printf 'Alternatively, set VK_ICD_FILENAMES to a Mesa lvp ICD.\n' >&2
        fi
        return 1
    fi
    printf 'Linux native smoke prerequisites are ready (Xvfb, X11, Mesa software Vulkan).\n'
}

if [[ "${1:-}" == --check ]]; then
    check
    exit
fi
if (($#)); then
    printf 'Usage: %s [--check]\n' "$0" >&2
    exit 2
fi

if command -v apt-get >/dev/null 2>&1; then
    sudo apt-get update
    sudo apt-get install -y build-essential cmake ninja-build libx11-dev pkg-config xvfb mesa-vulkan-drivers tar python3
elif command -v pacman >/dev/null 2>&1; then
    sudo pacman -S --needed base-devel cmake ninja libx11 pkgconf xorg-server-xvfb vulkan-swrast tar python
else
    printf 'Unsupported Linux package manager; install a C compiler, cmake, ninja, X11 development headers, pkg-config, xvfb, mesa-vulkan-drivers, tar, and python3.\n' >&2
    exit 1
fi
check
