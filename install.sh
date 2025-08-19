#!/usr/bin/env bash

set -euo pipefail

if [ -n "${BASH_SOURCE[0]:-}" ]; then
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
else
    SCRIPT_DIR=$(mktemp -d)
    REMOTE_INSTALL=true
    trap 'rm -rf "$SCRIPT_DIR"' EXIT
fi

EARTH_RED='\033[0;31m'
SAGE_GREEN='\033[0;32m'
GOLDEN_BROWN='\033[0;33m'
CLAY_BLUE='\033[2;34m'
MOSS_GREEN='\033[2;36m'
NC='\033[0m'

TARGET_DIR=""
INSTALL_CONTEXT7="y"
OVERWRITE_ALL="y"
SKIP_ALL="n"

print_color() {
    local color=$1
    shift
    echo -e "${color}$@${NC}"
}

download_file() {
    curl -fsSL "$1" -o "$2" 2>/dev/null || {
        print_color "$EARTH_RED" "❌ Failed to download $(basename "$2")"
        exit 1
    }
}

download_shallot_files() {
    [ "${REMOTE_INSTALL:-false}" != "true" ] && return
    
    print_color "$GOLDEN_BROWN" "Downloading files..."
    local base="https://raw.githubusercontent.com/dylanebert/shallot/main"
    
    mkdir -p "$SCRIPT_DIR/commands" "$SCRIPT_DIR/layers"
    
    for cmd in nourish peel plant; do
        download_file "$base/commands/$cmd.md" "$SCRIPT_DIR/commands/$cmd.md"
    done
    
    download_file "$base/layers/context-template.md" "$SCRIPT_DIR/layers/context-template.md"
    
    download_file "$base/layers/structure.md" "$SCRIPT_DIR/layers/structure.md"
    download_file "$base/layers/CLAUDE.md" "$SCRIPT_DIR/layers/CLAUDE.md"
    
    print_color "$SAGE_GREEN" "✓ Files downloaded"
}

check_claude_code() {
    print_color "$GOLDEN_BROWN" "Checking prerequisites..."
    
    if ! command -v claude &> /dev/null; then
        print_color "$EARTH_RED" "❌ Claude Code not found"
        echo "Install from: https://github.com/anthropics/claude-code"
        exit 1
    fi
    
    print_color "$SAGE_GREEN" "✓ Claude Code installed"
}

check_required_tools() {
    local missing=()
    for tool in grep cat mkdir cp; do
        command -v "$tool" &> /dev/null || missing+=("$tool")
    done
    
    if [ ${#missing[@]} -ne 0 ]; then
        print_color "$EARTH_RED" "❌ Missing tools: ${missing[*]}"
        exit 1
    fi
    
    print_color "$SAGE_GREEN" "✓ Tools available"
}

get_target_directory() {
    TARGET_DIR="${INSTALLER_ORIGINAL_PWD:-$(pwd)}"
    
    if [ "$TARGET_DIR" = "$SCRIPT_DIR" ]; then
        print_color "$EARTH_RED" "❌ Cannot install into source directory"
        exit 1
    fi
    
    print_color "$SAGE_GREEN" "✓ Target: $TARGET_DIR"
}

prompt_optional_components() {
    print_color "$MOSS_GREEN" "Context7 MCP Server will be installed"
}

create_directories() {
    print_color "$GOLDEN_BROWN" "Creating directories..."
    
    for dir in "layers" ".claude/commands"; do
        mkdir -p "$TARGET_DIR/$dir"
    done
    
    print_color "$SAGE_GREEN" "✓ Directories created"
}

handle_file_conflict() {
    local source_file="$1"
    local target_file="$2"
    local file_type="$3"
    
    # Always overwrite existing files
    cp "$source_file" "$target_file"
    print_color "$SAGE_GREEN" "✓ Updated $(basename "$target_file")"
    return 0
}

copy_with_check() {
    local source="$1"
    local target_file="$2"
    local file_type="$3"
    
    if [ -f "$target_file" ]; then
        handle_file_conflict "$source" "$target_file" "$file_type"
    else
        cp "$source" "$target_file"
    fi
}


copy_framework_files() {
    print_color "$GOLDEN_BROWN" "Copying files..."
    echo
    
    [ -d "$SCRIPT_DIR/commands" ] && \
        for file in "$SCRIPT_DIR/commands/"*.md; do
            [ -f "$file" ] && copy_with_check "$file" "$TARGET_DIR/.claude/commands/$(basename "$file")" "command"
        done
    
    [ -f "$SCRIPT_DIR/layers/context-template.md" ] && \
        copy_with_check "$SCRIPT_DIR/layers/context-template.md" "$TARGET_DIR/layers/context-template.md" "template"
    
    [ -f "$SCRIPT_DIR/layers/structure.md" ] && \
        copy_with_check "$SCRIPT_DIR/layers/structure.md" "$TARGET_DIR/layers/structure.md" "structure"
    
    if [ -f "$SCRIPT_DIR/layers/CLAUDE.md" ]; then
        if [ -f "$TARGET_DIR/CLAUDE.md" ]; then
            cp "$SCRIPT_DIR/layers/CLAUDE.md" "$TARGET_DIR/CLAUDE.md"
            print_color "$SAGE_GREEN" "✓ Updated CLAUDE.md"
        else
            cp "$SCRIPT_DIR/layers/CLAUDE.md" "$TARGET_DIR/CLAUDE.md"
            print_color "$SAGE_GREEN" "✓ Created CLAUDE.md"
        fi
    fi
    
    print_color "$SAGE_GREEN" "✓ Files copied"
}

install_context7() {
    [ "$INSTALL_CONTEXT7" = "y" ] && {
        echo
        print_color "$GOLDEN_BROWN" "Installing Context7..."
        
        if claude mcp add --transport sse context7 https://mcp.context7.com/sse 2>/dev/null; then
            print_color "$SAGE_GREEN" "✓ Context7 installed successfully"
        else
            print_color "$GOLDEN_BROWN" "→ Context7 already installed"
        fi
        echo
    }
}

display_context7_info() {
    [ "$INSTALL_CONTEXT7" = "y" ] && {
        echo
        print_color "$CLAY_BLUE" "Context7 Installed:"
        print_color "$MOSS_GREEN" "Access library docs with /context7 or claude resolve/get commands"
        echo
    }
}

show_next_steps() {
    echo
    print_color "$SAGE_GREEN" "Installation Complete!"
    echo
    print_color "$GOLDEN_BROWN" "Next:"
    echo "1. Initialize: claude → /plant"
    echo "2. Customize: $TARGET_DIR/CLAUDE.md"
    echo "3. Commands: /plant /nourish /peel"
    echo
    print_color "$CLAY_BLUE" "Docs: $TARGET_DIR/CLAUDE.md"
}

main() {
    echo
    print_color "$CLAY_BLUE" "Shallot Setup"
    echo
    
    check_claude_code
    check_required_tools
    download_shallot_files
    
    get_target_directory
    prompt_optional_components
    
    echo
    print_color "$GOLDEN_BROWN" "Installing to: $TARGET_DIR"
    
    create_directories
    copy_framework_files
    install_context7
    
    display_context7_info
    show_next_steps
}

main "$@"
