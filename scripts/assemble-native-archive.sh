#!/usr/bin/env bash
set -e

CRATE="${CRATE:-crates/native}"
STAGE="${STAGE:-stage}"
CARGO_MESSAGES="${CARGO_MESSAGES:-cargo-messages.json}"
rm -rf "$STAGE"
rm -f "$ARCHIVE"
mkdir -p "$STAGE"

cp "$CRATE/target/$TARGET/release/$BINARY" "$STAGE/"

if [ "$MODE" = "portable" ]; then
  if echo "$TARGET" | grep -q darwin; then
    CEF_DIR=$(node scripts/native-runtime.mjs "$CARGO_MESSAGES" "Chromium Embedded Framework.framework")
    cp "$CRATE/target/$TARGET/release/shallot-helper" "$STAGE/"
    ditto "$CEF_DIR/Chromium Embedded Framework.framework" "$STAGE/Chromium Embedded Framework.framework"

    RES="$STAGE/Chromium Embedded Framework.framework/Resources"
    if [ -d "$RES" ]; then
      for lproj in "$RES"/*.lproj; do
        base="$(basename "$lproj")"
        if [ "$base" != "en.lproj" ] && [ "$base" != "en_US.lproj" ]; then
          rm -rf "$lproj"
        fi
      done
      if [ -d "$RES/locales" ]; then
        for pak in "$RES/locales"/*.pak; do
          base="$(basename "$pak")"
          if [ "$base" != "en-US.pak" ]; then
            rm -f "$pak"
          fi
        done
      fi
    fi
    strip -x -S "$STAGE/Chromium Embedded Framework.framework/Chromium Embedded Framework" 2>/dev/null || true

  elif echo "$TARGET" | grep -q windows; then
    MARKER="$(node scripts/native-runtime.mjs "$CARGO_MESSAGES" libcef.dll)/libcef.dll"
    CEF_DIR="$(dirname "$MARKER")"
    mkdir -p "$STAGE/cef"
    for f in \
      libcef.dll chrome_elf.dll d3dcompiler_47.dll dxcompiler.dll dxil.dll \
      libEGL.dll libGLESv2.dll vulkan-1.dll icudtl.dat snapshot_blob.bin \
      v8_context_snapshot.bin chrome_100_percent.pak chrome_200_percent.pak \
      resources.pak vk_swiftshader.dll vk_swiftshader_icd.json; do
      [ -f "$CEF_DIR/$f" ] && cp "$CEF_DIR/$f" "$STAGE/cef/"
    done
    mkdir -p "$STAGE/cef/locales"
    [ -f "$CEF_DIR/locales/en-US.pak" ] && cp "$CEF_DIR/locales/en-US.pak" "$STAGE/cef/locales/"

  else
    MARKER="$(node scripts/native-runtime.mjs "$CARGO_MESSAGES" libcef.so)/libcef.so"
    CEF_DIR="$(dirname "$MARKER")"
    mkdir -p "$STAGE/cef"
    for f in \
      libcef.so libEGL.so libGLESv2.so libvulkan.so.1 icudtl.dat \
      v8_context_snapshot.bin chrome-sandbox chrome_100_percent.pak \
      chrome_200_percent.pak resources.pak libvk_swiftshader.so; do
      [ -f "$CEF_DIR/$f" ] && cp "$CEF_DIR/$f" "$STAGE/cef/"
    done
    mkdir -p "$STAGE/cef/locales"
    [ -f "$CEF_DIR/locales/en-US.pak" ] && cp "$CEF_DIR/locales/en-US.pak" "$STAGE/cef/locales/"
    if [ -d "$CEF_DIR/swiftshader" ]; then cp -r "$CEF_DIR/swiftshader" "$STAGE/cef/"; fi
    strip "$STAGE/cef/libcef.so" 2>/dev/null || true
  fi
fi

tar -czf "$ARCHIVE" -C "$STAGE" .
