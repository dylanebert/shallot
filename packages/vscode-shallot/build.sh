#!/bin/bash
set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
TMP=$(mktemp -d)
trap "rm -rf $TMP" EXIT

cp "$DIR"/package.json "$DIR"/language-configuration.json "$DIR"/icon.png "$TMP/"
cp -r "$DIR"/syntaxes "$TMP/"

cd "$TMP"
npx @vscode/vsce package --allow-missing-repository
VSIX=$(ls *.vsix)
CLI="${SHALLOT_CODE_CLI:-cursor}"
"$CLI" --install-extension "$VSIX"
echo "Installed $VSIX"
