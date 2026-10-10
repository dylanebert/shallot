$ErrorActionPreference = "Stop"

$archive = Join-Path $env:TEMP "handle.zip"
$directory = Join-Path $env:TEMP "handle"
$folderPaths = Get-ChildItem $env:TEMP -Directory -Filter "shallot-packed-capture-*"
Write-Host "Packed-capture scratch directories: $($folderPaths.Count)"
foreach ($folder in $folderPaths) { Write-Host $folder.FullName }

Invoke-WebRequest "https://download.sysinternals.com/files/Handle.zip" -OutFile $archive
Expand-Archive $archive -DestinationPath $directory -Force
$handle = Get-ChildItem $directory -Filter "handle*.exe" | Select-Object -First 1 -ExpandProperty FullName
if (-not $handle) { throw "Sysinternals Handle.exe was not found in $directory" }

foreach ($folder in $folderPaths) {
    Write-Host "`n=== Open handles for $($folder.FullName) ==="
    & $handle -accepteula $folder.FullName
}

Write-Host "`n=== Live Bun, Node and Chromium processes ==="
Get-CimInstance Win32_Process |
    Where-Object { $_.Name -match "^(bun|node|chrome|msedge).*\.exe$" } |
    Select-Object ProcessId, ParentProcessId, Name, CommandLine |
    Format-List
