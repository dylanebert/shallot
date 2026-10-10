$ErrorActionPreference = "Continue"

$bun = (Get-Command bun).Source
$node = (Get-Command node).Source
$root = $env:GITHUB_WORKSPACE
$repro = Join-Path $root "scripts/windows-dawn-repro.cjs"
$dawnDirectory = Join-Path $root "node_modules/webgpu/dist/win32-x64"
$compiler = Join-Path $dawnDirectory "d3dcompiler_47.dll"
if (-not (Test-Path $compiler)) { throw "Missing Dawn dependency: $compiler" }

Write-Host "Dawn directory: $dawnDirectory"
Write-Host "Bun executable: $bun"
Write-Host "Node executable: $node"
Write-Host "Compiler DLL present: $(Test-Path $compiler)"
& $bun --version
& $node --version

function Invoke-Repro($label, $executable, $backend, $prependDawnDirectory = $false) {
    $savedPath = $env:PATH
    $stdout = Join-Path $env:TEMP "dawn-repro-$([guid]::NewGuid()).out"
    $stderr = Join-Path $env:TEMP "dawn-repro-$([guid]::NewGuid()).err"
    try {
        if ($prependDawnDirectory) {
            $env:PATH = "$dawnDirectory;$savedPath"
        }
        $process = Start-Process -FilePath $executable `
            -ArgumentList @($repro, $backend) `
            -WorkingDirectory $root -NoNewWindow -Wait -PassThru `
            -RedirectStandardOutput $stdout -RedirectStandardError $stderr
        Write-Host "`n=== $label (exit $($process.ExitCode)) ==="
        if (Test-Path $stdout) { Get-Content $stdout }
        if (Test-Path $stderr) { Get-Content $stderr }
    } catch {
        Write-Host "`n=== $label (launcher error) ==="
        Write-Host $_
    } finally {
        $env:PATH = $savedPath
        Remove-Item $stdout, $stderr -Force -ErrorAction SilentlyContinue
    }
}

Invoke-Repro "Bun 1.4.3 create(['adapter=Microsoft']), normal DLL search" $bun "adapter=Microsoft"
Invoke-Repro "Node 26 create(['adapter=Microsoft']) control" $node "adapter=Microsoft"
Invoke-Repro "Bun 1.4.3 default backend, normal DLL search" $bun "default"
Invoke-Repro "Node 26 default backend, normal DLL search" $node "default"
foreach ($backend in @("d3d12", "d3d11", "vulkan", "null")) {
    Invoke-Repro "Bun backend=$backend, normal DLL search" $bun $backend
}
Invoke-Repro "Bun default backend, Dawn DLL directory prepended to PATH" $bun "default" $true

$bunDirectory = Split-Path $bun
$destination = Join-Path $bunDirectory "d3dcompiler_47.dll"
$backup = Join-Path $env:TEMP "dawn-d3dcompiler-backup-$([guid]::NewGuid()).dll"
$hadCompiler = Test-Path $destination
try {
    if ($hadCompiler) { Copy-Item $destination $backup -Force }
    Copy-Item $compiler $destination -Force
    Write-Host "Copied d3dcompiler_47.dll to $destination"
    Invoke-Repro "Bun default backend, d3dcompiler_47.dll beside bun.exe" $bun "default"
} finally {
    if ($hadCompiler) {
        Copy-Item $backup $destination -Force
        Remove-Item $backup -Force -ErrorAction SilentlyContinue
    } else {
        Remove-Item $destination -Force -ErrorAction SilentlyContinue
    }
}
