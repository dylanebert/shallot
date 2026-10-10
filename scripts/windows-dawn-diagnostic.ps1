$ErrorActionPreference = "Stop"

$bun = (Get-Command bun).Source
$node = (Get-Command node).Source
$root = $env:GITHUB_WORKSPACE
$repro = Join-Path $root "scripts/windows-dawn-repro.cjs"

Write-Host "Bun executable: $bun"
Write-Host "Node executable: $node"
Write-Host "PATH node matches:"
& where.exe node
if ($LASTEXITCODE -ne 0) { Write-Host "  none" }
& $bun --version
& $node --version

function Get-PathWithoutNode {
    $kept = [System.Collections.Generic.List[string]]::new()
    $removed = [System.Collections.Generic.List[string]]::new()
    foreach ($entry in $env:PATH -split ";") {
        $directory = if ($entry) { $entry.Trim('"') } else { (Get-Location).Path }
        if (Test-Path -LiteralPath (Join-Path $directory "node.exe")) {
            $removed.Add($directory)
        } else {
            $kept.Add($entry)
        }
    }

    Write-Host "Removed PATH directories containing node.exe: $($removed -join '; ')"
    $path = $kept -join ";"
    $savedPath = $env:PATH
    try {
        $env:PATH = $path
        $remaining = @(& where.exe node 2>$null)
        if ($LASTEXITCODE -eq 0 -or $remaining.Count -gt 0) {
            throw "node.exe remains discoverable after filtering PATH: $($remaining -join '; ')"
        }
        Write-Host "Verified where node returns no matches after PATH filtering."
    } finally {
        $env:PATH = $savedPath
    }
    return $path
}

function Invoke-Repro($label, $executable, $option, $path) {
    $savedPath = $env:PATH
    $stdout = Join-Path $env:TEMP "dawn-repro-$([guid]::NewGuid()).out"
    $stderr = Join-Path $env:TEMP "dawn-repro-$([guid]::NewGuid()).err"
    $exitCode = $null
    $launchError = $null
    try {
        $env:PATH = $path
        $process = Start-Process -FilePath $executable `
            -ArgumentList @($repro, $option) `
            -WorkingDirectory $root -NoNewWindow -Wait -PassThru `
            -RedirectStandardOutput $stdout -RedirectStandardError $stderr
        $exitCode = $process.ExitCode
    } catch {
        $launchError = $_.ToString()
    } finally {
        $env:PATH = $savedPath
    }

    $stdoutText = if (Test-Path $stdout) { [IO.File]::ReadAllText($stdout) } else { "" }
    $stderrText = if (Test-Path $stderr) { [IO.File]::ReadAllText($stderr) } else { "" }
    if ($launchError) { $stderrText = "$stderrText`n$launchError" }
    Write-Host "`n=== $label (exit $exitCode) ==="
    if ($stdoutText) { Write-Host $stdoutText.TrimEnd() }
    if ($stderrText) { Write-Host $stderrText.TrimEnd() }
    Remove-Item $stdout, $stderr -Force -ErrorAction SilentlyContinue

    return [PSCustomObject]@{
        Label = $label
        ExitCode = $exitCode
        Output = "$stdoutText`n$stderrText"
    }
}

$normalPath = $env:PATH
$nodeFreePath = Get-PathWithoutNode
$results = @(
    (Invoke-Repro "Bun 1.4.3, normal PATH, adapter=Microsoft" $bun "adapter=Microsoft" $normalPath)
    (Invoke-Repro "Node 26 control, normal PATH, adapter=Microsoft" $node "adapter=Microsoft" $normalPath)
    (Invoke-Repro "Bun 1.4.3, every node.exe PATH directory removed" $bun "adapter=Microsoft" $nodeFreePath)
)

$aliasDirectory = Join-Path $env:TEMP "dawn-bun-as-node-$([guid]::NewGuid())"
$alias = Join-Path $aliasDirectory "node.exe"
try {
    New-Item -ItemType Directory -Path $aliasDirectory | Out-Null
    Copy-Item $bun $alias
    Write-Host "`nCopied Bun to $alias"
    $results += Invoke-Repro "Bun copied to node.exe, node-free PATH, adapter=Microsoft" $alias "adapter=Microsoft" $nodeFreePath
} finally {
    Remove-Item $aliasDirectory -Recurse -Force -ErrorAction SilentlyContinue
}

Write-Host "`n=== Result table ==="
$results | Select-Object Label, ExitCode | Format-Table -AutoSize

$withoutNode = $results[2]
$asNode = $results[3]
$missingNodeFailure = $withoutNode.ExitCode -ne 0 -and $withoutNode.Output -match "node\.exe"
$renamedBunPass = $asNode.ExitCode -eq 0 -and $asNode.Output -match "first submission completed"
if (-not $missingNodeFailure) {
    throw "No-node-PATH case did not fail with an error naming node.exe."
}
if (-not $renamedBunPass) {
    throw "Bun copied to node.exe did not complete its first submission."
}
