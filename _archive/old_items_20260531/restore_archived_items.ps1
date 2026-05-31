param(
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$archiveRoot = $PSScriptRoot
$repoRoot = Resolve-Path (Join-Path $archiveRoot "..\..")
$mapPath = Join-Path $archiveRoot "restore_map.csv"

if (-not (Test-Path -LiteralPath $mapPath)) {
    throw "Missing restore map: $mapPath"
}

$rows = Import-Csv -LiteralPath $mapPath
$conflicts = @()

foreach ($row in $rows) {
    $destination = Join-Path $repoRoot $row.original_path
    if (Test-Path -LiteralPath $destination) {
        $conflicts += $row.original_path
    }
}

if ($conflicts.Count -gt 0) {
    Write-Host "Restore stopped because these destination paths already exist:"
    foreach ($conflict in $conflicts) {
        Write-Host "  - $conflict"
    }
    Write-Host "Move or remove those paths first, then run this script again."
    exit 1
}

foreach ($row in $rows) {
    $source = Join-Path $repoRoot $row.archived_path
    $destination = Join-Path $repoRoot $row.original_path

    if (-not (Test-Path -LiteralPath $source)) {
        Write-Warning "Missing archived item: $($row.archived_path)"
        continue
    }

    $destinationParent = Split-Path -Parent $destination
    if ($destinationParent -and -not (Test-Path -LiteralPath $destinationParent)) {
        if ($DryRun) {
            Write-Host "Would create directory: $destinationParent"
        } else {
            New-Item -ItemType Directory -Force -Path $destinationParent | Out-Null
        }
    }

    if ($DryRun) {
        Write-Host "Would restore: $($row.archived_path) -> $($row.original_path)"
    } else {
        Move-Item -LiteralPath $source -Destination $destination
        Write-Host "Restored: $($row.original_path)"
    }
}
