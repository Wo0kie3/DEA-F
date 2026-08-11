param(
    [string]$Target = "DMU_01",
    [ValidateSet("real", "fictive", "mixed")]
    [string]$Mode = "mixed",
    [int]$Stages = 3,
    [int]$TargetBestRank = 1,
    [double]$TargetBestEfficiency = 1.0,
    [double]$TargetScoreWidth = 0.25,
    [string]$Dimensions = "i1,i2,o1",
    [double]$PctAbove = 10.0,
    [double]$StepPct = 25.0,
    [int]$MinPointsPerDimension = 5,
    [int]$MaxCandidates = 200,
    [int]$PointsPerStage = 10,
    [int]$MaxPaths = 100,
    [switch]$RefineFictiveCandidates,
    [int]$RefineIterations = 8,
    [int]$RefineMaxSeeds = 20,
    [int]$LocalSearchSamples = 100,
    [double]$LocalSearchStepMultiplier = 1.0,
    [int]$LocalSearchRandomState = 42,
    [string]$PythonExecutable = "python",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$repoRoot = $PSScriptRoot
$inputCsv = Join-Path $repoRoot "input\EDU.csv"
$javaEntry = Join-Path $repoRoot "java"
$runStamp = Get-Date -Format "yyyyMMdd_HHmmss"
$experimentDir = Join-Path $repoRoot "output\edu_3d_experiment_$runStamp"
$columns = $Dimensions

$dimensionList = @($columns.Split(",") | ForEach-Object { $_.Trim() })
$inputDimensions = @($dimensionList | Where-Object { $_ -match "^i\d+$" })
$outputDimensions = @($dimensionList | Where-Object { $_ -match "^o\d+$" })
if ($dimensionList.Count -ne 3 -or $inputDimensions.Count -ne 2 -or $outputDimensions.Count -ne 1) {
    throw "Dimensions must contain exactly two inputs and one output, for example: i1,i2,o1"
}

New-Item -ItemType Directory -Path $experimentDir -Force | Out-Null

$experimentParams = [ordered]@{
    input = $inputCsv
    target = $Target
    output_dir = $experimentDir
    java_entry = $javaEntry
    mode = $Mode
    stages = $Stages
    target_best_rank = $TargetBestRank
    target_best_efficiency = $TargetBestEfficiency
    target_score_width = $TargetScoreWidth
    width_kind = "score"
    dimensions = $columns
    pct_above = $PctAbove
    step_pct = $StepPct
    min_points_per_dimension = $MinPointsPerDimension
    max_candidates = $MaxCandidates
    points_per_stage = $PointsPerStage
    points_per_stage_semantics = "limit per predecessor transition"
    transition_reference = "previous selected point"
    normalization_ranges = "observed ranges in reference input"
    max_paths = $MaxPaths
    refine_fictive_candidates = [bool]$RefineFictiveCandidates
    refine_iterations = $RefineIterations
    refine_max_seeds = $RefineMaxSeeds
    local_search_samples = $LocalSearchSamples
    local_search_step_multiplier = $LocalSearchStepMultiplier
    local_search_random_state = $LocalSearchRandomState
    python_executable = $PythonExecutable
    dry_run = [bool]$DryRun
    run_stamp = $runStamp
}

$experimentParams.GetEnumerator() |
    ForEach-Object { [PSCustomObject]@{ parameter = $_.Key; value = [string]$_.Value } } |
    Export-Csv -Path (Join-Path $experimentDir "experiment_params.csv") -NoTypeInformation -Encoding UTF8

$commonArgs = @(
    "--input", $inputCsv,
    "--target", $Target,
    "--output-dir", $experimentDir,
    "--java-entry", $javaEntry,
    "--mode", $Mode,
    "--columns", $columns,
    "--pct-above", $PctAbove,
    "--step-pct", $StepPct,
    "--min-points-per-dim", $MinPointsPerDimension,
    "--max-candidates", $MaxCandidates,
    "--points-per-stage", $PointsPerStage,
    "--max-paths", $MaxPaths
)

function Invoke-PathPipeline {
    param(
        [string]$ScriptName,
        [string[]]$MethodArgs = @()
    )

    $scriptPath = Join-Path $repoRoot "python\$ScriptName"
    $displayArgs = @($scriptPath) + $commonArgs + $MethodArgs
    Write-Host ""
    Write-Host "Running: $PythonExecutable $($displayArgs -join ' ')"

    if ($DryRun) {
        return
    }

    & $PythonExecutable $scriptPath @commonArgs @MethodArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Pipeline failed: $ScriptName"
    }
}

$refinementArgs = @()
if ($RefineFictiveCandidates) {
    $refinementArgs = @(
        "--refine-fictive-candidates",
        "--refine-iterations", $RefineIterations,
        "--refine-max-seeds", $RefineMaxSeeds,
        "--local-search-samples", $LocalSearchSamples,
        "--local-search-step-multiplier", $LocalSearchStepMultiplier,
        "--local-search-random-state", $LocalSearchRandomState
    )
}

Write-Host "EDU 3D experiment"
Write-Host "Target: $Target"
Write-Host "Modified dimensions: $columns"
Write-Host "Mode: $Mode"
Write-Host "Output: $experimentDir"

Invoke-PathPipeline -ScriptName "1_hasse_path_pipeline.py"
Invoke-PathPipeline -ScriptName "2_front_path_pipeline.py"

$rankArgs = @(
    "--target-best-rank", $TargetBestRank,
    "--stages", $Stages
) + $refinementArgs
$efficiencyArgs = @(
    "--target-best-efficiency", $TargetBestEfficiency,
    "--stages", $Stages
) + $refinementArgs
$widthArgs = @(
    "--target-width", $TargetScoreWidth,
    "--width-kind", "score",
    "--stages", $Stages
) + $refinementArgs

Invoke-PathPipeline -ScriptName "3_best_rank_path_pipeline.py" -MethodArgs $rankArgs
Invoke-PathPipeline -ScriptName "4_best_efficiency_path_pipeline.py" -MethodArgs $efficiencyArgs
Invoke-PathPipeline -ScriptName "5_robustness_width_path_pipeline.py" -MethodArgs $widthArgs

if (-not $DryRun) {
    $collector = Join-Path $repoRoot "python\collect_path_metrics.py"
    & $PythonExecutable $collector --experiment-dir $experimentDir
    if ($LASTEXITCODE -ne 0) {
        throw "Path metrics collection failed."
    }
}

Write-Host ""
Write-Host "Experiment directory: $experimentDir"
if ($DryRun) {
    Write-Host "Dry run completed. No pipelines were executed."
} else {
    Write-Host "Combined metrics: $(Join-Path $experimentDir 'all_path_metrics.csv')"
    Write-Host "Method summary: $(Join-Path $experimentDir 'method_summary.csv')"
}
