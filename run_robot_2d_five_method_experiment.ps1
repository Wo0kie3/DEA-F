param(
    [string]$Target = "Robot_18",
    [ValidateSet("real", "fictive", "mixed")]
    [string]$Mode = "fictive",
    [int]$Stages = 3,
    [int]$TargetBestRank = 5,
    [double]$TargetBestEfficiency = 0.92,
    [double]$TargetRankWidth = 9.0,
    [string]$Dimensions = "i1,i2",
    [double]$PctAbove = 5.0,
    [double]$StepPct = 25.0,
    [int]$MinPointsPerDimension = 5,
    [int]$MaxCandidates = 100,
    [int]$PointsPerStage = 15,
    [int]$MaxPaths = 1000,
    [string]$FactorWeights = "i1=0.6,i2=0.4",
    [string]$StageWeights = "0.2,0.3,0.5",
    [string]$WidthStateWeights = "0.1,0.2,0.3,0.4",
    [double]$MaxStageEffort = 0.30,
    [string]$MaxFactorChanges = "i1=0.40,i2=0.40",
    [switch]$RefineFictiveCandidates,
    [int]$RefineIterations = 3,
    [int]$RefineMaxSeeds = 3,
    [int]$LocalSearchSamples = 25,
    [double]$LocalSearchStepMultiplier = 1.5,
    [int]$LocalSearchRandomState = 42,
    [int]$GlobalSearchSamples = 100,
    [int]$GlobalSearchRandomState = 142,
    [string]$PythonExecutable = "python",
    [string]$OutputDir = "",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$repoRoot = $PSScriptRoot
$inputCsv = Join-Path $repoRoot "input\robot_technology_table1.csv"
$javaEntry = Join-Path $repoRoot "java"
$runStamp = Get-Date -Format "yyyyMMdd_HHmmss"
$experimentDir = if ($OutputDir) {
    [System.IO.Path]::GetFullPath($OutputDir)
} else {
    Join-Path $repoRoot "output\robot_2d_five_methods_$runStamp"
}

$dimensionList = @($Dimensions.Split(",") | ForEach-Object { $_.Trim() })
if ($dimensionList.Count -ne 2 -or @($dimensionList | Where-Object { $_ -notmatch "^i\d+$" }).Count -ne 0) {
    throw "Dimensions must contain exactly two inputs, for example: i1,i2"
}
if ($Stages -ne 3 -and ($StageWeights -or $WidthStateWeights)) {
    throw "The default publication weights have three transition and four state values; override them for another stage count."
}

New-Item -ItemType Directory -Path $experimentDir -Force | Out-Null

$gitCommit = (& git -C $repoRoot rev-parse HEAD).Trim()
$experimentParams = [ordered]@{
    input = $inputCsv
    input_semantics = "i1 Cost and i2 Repeatability minimized; o1 Load capacity and o2 Velocity fixed"
    target = $Target
    methods = "hasse_path,necessary_layer_path,best_rank_path,best_efficiency_path,robustness_width_path"
    output_dir = $experimentDir
    mode = $Mode
    stages = $Stages
    target_best_rank = $TargetBestRank
    target_best_efficiency = $TargetBestEfficiency
    target_rank_width = $TargetRankWidth
    width_kind = "rank"
    dimensions = $Dimensions
    factor_weights = $FactorWeights
    stage_weights = $StageWeights
    width_state_weights = $WidthStateWeights
    max_stage_effort = $MaxStageEffort
    max_factor_changes = $MaxFactorChanges
    pct_above = $PctAbove
    step_pct = $StepPct
    min_points_per_dimension = $MinPointsPerDimension
    max_candidates = $MaxCandidates
    points_per_stage = $PointsPerStage
    points_per_stage_semantics = "limit per actual predecessor transition"
    normalization_ranges = "observed ranges in the complete robot reference input"
    cdir_scope = "evaluated final-stage pool for methods 3-5; complete path endpoints for methods 1-2"
    max_paths = $MaxPaths
    refine_fictive_candidates = [bool]$RefineFictiveCandidates
    refine_iterations = $RefineIterations
    refine_max_seeds = $RefineMaxSeeds
    local_search_samples = $LocalSearchSamples
    local_search_step_multiplier = $LocalSearchStepMultiplier
    local_search_random_state = $LocalSearchRandomState
    global_search_samples_best_efficiency = $GlobalSearchSamples
    global_search_random_state = $GlobalSearchRandomState
    git_commit = $gitCommit
    git_worktree = "branch may contain uncommitted experiment fixes; see git_status.txt"
    dry_run = [bool]$DryRun
    started_at = (Get-Date).ToString("o")
}

$experimentParams.GetEnumerator() |
    ForEach-Object { [PSCustomObject]@{ parameter = $_.Key; value = [string]$_.Value } } |
    Export-Csv -Path (Join-Path $experimentDir "experiment_params.csv") -NoTypeInformation -Encoding UTF8
& git -C $repoRoot status --short | Out-File (Join-Path $experimentDir "git_status.txt") -Encoding utf8

function Format-Invariant {
    param([double]$Value)
    return $Value.ToString([System.Globalization.CultureInfo]::InvariantCulture)
}

$commonArgs = @(
    "--input", $inputCsv,
    "--target", $Target,
    "--output-dir", $experimentDir,
    "--java-entry", $javaEntry,
    "--mode", $Mode,
    "--columns", $Dimensions,
    "--pct-above", (Format-Invariant $PctAbove),
    "--step-pct", (Format-Invariant $StepPct),
    "--min-points-per-dim", $MinPointsPerDimension,
    "--max-candidates", $MaxCandidates,
    "--points-per-stage", $PointsPerStage,
    "--max-paths", $MaxPaths,
    "--factor-weights", $FactorWeights,
    "--max-stage-effort", (Format-Invariant $MaxStageEffort),
    "--max-factor-changes", $MaxFactorChanges
)

function Invoke-PathPipeline {
    param(
        [string]$ScriptName,
        [string[]]$MethodArgs = @()
    )
    $scriptPath = Join-Path $repoRoot "python\$ScriptName"
    $displayArgs = @($scriptPath) + $commonArgs + $MethodArgs
    Write-Host "Running: $PythonExecutable $($displayArgs -join ' ')"
    if (-not $DryRun) {
        & $PythonExecutable $scriptPath @commonArgs @MethodArgs
        if ($LASTEXITCODE -ne 0) {
            throw "Pipeline failed: $ScriptName"
        }
    }
}

$refinementArgs = @()
if ($RefineFictiveCandidates) {
    $refinementArgs = @(
        "--refine-fictive-candidates",
        "--refine-iterations", $RefineIterations,
        "--refine-max-seeds", $RefineMaxSeeds,
        "--local-search-samples", $LocalSearchSamples,
        "--local-search-step-multiplier", (Format-Invariant $LocalSearchStepMultiplier),
        "--local-search-random-state", $LocalSearchRandomState,
        "--local-search-sampling", "stratified"
    )
}

Write-Host "Robot 2D five-method experiment"
Write-Host "Target: $Target; modified inputs: $Dimensions; fixed outputs: o1,o2"
Write-Host "Output: $experimentDir"

if (-not $DryRun) {
    & $PythonExecutable (Join-Path $repoRoot "python\generate_dea_input_snapshot.py") `
        --input $inputCsv `
        --output-dir (Join-Path $experimentDir "dea_input_snapshot") `
        --java-entry $javaEntry
    if ($LASTEXITCODE -ne 0) {
        throw "DEA input snapshot failed."
    }
}

Invoke-PathPipeline -ScriptName "1_hasse_path_pipeline.py"
Invoke-PathPipeline -ScriptName "2_necessary_layer_path_pipeline.py"
Invoke-PathPipeline -ScriptName "3_best_rank_path_pipeline.py" -MethodArgs @(
    "--target-best-rank", $TargetBestRank,
    "--stages", $Stages,
    "--stage-weights", $StageWeights,
    "--width-state-weights", $WidthStateWeights
) + $refinementArgs
Invoke-PathPipeline -ScriptName "4_best_efficiency_path_pipeline.py" -MethodArgs @(
    "--target-best-efficiency", (Format-Invariant $TargetBestEfficiency),
    "--stages", $Stages,
    "--stage-weights", $StageWeights,
    "--width-state-weights", $WidthStateWeights,
    "--global-search-samples", $GlobalSearchSamples,
    "--global-search-random-state", $GlobalSearchRandomState
) + $refinementArgs
Invoke-PathPipeline -ScriptName "5_robustness_width_path_pipeline.py" -MethodArgs @(
    "--target-width", (Format-Invariant $TargetRankWidth),
    "--width-kind", "rank",
    "--stages", $Stages,
    "--stage-weights", $StageWeights,
    "--width-state-weights", $WidthStateWeights
) + $refinementArgs

if (-not $DryRun) {
    & $PythonExecutable (Join-Path $repoRoot "python\collect_path_metrics.py") `
        --experiment-dir $experimentDir
    if ($LASTEXITCODE -ne 0) {
        throw "Path metrics collection failed."
    }
    & $PythonExecutable (Join-Path $repoRoot "python\validate_controlled_experiment.py") `
        --root $experimentDir `
        --fixed-columns "o1,o2" `
        --max-paths $MaxPaths `
        --max-stage-effort (Format-Invariant $MaxStageEffort)
    if ($LASTEXITCODE -ne 0) {
        throw "Controlled experiment validation failed."
    }
    & $PythonExecutable (Join-Path $repoRoot "python\generate_five_method_robot_report.py") `
        --root $experimentDir `
        --input $inputCsv `
        --target $Target `
        --output (Join-Path $experimentDir "robot_18_five_method_report.html") `
        --plotly-mode embedded
    if ($LASTEXITCODE -ne 0) {
        throw "Five-method HTML report failed."
    }
    @(
        [PSCustomObject]@{ parameter = "completed_at"; value = (Get-Date).ToString("o") },
        [PSCustomObject]@{ parameter = "status"; value = "success" }
    ) | Export-Csv -Path (Join-Path $experimentDir "experiment_params.csv") -NoTypeInformation -Encoding UTF8 -Append
}

Write-Host "Experiment directory: $experimentDir"
if ($DryRun) {
    Write-Host "Dry run completed."
} else {
    Write-Host "Combined metrics: $(Join-Path $experimentDir 'all_path_metrics.csv')"
    Write-Host "Method summary: $(Join-Path $experimentDir 'method_summary.csv')"
    Write-Host "HTML report: $(Join-Path $experimentDir 'robot_18_five_method_report.html')"
}
