param(
    [ValidateSet('smoke', 'full')]
    [string]$Preset = 'smoke',

    [string]$PythonExe = 'python',

    [ValidateSet('auto', 'cuda', 'cpu')]
    [string]$Device = 'auto',

    [string[]]$Datasets = @('fMRI', 'sim2', 'sim3', 'sim4'),

    [int]$Epochs = -1,

    [int]$SubjectLimit = -2,

    [int]$TimeLimit = -2,

    [int]$LogInterval = -1,

    [switch]$UsePretrain,

    [switch]$IncludeAblations,

    [switch]$IncludeDirectionOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent $PSScriptRoot
$resultsRoot = Join-Path $repoRoot 'results'
$batchTag = Get-Date -Format 'yyyyMMdd_HHmmss'
$batchDir = Join-Path $resultsRoot "prior_compare_$batchTag"
New-Item -ItemType Directory -Path $batchDir -Force | Out-Null

$datasetMap = @{
    'fMRI' = @{
        Name = 'fMRI'
        Csv = Join-Path $repoRoot 'fMRI_dataset\fMRI.csv'
        Gt = Join-Path $repoRoot 'fMRI_dataset\h1.txt'
    }
    'sim2' = @{
        Name = 'sim2'
        Csv = Join-Path $repoRoot 'fMRI_dataset\sim2.csv'
        Gt = Join-Path $repoRoot 'fMRI_dataset\h2.txt'
    }
    'sim3' = @{
        Name = 'sim3'
        Csv = Join-Path $repoRoot 'fMRI_dataset\sim3.csv'
        Gt = Join-Path $repoRoot 'fMRI_dataset\h3.txt'
    }
    'sim4' = @{
        Name = 'sim4'
        Csv = Join-Path $repoRoot 'fMRI_dataset\sim4.csv'
        Gt = Join-Path $repoRoot 'fMRI_dataset\h4.txt'
    }
}

$variants = @(
    @{
        Name = 'baseline_patel'
        SupportPrior = 'patel'
        DirectionPrior = 'patel'
        DirectionInit = 'patel_tau'
    }
)

if ($IncludeAblations) {
    $variants += @(
        @{
            Name = 'soft_support_patel_tau'
            SupportPrior = 'soft_patel'
            DirectionPrior = 'patel'
            DirectionInit = 'patel_tau'
        },
        @{
            Name = 'soft_support_lag_gain_zero_init'
            SupportPrior = 'soft_patel'
            DirectionPrior = 'lag_gain'
            DirectionInit = 'zeros'
        }
    )
}

if ($IncludeDirectionOnly) {
    $variants += @(
        @{
            Name = 'patel_support_lag_gain_zero_init'
            SupportPrior = 'patel'
            DirectionPrior = 'lag_gain'
            DirectionInit = 'zeros'
        }
    )
}

$variants += @(
    @{
        Name = 'soft_support_lag_gain'
        SupportPrior = 'soft_patel'
        DirectionPrior = 'lag_gain'
        DirectionInit = 'lag_gain'
    }
)

if (-not $IncludeAblations -and -not $IncludeDirectionOnly) {
    $variants = @(
        $variants | Where-Object { $_.Name -in @('baseline_patel', 'soft_support_lag_gain') }
    )
}

if ($IncludeDirectionOnly -and -not $IncludeAblations) {
    $variants = @(
        $variants | Where-Object { $_.Name -in @('baseline_patel', 'patel_support_lag_gain_zero_init') }
    )
}

function Get-ResolvedDevice {
    param(
        [string]$Requested,
        [string]$PythonExe,
        [string]$RepoRoot
    )

    if ($Requested -ne 'auto') {
        return $Requested
    }

    $probe = @'
import torch
print("cuda" if torch.cuda.is_available() else "cpu")
'@
    $resolved = $probe | & $PythonExe - 2>$null
    if ($LASTEXITCODE -ne 0) {
        return 'cpu'
    }
    $resolved = ($resolved | Select-Object -Last 1).Trim()
    if ([string]::IsNullOrWhiteSpace($resolved)) {
        return 'cpu'
    }
    return $resolved
}

function Get-NewRunDirectory {
    param(
        [string[]]$ExistingRunDirs,
        [string]$ResultsRoot
    )

    $currentRunDirs = @(Get-ChildItem -Path $ResultsRoot -Directory -Filter 'run_*' | Select-Object -ExpandProperty FullName)
    $newRunDirs = @($currentRunDirs | Where-Object { $_ -notin $ExistingRunDirs })
    if ($newRunDirs.Count -gt 0) {
        return Get-Item ($newRunDirs | Sort-Object | Select-Object -Last 1)
    }

    $latest = Get-ChildItem -Path $ResultsRoot -Directory -Filter 'run_*' |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
    return $latest
}

function Get-RunSummary {
    param(
        [string]$BatchTag,
        [hashtable]$DatasetInfo,
        [hashtable]$VariantInfo,
        [string]$RunDir,
        [double]$RuntimeSeconds,
        [string]$ResolvedDevice,
        [int]$EffectiveEpochs,
        [int]$EffectiveSubjectLimit,
        [int]$EffectiveTimeLimit,
        [bool]$SkipPretrain,
        [bool]$Succeeded,
        [string]$ErrorMessage
    )

    $qualityPath = Join-Path $RunDir 'quality_history.csv'
    $auditPath = Join-Path $RunDir 'selector_audit_summary.csv'

    $qualityRows = @()
    if (Test-Path $qualityPath) {
        $qualityRows = @(Import-Csv $qualityPath)
    }

    $auditRow = $null
    if (Test-Path $auditPath) {
        $auditRow = Import-Csv $auditPath | Select-Object -First 1
    }

    $bestPrimary = $null
    $finalPrimary = $null
    if ($qualityRows.Count -gt 0) {
        $bestPrimary = $qualityRows |
            Sort-Object { [double]$_.score_primary_total } -Descending |
            Select-Object -First 1
        $finalPrimary = $qualityRows[-1]
    }

    [pscustomobject]@{
        batch_tag = $BatchTag
        dataset = $DatasetInfo.Name
        csv_path = $DatasetInfo.Csv
        gt_path = $DatasetInfo.Gt
        variant = $VariantInfo.Name
        support_prior_algorithm = $VariantInfo.SupportPrior
        direction_prior_algorithm = $VariantInfo.DirectionPrior
        direction_init_mode = $VariantInfo.DirectionInit
        run_dir = $RunDir
        runtime_sec = [math]::Round($RuntimeSeconds, 2)
        device = $ResolvedDevice
        epochs = $EffectiveEpochs
        subject_limit = $EffectiveSubjectLimit
        time_limit = $EffectiveTimeLimit
        skip_pretrain = [int]$SkipPretrain
        status = if ($Succeeded) { 'ok' } else { 'failed' }
        error_message = $ErrorMessage
        best_proxy_epoch = if ($bestPrimary) { [int]$bestPrimary.epoch } else { $null }
        best_proxy_score_primary_total = if ($bestPrimary) { [double]$bestPrimary.score_primary_total } else { $null }
        best_proxy_skeleton_overlap = if ($bestPrimary) { [double]$bestPrimary.skeleton_overlap } else { $null }
        best_proxy_agreement = if ($bestPrimary) { [double]$bestPrimary.agreement } else { $null }
        best_proxy_dir_margin = if ($bestPrimary) { [double]$bestPrimary.dir_margin } else { $null }
        final_epoch = if ($finalPrimary) { [int]$finalPrimary.epoch } else { $null }
        final_score_primary_total = if ($finalPrimary) { [double]$finalPrimary.score_primary_total } else { $null }
        final_skeleton_overlap = if ($finalPrimary) { [double]$finalPrimary.skeleton_overlap } else { $null }
        final_agreement = if ($finalPrimary) { [double]$finalPrimary.agreement } else { $null }
        final_dir_margin = if ($finalPrimary) { [double]$finalPrimary.dir_margin } else { $null }
        best_gt_epoch = if ($auditRow) { [int]$auditRow.selector_audit_best_gt_epoch } else { $null }
        best_gt_primary_strict_f1 = if ($auditRow) { [double]$auditRow.selector_audit_best_gt_primary_strict_f1 } else { $null }
        exported_epoch = if ($auditRow) { [int]$auditRow.selector_audit_exported_epoch } else { $null }
        exported_primary_strict_f1 = if ($auditRow) { [double]$auditRow.selector_audit_exported_primary_strict_f1 } else { $null }
        final_primary_strict_f1 = if ($auditRow) { [double]$auditRow.selector_audit_final_primary_strict_f1 } else { $null }
        exported_vs_best_gt_gap_primary_strict_f1 = if ($auditRow) { [double]$auditRow.selector_audit_exported_vs_best_gt_gap_primary_strict_f1 } else { $null }
        exported_failure_mode = if ($auditRow) { [string]$auditRow.selector_audit_exported_failure_mode } else { $null }
        final_failure_mode = if ($auditRow) { [string]$auditRow.selector_audit_final_failure_mode } else { $null }
    }
}

function New-ComparisonRows {
    param(
        [object[]]$SummaryRows
    )

    $rows = @()
    $datasetNames = $SummaryRows | Select-Object -ExpandProperty dataset -Unique
    foreach ($datasetName in $datasetNames) {
        $baseline = $SummaryRows | Where-Object {
            $_.dataset -eq $datasetName -and $_.variant -eq 'baseline_patel'
        } | Select-Object -First 1

        if (-not $baseline) {
            continue
        }

        $candidates = $SummaryRows | Where-Object {
            $_.dataset -eq $datasetName -and $_.variant -ne 'baseline_patel'
        }

        foreach ($candidate in $candidates) {
            $rows += [pscustomobject]@{
                dataset = $datasetName
                variant = $candidate.variant
                baseline_status = $baseline.status
                candidate_status = $candidate.status
                baseline_run_dir = $baseline.run_dir
                candidate_run_dir = $candidate.run_dir
                baseline_best_proxy_score_primary_total = $baseline.best_proxy_score_primary_total
                candidate_best_proxy_score_primary_total = $candidate.best_proxy_score_primary_total
                delta_best_proxy_score_primary_total = (
                    [double]$candidate.best_proxy_score_primary_total - [double]$baseline.best_proxy_score_primary_total
                )
                baseline_exported_primary_strict_f1 = $baseline.exported_primary_strict_f1
                candidate_exported_primary_strict_f1 = $candidate.exported_primary_strict_f1
                delta_exported_primary_strict_f1 = (
                    [double]$candidate.exported_primary_strict_f1 - [double]$baseline.exported_primary_strict_f1
                )
                baseline_best_gt_primary_strict_f1 = $baseline.best_gt_primary_strict_f1
                candidate_best_gt_primary_strict_f1 = $candidate.best_gt_primary_strict_f1
                delta_best_gt_primary_strict_f1 = (
                    [double]$candidate.best_gt_primary_strict_f1 - [double]$baseline.best_gt_primary_strict_f1
                )
                baseline_exported_vs_best_gt_gap_primary_strict_f1 = $baseline.exported_vs_best_gt_gap_primary_strict_f1
                candidate_exported_vs_best_gt_gap_primary_strict_f1 = $candidate.exported_vs_best_gt_gap_primary_strict_f1
                delta_exported_gap_primary_strict_f1 = (
                    [double]$candidate.exported_vs_best_gt_gap_primary_strict_f1 -
                    [double]$baseline.exported_vs_best_gt_gap_primary_strict_f1
                )
            }
        }
    }
    return $rows
}

$resolvedDevice = Get-ResolvedDevice -Requested $Device -PythonExe $PythonExe -RepoRoot $repoRoot

switch ($Preset) {
    'smoke' {
        $effectiveEpochs = if ($Epochs -gt 0) { $Epochs } else { 6 }
        $effectiveSubjectLimit = if ($SubjectLimit -ne -2) { $SubjectLimit } else { 8 }
        $effectiveTimeLimit = if ($TimeLimit -ne -2) { $TimeLimit } else { 100 }
        $effectiveLogInterval = if ($LogInterval -gt 0) { $LogInterval } else { 1 }
    }
    'full' {
        $effectiveEpochs = if ($Epochs -gt 0) { $Epochs } else { 30 }
        $effectiveSubjectLimit = if ($SubjectLimit -ne -2) { $SubjectLimit } else { -1 }
        $effectiveTimeLimit = if ($TimeLimit -ne -2) { $TimeLimit } else { -1 }
        $effectiveLogInterval = if ($LogInterval -gt 0) { $LogInterval } else { 5 }
    }
    default {
        throw "Unsupported preset: $Preset"
    }
}

$skipPretrain = -not $UsePretrain.IsPresent
$summaryRows = @()

Write-Host "Batch output: $batchDir"
Write-Host "Preset: $Preset | device=$resolvedDevice | epochs=$effectiveEpochs | skip_pretrain=$skipPretrain"
Write-Host "Datasets: $($Datasets -join ', ')"
Write-Host "Variants: $((@($variants | ForEach-Object { $_.Name })) -join ', ')"

foreach ($datasetKey in $Datasets) {
    if (-not $datasetMap.ContainsKey($datasetKey)) {
        throw "Unknown dataset key: $datasetKey"
    }
    $datasetInfo = $datasetMap[$datasetKey]
    foreach ($variant in $variants) {
        $existingRunDirs = @(Get-ChildItem -Path $resultsRoot -Directory -Filter 'run_*' | Select-Object -ExpandProperty FullName)
        $logPath = Join-Path $batchDir ("{0}_{1}.log" -f $datasetInfo.Name, $variant.Name)

        $commandArgs = @(
            'GraphExp\main_structure_learning.py'
            '--csv_path', $datasetInfo.Csv
            '--selector_audit_gt_path', $datasetInfo.Gt
            '--device', $resolvedDevice
            '--epochs', $effectiveEpochs
            '--log_interval', $effectiveLogInterval
            '--support_prior_algorithm', $variant.SupportPrior
            '--direction_prior_algorithm', $variant.DirectionPrior
            '--direction_init_mode', $variant.DirectionInit
        )

        if ($effectiveSubjectLimit -gt 0) {
            $commandArgs += @('--subject_limit', $effectiveSubjectLimit)
        }
        if ($effectiveTimeLimit -gt 0) {
            $commandArgs += @('--time_limit', $effectiveTimeLimit)
        }
        if ($skipPretrain) {
            $commandArgs += '--skip_pretrain'
        }

        Write-Host ""
        Write-Host ("=== Running {0} | {1} ===" -f $datasetInfo.Name, $variant.Name)
        Write-Host ($PythonExe + ' ' + ($commandArgs -join ' '))

        $stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
        $success = $true
        $errorMessage = ''
        try {
            & $PythonExe @commandArgs 2>&1 | Tee-Object -FilePath $logPath
            if ($LASTEXITCODE -ne 0) {
                throw "Training exited with code $LASTEXITCODE"
            }
        } catch {
            $success = $false
            $errorMessage = $_.Exception.Message
            $_ | Out-String | Tee-Object -FilePath $logPath -Append | Out-Null
        }
        $stopwatch.Stop()

        $runDir = Get-NewRunDirectory -ExistingRunDirs $existingRunDirs -ResultsRoot $resultsRoot
        if (-not $runDir) {
            throw "Could not locate a result run directory after executing $($variant.Name) on $($datasetInfo.Name)"
        }

        $summaryRows += Get-RunSummary `
            -BatchTag $batchTag `
            -DatasetInfo $datasetInfo `
            -VariantInfo $variant `
            -RunDir $runDir.FullName `
            -RuntimeSeconds $stopwatch.Elapsed.TotalSeconds `
            -ResolvedDevice $resolvedDevice `
            -EffectiveEpochs $effectiveEpochs `
            -EffectiveSubjectLimit $effectiveSubjectLimit `
            -EffectiveTimeLimit $effectiveTimeLimit `
            -SkipPretrain $skipPretrain `
            -Succeeded $success `
            -ErrorMessage $errorMessage
    }
}

$summaryPath = Join-Path $batchDir 'summary.csv'
$summaryRows | Export-Csv -Path $summaryPath -NoTypeInformation -Encoding UTF8

$compareRows = New-ComparisonRows -SummaryRows $summaryRows
$comparePath = Join-Path $batchDir 'compare.csv'
$compareRows | Export-Csv -Path $comparePath -NoTypeInformation -Encoding UTF8

Write-Host ""
Write-Host "Saved summary: $summaryPath"
Write-Host "Saved compare: $comparePath"

$compareRows | Format-Table -AutoSize
