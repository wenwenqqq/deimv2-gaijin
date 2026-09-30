# Run from PowerShell after activating the deimv2 environment.
# Only this experiment's own checkpoint is used for automatic resume.
param(
    [string]$PythonBin = 'python'
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$experimentName = 'deimv2_hgnetv2_moe_v2_n_uavdt_HSFPN_dw_QUEST_SEA_decoder'
$configPath = Join-Path $projectRoot "configs/deimv2/$experimentName.yml"
$outputPath = Join-Path $projectRoot "outputs/$experimentName"
$lastCheckpoint = Join-Path $outputPath 'last.pth'
$trainingArgs = @(
    'train.py', '-c', $configPath,
    '--output-dir', $outputPath,
    '--use-amp', '--seed=0'
)

if (Test-Path -LiteralPath $lastCheckpoint -PathType Leaf) {
    $trainingArgs += @('-r', $lastCheckpoint)
    Write-Host "Resuming QUEST-SEA: $lastCheckpoint"
} else {
    Write-Host "Starting a new QUEST-SEA experiment: $outputPath"
}

Push-Location -LiteralPath $projectRoot
try {
    & $PythonBin @trainingArgs
    if ($LASTEXITCODE -ne 0) {
        throw "QUEST-SEA training exited with code $LASTEXITCODE."
    }
} finally {
    Pop-Location
}
