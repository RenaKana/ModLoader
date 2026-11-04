$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$env:HF_HUB_OFFLINE = '1'
$env:HF_HUB_DISABLE_TELEMETRY = '1'
Push-Location $projectRoot
try {
    & (Join-Path $PSScriptRoot 'build-cpu-sensors.ps1')
    $sensorData = (Join-Path $projectRoot 'core\modloader\sensors') + ';modloader/sensors'
    # Send these same stdlib parsers to WSL; frozen modules do not retain source.
    $assetParser = (Join-Path $projectRoot 'core\modloader\assets.py') + ';modloader'
    $mtpParser = (Join-Path $projectRoot 'core\modloader\mtp.py') + ';modloader'
    $resourceParser = (Join-Path $projectRoot 'core\modloader\resource_metadata.py') + ';modloader'
    $inspectionParser = (Join-Path $projectRoot 'core\modloader\resource_inspection.py') + ';modloader'
    $gpuIdentity = (Join-Path $projectRoot 'core\modloader\gpu_identity.py') + ';modloader'
    $strataPolicy = (Join-Path $projectRoot 'core\modloader\strata_policy.py') + ';modloader'
    $prefetchBuilder = (Join-Path $projectRoot 'scripts\build-strata-prefetch.ps1') + ';scripts'
    $prefetchPatches = (Join-Path $projectRoot 'patches/strata-prefetch') + ';patches/strata-prefetch'
    $continuationPatches = (Join-Path $projectRoot 'patches/strata-token-continuation') + ';patches/strata-token-continuation'
    & .\.venv\Scripts\python.exe -m PyInstaller --noconfirm --clean --name modloader-core --paths core --distpath work/core-dist --workpath work/core-build --specpath work --add-data $sensorData --add-data $assetParser --add-data $mtpParser --add-data $resourceParser --add-data $inspectionParser --add-data $gpuIdentity --add-data $strataPolicy --add-data $prefetchBuilder --add-data $prefetchPatches --add-data $continuationPatches --collect-submodules modloader --collect-all huggingface_hub --collect-all modelscope_hub --collect-all hf_xet scripts/core_entry.py
    if ($LASTEXITCODE -ne 0) { throw 'Core packaging failed' }
} finally { Pop-Location }
