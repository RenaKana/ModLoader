param(
    [switch]$Restore,
    [string]$NuGetPath = ''
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$source = Join-Path $projectRoot 'native\cpu-sensors'
$packages = Join-Path $projectRoot 'work\cpu-sensors-packages'
$output = Join-Path $projectRoot 'core\modloader\sensors'
$compiler = Join-Path $env:WINDIR 'Microsoft.NET\Framework64\v4.0.30319\csc.exe'

if ($Restore) {
    if (!$NuGetPath) { $NuGetPath = Join-Path $projectRoot 'work\cpu-sensors-tools\nuget.exe' }
    if (!(Test-Path -LiteralPath $NuGetPath)) { throw 'NuGet CLI missing. Pass -NuGetPath with the official nuget.exe path; see README.' }
    & $NuGetPath restore (Join-Path $source 'packages.config') -PackagesDirectory $packages -Source https://api.nuget.org/v3/index.json -NonInteractive
    if ($LASTEXITCODE -ne 0) { throw 'CPU sensor dependency restore failed' }
}

# Explicit runtime assets for net472/x64. System.Management uses the Framework
# assembly (the package's net462 asset is _._); never copy its netstandard stub.
$assets = @{
    'BlackSharp.Core' = 'lib\net472\BlackSharp.Core.dll'
    'DiskInfoToolkit' = 'lib\net472\DiskInfoToolkit.dll'
    'HidSharp' = 'lib\net35\HidSharp.dll'
    'LibreHardwareMonitorLib' = 'runtimes\win-x64\lib\net472\LibreHardwareMonitorLib.dll'
    'RAMSPDToolkit-NDD' = 'lib\net472\RAMSPDToolkit-NDD.dll'
    'System.Buffers' = 'lib\net462\System.Buffers.dll'
    'System.CodeDom' = 'lib\net462\System.CodeDom.dll'
    'System.Memory' = 'lib\net462\System.Memory.dll'
    'System.Numerics.Vectors' = 'lib\net462\System.Numerics.Vectors.dll'
    'System.Runtime.CompilerServices.Unsafe' = 'lib\net462\System.Runtime.CompilerServices.Unsafe.dll'
    'System.Security.AccessControl' = 'runtimes\win\lib\net461\System.Security.AccessControl.dll'
    'System.Security.Principal.Windows' = 'runtimes\win\lib\net461\System.Security.Principal.Windows.dll'
    'System.Threading.AccessControl' = 'lib\net462\System.Threading.AccessControl.dll'
}
[xml]$manifest = Get-Content -LiteralPath (Join-Path $source 'packages.config')
foreach ($package in $manifest.packages.package) {
    $directory = Join-Path $packages "$($package.id).$($package.version)"
    if (!(Test-Path -LiteralPath $directory)) { throw "Missing $($package.id). Run scripts/build-cpu-sensors.ps1 -Restore first." }
    if ($assets.ContainsKey($package.id) -and !(Test-Path -LiteralPath (Join-Path $directory $assets[$package.id]))) {
        throw "Missing runtime asset for $($package.id). Restore dependencies again."
    }
}
New-Item -ItemType Directory -Force $output | Out-Null
foreach ($package in $manifest.packages.package) {
    $directory = Join-Path $packages "$($package.id).$($package.version)"
    if ($assets.ContainsKey($package.id)) {
        Copy-Item -LiteralPath (Join-Path $directory $assets[$package.id]) -Destination $output -Force
    }
    $notices = Join-Path $output "licenses\$($package.id)"
    New-Item -ItemType Directory -Force $notices | Out-Null
    Get-ChildItem -LiteralPath $directory -File | Where-Object { $_.Name -match 'license|notice' } | Copy-Item -Destination $notices -Force
}
Copy-Item -LiteralPath (Join-Path $source 'licenses') -Destination $output -Recurse -Force
Copy-Item -LiteralPath (Join-Path $source 'THIRD-PARTY.md'), (Join-Path $source 'packages.config') -Destination $output -Force
& $compiler /nologo /target:winexe /platform:x64 /langversion:5 /optimize+ "/out:$output\ModLoader.CpuSensors.exe" "/reference:$output\LibreHardwareMonitorLib.dll" /reference:System.Web.Extensions.dll (Join-Path $source 'Program.cs')
if ($LASTEXITCODE -ne 0) { throw 'CPU sensor helper build failed' }
Write-Host "CPU sensor helper built: $output\ModLoader.CpuSensors.exe"
