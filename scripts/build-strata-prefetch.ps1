[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$RuntimeRoot,
    [Parameter(Mandatory = $true)][string]$OutputDirectory,
    [Parameter(Mandatory = $true)][string]$WorkDirectory,
    [string]$ToolchainConfig,
    [string]$SourceDirectory,
    [string]$GgmlSourceDirectory,
    [switch]$PlanOnly
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version 2.0

function Read-JsonObject([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { throw "Missing file: $Path" }
    $value = Get-Content -LiteralPath $Path -Raw -Encoding UTF8 | ConvertFrom-Json
    if ($null -eq $value -or $value -isnot [pscustomobject]) { throw "Expected a JSON object: $Path" }
    return $value
}

function Field($Object, [string]$Name) {
    $property = $Object.PSObject.Properties[$Name]
    if ($null -ne $property) { return $property.Value }
    return $null
}

function Full-Path([string]$Path) {
    return [IO.Path]::GetFullPath($Path).TrimEnd([IO.Path]::DirectorySeparatorChar, [IO.Path]::AltDirectorySeparatorChar)
}

function Paths-Overlap([string]$Left, [string]$Right) {
    return ($Left.Equals($Right, [StringComparison]::OrdinalIgnoreCase) -or
        $Left.StartsWith($Right + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase) -or
        $Right.StartsWith($Left + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase))
}

function Assert-NoLinkedParent([string]$Path) {
    $current = $Path
    while ($current) {
        if (Test-Path -LiteralPath $current) {
            $item = Get-Item -LiteralPath $current -Force
            if (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
                throw "Refusing a linked build/output path: $current"
            }
        }
        $current = Split-Path -Parent $current
    }
}

function Run([string]$Program, [string[]]$Arguments, [string]$Directory, [switch]$Capture, [switch]$Quiet, [string]$Label) {
    $description = $Label
    if (-not $description) { $description = $Program + ' ' + ($Arguments -join ' ') }
    if (-not $Quiet) { [Console]::Out.WriteLine('> ' + $description) }
    Push-Location -LiteralPath $Directory
    try {
        # Windows PowerShell represents native stderr as ErrorRecord; the exit code is authoritative.
        $oldPreference = $ErrorActionPreference
        $ErrorActionPreference = 'Continue'
        try {
            $lines = @(& $Program @Arguments 2>&1 | ForEach-Object {
                $line = [string]$_
                if (-not $Quiet) { [Console]::Out.WriteLine($line) }
                $line
            })
            $code = $LASTEXITCODE
        } finally { $ErrorActionPreference = $oldPreference }
        if ($code -ne 0) { throw "Command failed (exit $code): $description" }
        if ($Capture) { return ($lines -join "`n") }
    } finally { Pop-Location }
}

function Configured-Path($Config, [string]$Key, [switch]$Directory) {
    $value = Field $Config $Key
    if ($null -eq $value) { return $null }
    if ($value -isnot [string] -or -not [IO.Path]::IsPathRooted($value)) {
        throw "ToolchainConfig.$Key must be an absolute path"
    }
    $kind = 'Leaf'
    if ($Directory) { $kind = 'Container' }
    if (-not (Test-Path -LiteralPath $value -PathType $kind)) { throw "ToolchainConfig.$Key not found: $value" }
    return (Full-Path $value)
}

function Find-Tool([string]$Name, [string]$Configured, [string[]]$Candidates = @()) {
    if ($Configured) { return $Configured }
    $command = Get-Command $Name -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($command) { return $command.Source }
    foreach ($candidate in $Candidates) {
        if ($candidate -and (Test-Path -LiteralPath $candidate -PathType Leaf)) { return $candidate }
    }
    throw "Missing $Name; install the build tools or set ToolchainConfig"
}

function Download([string]$Url, [string]$Destination) {
    $curl = Find-Tool 'curl.exe' '' @((Join-Path $env:SystemRoot 'System32/curl.exe'))
    $arguments = @('--disable', '--fail', '--location', '--connect-timeout', '15', '--max-time', '180',
        '--silent', '--show-error', '--user-agent', 'ModLoader-Strata-Prefetch', '--output', $Destination)
    $proxy = $env:HTTPS_PROXY
    if (-not $proxy) { $proxy = $env:HTTP_PROXY }
    if ($proxy) { $arguments += @('--proxy', $proxy) }
    $arguments += @('--', $Url)
    # Do not include proxy credentials in command/error logs.
    Run $curl $arguments $work -Label "Downloading $Url"
}

function Assert-CudaMajor([string]$EngineCuda, [string]$NvccVersion) {
    $expected = [regex]::Match($EngineCuda, '^(?<major>[0-9]+)(?:\.|$)')
    $actual = [regex]::Match($NvccVersion, '(?i)\brelease\s+(?<major>[0-9]+)\.')
    if (-not $expected.Success) { throw 'BUILD.json cuda must identify the engine CUDA toolkit version' }
    if (-not $actual.Success) { throw 'Cannot identify the CUDA toolkit major from nvcc --version' }
    if ([int]$expected.Groups['major'].Value -ne [int]$actual.Groups['major'].Value) {
        throw "CUDA toolkit major mismatch: engine requires $EngineCuda; nvcc is $($actual.Groups['major'].Value). Set ToolchainConfig.cuda_root"
    }
}

function Import-MsvcEnvironment([string]$Vcvars, [string]$TemporaryDirectory) {
    if ($Vcvars.Contains('"') -or $Vcvars.Contains('%')) { throw 'vcvars path cannot contain quotes or percent signs' }
    $vcArguments = ''
    if ([IO.Path]::GetFileName($Vcvars) -ieq 'vcvarsall.bat') { $vcArguments = ' x64' }
    [Console]::Out.WriteLine("Loading MSVC environment: $Vcvars")
    $bootstrap = Join-Path $TemporaryDirectory 'load-msvc.cmd'
    [IO.File]::WriteAllLines($bootstrap, @('@echo off',
        ('call "%MODLOADER_PREFETCH_VCVARS%"' + $vcArguments + ' >nul'),
        'if errorlevel 1 exit /b %errorlevel%', 'set'), [Text.Encoding]::ASCII)
    $previous = [Environment]::GetEnvironmentVariable('MODLOADER_PREFETCH_VCVARS', 'Process')
    $env:MODLOADER_PREFETCH_VCVARS = $Vcvars
    try {
        $environment = Run $env:ComSpec @('/d', '/s', '/c', 'call', $bootstrap) $work -Capture -Quiet
        foreach ($line in ($environment -split "`r?`n")) {
            $equals = $line.IndexOf('=')
            if ($equals -gt 0) { [Environment]::SetEnvironmentVariable($line.Substring(0, $equals), $line.Substring($equals + 1), 'Process') }
        }
    } finally {
        [Environment]::SetEnvironmentVariable('MODLOADER_PREFETCH_VCVARS', $previous, 'Process')
    }
    if (-not (Get-Command 'cl.exe' -CommandType Application -ErrorAction SilentlyContinue)) { throw 'MSVC environment did not expose cl.exe' }
}

function Get-GitHubSource([string]$Repository, [string]$Ref, [string]$Destination, [string]$Label) {
    $commit = $Ref
    if ($commit -notmatch '^[0-9a-fA-F]{40}$') {
        $response = Join-Path $work ($Label + '-commit.json')
        Download "https://api.github.com/repos/$Repository/commits/$Ref" $response
        $commit = [string](Field (Read-JsonObject $response) 'sha')
        if ($commit -notmatch '^[0-9a-fA-F]{40}$' -or
            -not $commit.StartsWith($Ref, [StringComparison]::OrdinalIgnoreCase)) {
            throw "GitHub did not resolve the exact $Label commit: $Ref"
        }
    }
    switch ($Repository) {
        'Niko1221/Strata' { $directories = @('include', 'src', 'cmake', 'third_party', 'tests', 'tools', 'bench') }
        'ggml-org/llama.cpp' { $directories = @('ggml') }
        default { throw "Unsupported public source repository: $Repository" }
    }
    $gitOptions = @('-c', 'http.version=HTTP/1.1', '-c', 'http.lowSpeedLimit=1024', '-c', 'http.lowSpeedTime=30',
        '-c', 'credential.helper=', '-c', 'credential.interactive=false', '-c', 'core.askPass=', '-c', 'http.extraHeader=')
    $proxy = $env:HTTPS_PROXY
    if (-not $proxy) { $proxy = $env:HTTP_PROXY }
    # Isolate public fetches from credential stores, netrc and interactive prompts.
    # The selected proxy travels only in child-process environment, never argv/logs.
    $gitHome = Join-Path $work 'git-home'
    New-Item -ItemType Directory -Path $gitHome -Force | Out-Null
    $environment = @{
        HOME = $gitHome; USERPROFILE = $gitHome; CURL_HOME = $gitHome; XDG_CONFIG_HOME = $gitHome;
        GIT_CONFIG_GLOBAL = 'NUL'; GIT_CONFIG_PARAMETERS = $null; GIT_CONFIG_COUNT = '0';
        GIT_TERMINAL_PROMPT = '0'; GCM_INTERACTIVE = 'Never'; GIT_ASKPASS = $null; SSH_ASKPASS = $null;
        GIT_NO_LAZY_FETCH = '0'
    }
    if ($proxy) {
        $environment.GIT_CONFIG_COUNT = '1'
        $environment.GIT_CONFIG_KEY_0 = 'http.proxy'
        $environment.GIT_CONFIG_VALUE_0 = $proxy
    }
    $previous = @{}
    foreach ($name in $environment.Keys) {
        $previous[$name] = [Environment]::GetEnvironmentVariable($name, 'Process')
        [Environment]::SetEnvironmentVariable($name, $environment[$name], 'Process')
    }
    try {
        New-Item -ItemType Directory -Path $Destination | Out-Null
        # This root also prevents parent-repository prefixes from filtering patches.
        Run $git ($gitOptions + @('-C', $Destination, 'init')) $work
        Run $git ($gitOptions + @('-C', $Destination, 'remote', 'add', 'origin', "https://github.com/$Repository.git")) $work
        Run $git ($gitOptions + @('-C', $Destination, 'sparse-checkout', 'set', '--cone', '--') + $directories) $work
        Run $git ($gitOptions + @('-C', $Destination, 'fetch', '--depth=1', '--filter=blob:none', 'origin', $commit)) $work
        Run $git ($gitOptions + @('-C', $Destination, 'checkout', '--detach', $commit)) $work
        $head = (Run $git ($gitOptions + @('-C', $Destination, 'rev-parse', 'HEAD')) $work -Capture).Trim()
        if (-not $head.Equals($commit, [StringComparison]::OrdinalIgnoreCase)) {
            throw "Fetched $Label HEAD does not match requested commit $commit"
        }
        return $head.ToLowerInvariant()
    } finally {
        foreach ($name in $previous.Keys) { [Environment]::SetEnvironmentVariable($name, $previous[$name], 'Process') }
    }
}

function Copy-GitSourceSnapshot([string]$Original, [string]$Destination, [string]$Commit, [string]$Name = 'SourceDirectory') {
    # A partial/sparse checkout may deliberately lack documentation blobs. Read the
    # clean materialized tree without refreshing its index or fetching missing blobs.
    $oldLazyFetch = [Environment]::GetEnvironmentVariable('GIT_NO_LAZY_FETCH', 'Process')
    $env:GIT_NO_LAZY_FETCH = '1'
    try {
        $head = (Run $git @('--no-optional-locks', '-C', $Original, 'rev-parse', 'HEAD') $work -Capture).Trim()
        if ($head -notmatch '^[0-9a-fA-F]{40}$' -or -not $head.StartsWith($Commit, [StringComparison]::OrdinalIgnoreCase)) {
            throw "$Name HEAD does not match requested commit $Commit"
        }
        $status = Run $git @('--no-optional-locks', '-C', $Original, 'status', '--porcelain=v1', '--untracked-files=no') $work -Capture -Quiet
        if ($status) { throw "$Name has tracked changes; use a clean checkout: $status" }
        $tracked = Run $git @('--no-optional-locks', '-C', $Original, 'ls-files', '--cached', '-z') $work -Capture -Quiet
        New-Item -ItemType Directory -Path $Destination | Out-Null
        foreach ($relative in ($tracked -split "`0")) {
            if (-not $relative) { continue }
            $from = Full-Path (Join-Path $Original $relative)
            $to = Full-Path (Join-Path $Destination $relative)
            if (-not $from.StartsWith($Original + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase) -or
                -not $to.StartsWith($Destination + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
                throw "Tracked source path escapes its tree: $relative"
            }
            if (-not (Test-Path -LiteralPath $from -PathType Leaf)) { continue }
            New-Item -ItemType Directory -Path (Split-Path -Parent $to) -Force | Out-Null
            Copy-Item -LiteralPath $from -Destination $to
        }
        Run $git @('-C', $Destination, 'init') $work
        return $head
    } finally { [Environment]::SetEnvironmentVariable('GIT_NO_LAZY_FETCH', $oldLazyFetch, 'Process') }
}

function Get-GgmlSource([string]$StrataSource, [string]$SourceOverride) {
    $vendor = Join-Path $StrataSource 'third_party/llama.cpp'
    if (-not $SourceOverride -and (Test-Path -LiteralPath (Join-Path $vendor 'ggml/CMakeLists.txt') -PathType Leaf)) {
        return [pscustomobject]@{ path = $vendor; commit = 'vendored' }
    }
    $cmakeSource = Get-Content -LiteralPath (Join-Path $StrataSource 'CMakeLists.txt') -Raw -Encoding UTF8
    $declaration = [regex]::Matches($cmakeSource, '(?is)FetchContent_Declare\s*\(\s*[^\s()]+\s+(?<body>.*?)\)') |
        Where-Object { $_.Groups['body'].Value -match 'GIT_REPOSITORY\s+"?https://github\.com/ggml-org/llama\.cpp(?:\.git)?"?(?:\s|$)' } |
        Select-Object -First 1
    if ($null -eq $declaration) { throw 'Cannot find the llama.cpp source declaration in CMakeLists.txt' }
    $tag = [regex]::Match($declaration.Groups['body'].Value, '(?im)\bGIT_TAG\s+"?(?<ref>[0-9a-fA-F]{40})"?(?:\s|$)')
    if (-not $tag.Success) { throw 'Cannot find the fixed llama.cpp commit in source CMakeLists.txt' }
    $destination = Join-Path $work 'llama.cpp'
    if ($SourceOverride) {
        $original = Full-Path $SourceOverride
        if ((Paths-Overlap $original $work) -or (Paths-Overlap $original $output)) { throw 'GgmlSourceDirectory must remain separate and read-only' }
        if (-not (Test-Path -LiteralPath (Join-Path $original '.git'))) { throw 'GgmlSourceDirectory must be a Git checkout matching the CMake pin' }
        $commit = Copy-GitSourceSnapshot $original $destination $tag.Groups['ref'].Value 'GgmlSourceDirectory'
    } else {
        $commit = Get-GitHubSource 'ggml-org/llama.cpp' $tag.Groups['ref'].Value $destination 'llama'
    }
    if (-not (Test-Path -LiteralPath (Join-Path $destination 'ggml/CMakeLists.txt') -PathType Leaf)) { throw 'Fixed llama.cpp source is missing ggml' }
    return [pscustomobject]@{ path = $destination; commit = $commit }
}

try {
    $runtime = Full-Path $RuntimeRoot
    $output = Full-Path $OutputDirectory
    $work = Full-Path $WorkDirectory
    if ((Paths-Overlap $runtime $output) -or (Paths-Overlap $runtime $work)) {
        throw 'WorkDirectory and OutputDirectory must not overlap RuntimeRoot'
    }
    Assert-NoLinkedParent $runtime
    Assert-NoLinkedParent $output
    Assert-NoLinkedParent $work
    if ([Environment]::OSVersion.Platform -ne [PlatformID]::Win32NT) { throw 'Only Windows CUDA engines are supported' }
    $engine = Join-Path $runtime 'engine'
    $build = Read-JsonObject (Join-Path $engine 'BUILD.json')
    $component = Read-JsonObject (Join-Path $engine 'engine-component.json')
    $version = Field $build 'version'
    $componentVersion = Field $component 'version'
    if ($version -and $componentVersion -and $version -cne $componentVersion) { throw 'Engine version manifests disagree' }
    if (-not $version) { $version = $componentVersion }
    $commit = Field $build 'commit'
    $buildId = Field $component 'build_id'
    $backend = Field $component 'backend'
    if (-not $backend) { $backend = Field $build 'backend' }
    if (-not $backend -and (Field $build 'cuda')) { $backend = 'cuda' }
    foreach ($entry in @($version, $commit, $backend, $buildId)) {
        if ($entry -isnot [string] -or [string]::IsNullOrWhiteSpace($entry)) { throw 'Engine identity requires version, commit, backend and build_id strings' }
    }
    if ($commit -notmatch '^[0-9a-fA-F]{7,40}$') { throw 'Engine commit must contain 7-40 hexadecimal characters' }
    if (-not $buildId.StartsWith($commit, [StringComparison]::Ordinal)) { throw 'Engine commit and build_id disagree' }
    $platform = Field $component 'platform'
    if ($backend -cne 'cuda' -or ($platform -and $platform -notmatch '^windows(?:-|$)')) {
        throw 'Only Windows CUDA engines are supported'
    }
    $identity = [ordered]@{ version = $version; commit = $commit; backend = $backend; build_id = $buildId }
    $archs = @(Field $build 'archs')
    if ($archs.Count -eq 0) { throw 'BUILD.json has no CUDA architectures' }
    $architectures = @()
    $highest = 0
    foreach ($arch in $archs) {
        if ([string]$arch -notmatch '^[1-9][0-9]*$') { throw "Invalid CUDA architecture in BUILD.json: $arch" }
        $architectures += ([string]$arch + '-real')
        $highest = [Math]::Max($highest, [int]$arch)
    }
    $ptx = Field $build 'ptx'
    if ($null -ne $ptx -and $ptx -isnot [bool]) { throw 'BUILD.json ptx must be boolean' }
    if ($ptx) { $architectures += ([string]$highest + '-virtual') }
    $patchRoot = Join-Path (Split-Path -Parent $PSScriptRoot) 'patches/strata-prefetch'
    $patchManifest = Read-JsonObject (Join-Path $patchRoot 'manifest.json')
    $schemaVersion = Field $patchManifest 'schema_version'
    foreach ($variant in @(Field $patchManifest 'variants')) {
        if ($null -ne $variant -and (Field $variant 'engine_version') -ceq $version) {
            $patchManifest = $variant
            break
        }
    }
    $features = @(Field $patchManifest 'features')
    $rawContinuation = $features -contains 'raw_token_continuation'
    $revision = Field $patchManifest 'patch_revision'
    $patchName = Field $patchManifest 'patch'
    if ($schemaVersion -ne 1 -or $revision -isnot [string] -or -not $revision -or
        $patchName -isnot [string] -or $patchName -ne [IO.Path]::GetFileName($patchName) -or
        (Field $patchManifest 'base_commit') -notmatch '^[0-9a-fA-F]{40}$') { throw 'Invalid prefetch patch manifest' }
    $patch = Join-Path $patchRoot $patchName
    if (-not (Test-Path -LiteralPath $patch -PathType Leaf)) { throw "Missing versioned patch: $patch" }
    $executable = 'strata-startup-prefetch.exe'
    $plan = [ordered]@{ schema_version = 1; feature = 'startup_expert_prefetch'; engine_identity = $identity;
        source_ref = $commit; cuda_architectures = @($architectures); patch_revision = $revision; executable = $executable;
        features = @($features) }
    if ($PlanOnly) { $plan | ConvertTo-Json -Depth 8; exit 0 }

    foreach ($name in @($executable, 'startup-prefetch.json')) {
        if (Test-Path -LiteralPath (Join-Path $output $name)) { throw "Output already contains $name; use an empty staging directory" }
    }
    $source = Join-Path $work 'source'
    $buildDirectory = Join-Path $work 'build'
    if ((Test-Path -LiteralPath $source) -or (Test-Path -LiteralPath $buildDirectory)) {
        throw 'WorkDirectory already contains a source/build tree; use a fresh work directory'
    }
    $config = [pscustomobject]@{}
    if ($ToolchainConfig -and (Test-Path -LiteralPath $ToolchainConfig)) { $config = Read-JsonObject $ToolchainConfig }
    $cuda = Configured-Path $config 'cuda_root' -Directory
    if (-not $cuda) { $cuda = $env:CUDA_PATH }
    if (-not $cuda) {
        $cudaParent = Join-Path $env:ProgramFiles 'NVIDIA GPU Computing Toolkit/CUDA'
        if (Test-Path -LiteralPath $cudaParent) {
            $cuda = Get-ChildItem -LiteralPath $cudaParent -Directory | Where-Object { $_.Name -match '^v[0-9]+\.[0-9]+$' } |
                Sort-Object { [version]$_.Name.Substring(1) } -Descending | Select-Object -First 1 -ExpandProperty FullName
        }
    }
    if (-not $cuda -or -not (Test-Path -LiteralPath (Join-Path $cuda 'bin/nvcc.exe') -PathType Leaf)) {
        throw 'Missing CUDA toolkit (nvcc.exe); set ToolchainConfig.cuda_root'
    }
    $vcvars = Configured-Path $config 'vcvars'
    $vsRoot = $null
    if (-not $vcvars) {
        $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio/Installer/vswhere.exe'
        if (Test-Path -LiteralPath $vswhere -PathType Leaf) {
            $vsRoot = (Run $vswhere @('-latest', '-products', '*', '-requires', 'Microsoft.VisualStudio.Component.VC.Tools.x86.x64', '-property', 'installationPath') $runtime -Capture).Trim()
        }
        if ($vsRoot) { $vcvars = Join-Path $vsRoot 'VC/Auxiliary/Build/vcvars64.bat' }
    }
    if (-not $vcvars -or -not (Test-Path -LiteralPath $vcvars -PathType Leaf)) { throw 'Missing Visual C++ environment; set ToolchainConfig.vcvars' }
    $cmakeCandidates = @((Join-Path $env:ProgramFiles 'CMake/bin/cmake.exe'))
    $ninjaCandidates = @()
    if ($vsRoot) {
        $cmakeCandidates += (Join-Path $vsRoot 'Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin/cmake.exe')
        $ninjaCandidates += (Join-Path $vsRoot 'Common7/IDE/CommonExtensions/Microsoft/CMake/Ninja/ninja.exe')
    }
    $cmake = Find-Tool 'cmake.exe' (Configured-Path $config 'cmake') $cmakeCandidates
    $ctest = Join-Path (Split-Path -Parent $cmake) 'ctest.exe'
    if (-not (Test-Path -LiteralPath $ctest -PathType Leaf)) { throw "Missing ctest.exe beside CMake: $ctest" }
    $ninja = Find-Tool 'ninja.exe' (Configured-Path $config 'ninja') $ninjaCandidates
    $git = Find-Tool 'git.exe' (Configured-Path $config 'git') @((Join-Path $env:ProgramFiles 'Git/cmd/git.exe'))
    if ($SourceDirectory) {
        $originalSource = Full-Path $SourceDirectory
        if ((Paths-Overlap $originalSource $work) -or (Paths-Overlap $originalSource $output)) { throw 'SourceDirectory must remain separate and read-only' }
        if (-not (Test-Path -LiteralPath (Join-Path $originalSource '.git'))) { throw 'SourceDirectory must be a Git checkout whose HEAD matches the engine commit' }
    }
    New-Item -ItemType Directory -Path $work -Force | Out-Null
    $temporary = Join-Path $work 'temp'
    New-Item -ItemType Directory -Path $temporary -Force | Out-Null
    $env:TEMP = $temporary
    $env:TMP = $temporary
    Import-MsvcEnvironment $vcvars $temporary
    $env:CUDA_PATH = $cuda
    $env:PATH = (Join-Path $cuda 'bin') + ';' + $env:PATH
    $cudaVersion = Run (Join-Path $cuda 'bin/nvcc.exe') @('--version') $work -Capture
    Assert-CudaMajor (Field $build 'cuda') $cudaVersion
    if ($SourceDirectory) {
        $sourceCommit = Copy-GitSourceSnapshot $originalSource $source $commit
    } else {
        $sourceCommit = Get-GitHubSource 'Niko1221/Strata' $commit $source 'strata'
    }
    $sourceCommit = $sourceCommit.ToLowerInvariant()
    if (-not (Test-Path -LiteralPath (Join-Path $source 'CMakeLists.txt') -PathType Leaf)) { throw 'Downloaded source has no CMakeLists.txt' }
    try {
        Run $git @('-C', $source, 'apply', '--check', $patch) $work
        Run $git @('-C', $source, 'apply', $patch) $work
        Run $git @('-C', $source, 'apply', '--reverse', '--check', $patch) $work
    } catch { throw "Patch $revision needs adaptation for $sourceCommit : $($_.Exception.Message)" }
    $ggmlSource = Get-GgmlSource $source $GgmlSourceDirectory
    $ggml = $ggmlSource.path
    $ggmlCommit = $ggmlSource.commit

    $cmakeVersion = Run $cmake @('--version') $work -Capture
    $ninjaVersion = Run $ninja @('--version') $work -Capture
    Run $cmake @('-S', $source, '-B', $buildDirectory, '-G', 'Ninja', '-DCMAKE_BUILD_TYPE=Release',
        "-DCMAKE_MAKE_PROGRAM=$ninja", "-DCUDAToolkit_ROOT=$cuda", "-DCMAKE_CUDA_COMPILER=$(Join-Path $cuda 'bin/nvcc.exe')",
        '-DCMAKE_CUDA_RUNTIME_LIBRARY=Static', '-DSTRATA_ENABLE_CUDA=ON', '-DSTRATA_BUILD_TESTS=ON',
        '-DSTRATA_NATIVE_EXPERTS=ON', '-DSTRATA_PORTABLE=ON', "-DSTRATA_GGML_DIR=$ggml",
        ('-DCMAKE_CUDA_ARCHITECTURES=' + ($architectures -join ';'))) $work
    $targets = @('strata', 'expert_layout_test', 'file_expert_source_test')
    $testPattern = '^(expert_layout_test|file_expert_source_test)$'
    if ($rawContinuation) {
        $targets += 'decode_boundary_test'
        $testPattern = '^(expert_layout_test|file_expert_source_test|decode_boundary_test)$'
    }
    Run $cmake (@('--build', $buildDirectory, '--config', 'Release', '--target') + $targets + @('--parallel', '4')) $work
    $oldPath = $env:PATH
    try {
        # The managed engine supplies its CUDA DLLs. Do not borrow another Strata/Python/toolkit installation.
        $env:PATH = $engine + ';' + (Join-Path $env:SystemRoot 'System32') + ';' + $env:SystemRoot
        Run $ctest @('--test-dir', $buildDirectory, '--build-config', 'Release', '-R',
            $testPattern, '--output-on-failure', '--no-tests=error') $work
        $builtExe = Join-Path $buildDirectory 'strata.exe'
        $help = Run $builtExe @('--help') $buildDirectory -Capture
        if ($help -notmatch '(?m)--startup-expert-prefetch\b') { throw 'Built engine is missing --startup-expert-prefetch' }
        if ($rawContinuation -and $help -notmatch '(?m)--raw-token-continuation\b') { throw 'Built engine is missing --raw-token-continuation' }
    } finally { $env:PATH = $oldPath }
    New-Item -ItemType Directory -Path $output -Force | Out-Null
    Copy-Item -LiteralPath $builtExe -Destination (Join-Path $output $executable)
    $manifest = [ordered]@{ schema_version = 1; feature = 'startup_expert_prefetch'; engine_identity = $identity;
        source_commit = $sourceCommit; executable = $executable; patch_revision = $revision; features = @($features);
        build = [ordered]@{ configuration = 'Release'; generator = 'Ninja'; cuda_architectures = @($architectures);
            cuda_runtime = 'Static'; cuda = $cudaVersion; cmake = $cmakeVersion; ninja = $ninjaVersion; ggml_commit = $ggmlCommit } }
    [IO.File]::WriteAllText((Join-Path $output 'startup-prefetch.json'), ($manifest | ConvertTo-Json -Depth 8), (New-Object Text.UTF8Encoding($false)))
    [Console]::Out.WriteLine("Built $output/$executable")
} catch {
    [Console]::Error.WriteLine('PREFETCH_BUILD: ' + $_.Exception.Message)
    exit 1
}
