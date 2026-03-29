param(
    [string]$EnvName = 'openmmlab_env',
    [switch]$ForceRecreate
)

$ErrorActionPreference = 'Stop'

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$envRoot = Join-Path $repoRoot $EnvName
$envScripts = Join-Path $envRoot 'Scripts'
$python = Join-Path $envRoot 'Scripts\python.exe'
$sanityScript = Join-Path $repoRoot 'hydrosat\tools\check_openmmlab_env.py'
$mmcvPatchScript = Join-Path $repoRoot 'hydrosat\tools\patch_mmcv_windows_build.py'
$buildRoot = Join-Path $repoRoot '.build'
$mmcvRoot = Join-Path $buildRoot 'mmcv-2.1.0'
$vswhere = 'C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe'
$expectedCudaToolkit = 'v13.0'
$supportedVsVersionRange = '[17.0,18.0)'

function Require-Command {
    param([string]$Name)
    $command = Get-Command $Name -ErrorAction SilentlyContinue
    if (-not $command) {
        throw "Missing required command: $Name"
    }
}

function Invoke-VcvarsCommand {
    param(
        [string]$Command,
        [string]$WorkingDirectory = $repoRoot
    )

    $cmdLine = @(
        "set `"CUDA_HOME=$($script:CUDAHome)`"",
        "set `"PATH=$($script:EnvScripts);$($script:CUDAHome)\bin;%PATH%`"",
        "set `"MMCV_WITH_OPS=1`"",
        "set `"FORCE_CUDA=1`"",
        "set `"TORCH_CUDA_ARCH_LIST=12.0`"",
        "set `"MMCV_DISABLE_SPARSE_OPS=1`"",
        "set `"DISTUTILS_USE_SDK=1`"",
        "set `"MSSdk=1`"",
        "call `"$script:VcVars`" >nul",
        "cd /d `"$WorkingDirectory`"",
        $Command
    ) -join ' && '

    cmd /c $cmdLine
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed: $Command"
    }
}

function Invoke-CheckedCommand {
    param(
        [string]$FilePath,
        [string[]]$ArgumentList = @()
    )

    & $FilePath @ArgumentList
    if ($LASTEXITCODE -ne 0) {
        $renderedArgs = $ArgumentList -join ' '
        throw "Command failed: $FilePath $renderedArgs"
    }
}

Require-Command git
Require-Command py
Require-Command nvidia-smi

if (-not (Test-Path $vswhere)) {
    throw "Missing Visual Studio installer helper at $vswhere"
}

$VcVars = & $vswhere -latest -version $supportedVsVersionRange -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -find **\VC\Auxiliary\Build\vcvars64.bat | Select-Object -First 1
if (-not $VcVars) {
    $unsupportedVcVars = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -find **\VC\Auxiliary\Build\vcvars64.bat | Select-Object -First 1
    if ($unsupportedVcVars) {
        throw "CUDA Toolkit $expectedCudaToolkit supports Visual Studio 2019/2022 host compilers, but the newest detected toolchain is unsupported: $unsupportedVcVars. Install Visual Studio Build Tools 2022 with the C++ workload and rerun."
    }

    throw 'Unable to find vcvars64.bat for Visual Studio Build Tools 2022. Install Visual Studio Build Tools 2022 with the C++ workload.'
}

$cudaRoots = Get-ChildItem 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA' -Directory -ErrorAction SilentlyContinue | Sort-Object Name -Descending
if (-not $cudaRoots) {
    throw 'Unable to find a native CUDA toolkit installation under C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA.'
}

$cudaDir = $cudaRoots | Where-Object { $_.Name -eq $expectedCudaToolkit } | Select-Object -First 1
if (-not $cudaDir) {
    $foundVersions = ($cudaRoots | Select-Object -ExpandProperty Name) -join ', '
    throw "This bootstrap expects CUDA Toolkit $expectedCudaToolkit because it installs torch==2.10.0+cu130. Found: $foundVersions. Install CUDA Toolkit $expectedCudaToolkit and rerun."
}

$CUDAHome = $cudaDir.FullName
$nvcc = Join-Path $CUDAHome 'bin\nvcc.exe'
if (-not (Test-Path $nvcc)) {
    throw "Missing nvcc at $nvcc"
}

Invoke-CheckedCommand py @('-3.10', '--version')

if ($ForceRecreate -and (Test-Path $envRoot)) {
    Remove-Item -Recurse -Force $envRoot
}

if (-not (Test-Path $envRoot)) {
    Invoke-CheckedCommand py @('-3.10', '-m', 'venv', $envRoot)
}

if (-not (Test-Path $python)) {
    throw "Missing environment interpreter at $python"
}

if (-not (Test-Path $buildRoot)) {
    New-Item -ItemType Directory -Path $buildRoot | Out-Null
}

if (-not (Test-Path $mmcvRoot)) {
    Invoke-CheckedCommand git @('clone', '--branch', 'v2.1.0', '--depth', '1', 'https://github.com/open-mmlab/mmcv.git', $mmcvRoot)
}
else {
    Invoke-CheckedCommand git @('-C', $mmcvRoot, 'fetch', '--tags')
    Invoke-CheckedCommand git @('-C', $mmcvRoot, 'checkout', 'v2.1.0')
}

Invoke-CheckedCommand $python @($mmcvPatchScript, $mmcvRoot)

foreach ($artifact in @('build', 'dist', 'mmcv.egg-info')) {
    $artifactPath = Join-Path $mmcvRoot $artifact
    if (Test-Path $artifactPath) {
        Remove-Item -Recurse -Force $artifactPath
    }
}

Invoke-CheckedCommand $python @('-m', 'pip', 'install', '--upgrade', 'pip', 'wheel')
Invoke-CheckedCommand $python @('-m', 'pip', 'install', 'ninja', 'packaging')
Invoke-CheckedCommand $python @('-m', 'pip', 'install', 'torch==2.10.0', 'torchvision==0.25.0', '--index-url', 'https://download.pytorch.org/whl/cu130')
Invoke-CheckedCommand $python @('-m', 'pip', 'install', 'openmim==0.3.9', 'mmengine==0.10.7', 'ftfy==6.3.1', 'numpy==1.26.4', 'opencv-python==4.10.0.84')
Invoke-CheckedCommand $python @('-m', 'pip', 'uninstall', '-y', 'mmcv', 'mmcv-lite')

Invoke-VcvarsCommand "`"$python`" -m pip install --no-build-isolation `"$mmcvRoot`""

Invoke-CheckedCommand $python @('-m', 'pip', 'install', 'mmdet==3.3.0', 'mmsegmentation==1.2.2')
Invoke-CheckedCommand $python @($sanityScript, '--require-cuda')
Invoke-CheckedCommand $python @('-m', 'pip', 'install', '-e', "$repoRoot[dev]")

Write-Host "OpenMMLab environment is ready at $envRoot"
