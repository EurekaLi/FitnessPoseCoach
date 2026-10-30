param(
    [string]$BasePython = 'python',
    [switch]$SkipTensorRT
)

$ErrorActionPreference = 'Stop'
$runtimeDirectory = $PSScriptRoot
$deploymentPython = Join-Path $runtimeDirectory '.venv\Scripts\python.exe'

function Invoke-CheckedPython {
    param([string]$Executable, [string[]]$PythonArguments)
    & $Executable @PythonArguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python命令失败，停止安装，不修改其他环境：$Executable"
    }
}

if (-not (Test-Path -LiteralPath (Join-Path $runtimeDirectory '.venv'))) {
    Invoke-CheckedPython $BasePython @('-m', 'venv', (Join-Path $runtimeDirectory '.venv'))
}
if (-not (Test-Path -LiteralPath (Join-Path $runtimeDirectory '.venv\pyvenv.cfg'))) {
    throw '目标目录不是可识别的虚拟环境，不覆盖已有内容。'
}
Invoke-CheckedPython $deploymentPython @('-m', 'ensurepip', '--upgrade')
Invoke-CheckedPython $deploymentPython @('-m', 'pip', 'install', '--index-url', 'https://pypi.org/simple',
    '-r', (Join-Path $runtimeDirectory 'requirements-runtime.txt'))

# rtmlib元数据同时声明CPU ORT及两种cv2发行包；已有等价GPU接口，避免重复覆盖同名模块。
Invoke-CheckedPython $deploymentPython @('-m', 'pip', 'install', '--index-url', 'https://pypi.org/simple',
    '--no-deps', 'rtmlib==0.0.16')
Invoke-CheckedPython $deploymentPython @('-c',
    'from rtmlib import Body, RTMPose, YOLOX, draw_skeleton; print("RTMLIB_IMPORT_OK")')
if (-not $SkipTensorRT) {
    Invoke-CheckedPython $deploymentPython @('-m', 'pip', 'install', '--index-url', 'https://pypi.org/simple',
        '-r', (Join-Path $runtimeDirectory 'requirements-tensorrt.txt'))
}
Write-Output "WINDOWS_DEPLOY_PACKAGES_READY python=$deploymentPython"
