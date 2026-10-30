param(
    [int]$CameraIndex = 0,
    [ValidateSet('DSHOW', 'MSMF')][string]$Backend = 'DSHOW'
)

$ErrorActionPreference = 'Stop'
$projectDirectory = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$deploymentPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $deploymentPython)) {
    throw 'Windows部署解释器不存在，请先查看docs/GETTING_STARTED.md，不要改WSL环境。'
}
& $deploymentPython -X utf8 (Join-Path $projectDirectory 'deployment\tools\live_camera.py') `
    --camera-index $CameraIndex --backend $Backend
exit $LASTEXITCODE
