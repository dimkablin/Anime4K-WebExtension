param(
    [string]$ExtensionId = 'pogjjeeohgepdfjbdikpkoiljgnccchg',
    [string]$PythonPath = "$env:USERPROFILE\Documents\apps\TheAnimeScripter\.venv\Scripts\python.exe",
    [string]$ModelDirectory = "$env:USERPROFILE\Documents\apps\TheAnimeScripter\weights\animesr-onnx"
)

$ErrorActionPreference = 'Stop'
$hostName = 'com.dimkablin.animesr'
$installRoot = Join-Path $env:LOCALAPPDATA 'Anime4K-WebExtension\AnimeSR'
$modelRoot = Join-Path $installRoot 'models'
$engineNames = @(
    'AnimeSR_v2_fp16_op20_fp16_720x1280.engine',
    'AnimeSR_v2_fp16_op20_fp16_1080x1920.engine'
)
$ffmpeg = (Get-Command ffmpeg -ErrorAction Stop).Source

if (-not (Test-Path -LiteralPath $PythonPath -PathType Leaf)) {
    throw "Compatible Python runtime was not found at $PythonPath"
}
foreach ($engineName in $engineNames) {
    $engineSource = Join-Path $ModelDirectory $engineName
    if (-not (Test-Path -LiteralPath $engineSource -PathType Leaf)) {
        throw "AnimeSR TensorRT engine was not found at $engineSource"
    }
}

& $PythonPath -c 'import torch, tensorrt, simple_websocket, werkzeug' | Out-Null
if ($LASTEXITCODE -ne 0) {
    throw 'Python runtime is missing torch, TensorRT, simple-websocket, or Werkzeug.'
}

New-Item -ItemType Directory -Force -Path $modelRoot | Out-Null
Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'animesr_host.py') -Destination $installRoot -Force
Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'animesr_engine.py') -Destination $installRoot -Force
foreach ($engineName in $engineNames) {
    Copy-Item -LiteralPath (Join-Path $ModelDirectory $engineName) -Destination $modelRoot -Force
}

$launcherPath = Join-Path $installRoot 'animesr_host.cmd'
$hostPath = Join-Path $installRoot 'animesr_host.py'
$ffmpegDirectory = Split-Path -Parent $ffmpeg
$launcher = @"
@echo off
set "ANIMESR_MODEL_DIR=$modelRoot"
set "PATH=$ffmpegDirectory;%PATH%"
"$PythonPath" "$hostPath"
"@
Set-Content -LiteralPath $launcherPath -Value $launcher -Encoding Ascii

$manifestPath = Join-Path $installRoot 'native-host.json'
$manifest = [ordered]@{
    name = $hostName
    description = 'AnimeSR v2 TensorRT native host for Anime4K WebExtension'
    path = $launcherPath
    type = 'stdio'
    allowed_origins = @("chrome-extension://$ExtensionId/")
}
$manifest | ConvertTo-Json -Depth 3 | Set-Content -LiteralPath $manifestPath -Encoding Utf8

$registryPaths = @(
    "HKCU:\Software\Microsoft\Edge\NativeMessagingHosts\$hostName",
    "HKCU:\Software\Google\Chrome\NativeMessagingHosts\$hostName"
)
foreach ($registryPath in $registryPaths) {
    New-Item -Path $registryPath -Force | Out-Null
    Set-Item -Path $registryPath -Value $manifestPath
}

Write-Host "AnimeSR native host installed for extension $ExtensionId"
Write-Host 'Restart Edge/Chrome and reload dist-chrome on the extensions page.'
