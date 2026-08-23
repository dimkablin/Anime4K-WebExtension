$ErrorActionPreference = 'Stop'
$hostName = 'com.dimkablin.animesr'
$installParent = [System.IO.Path]::GetFullPath((Join-Path $env:LOCALAPPDATA 'Anime4K-WebExtension'))
$installRoot = [System.IO.Path]::GetFullPath((Join-Path $installParent 'AnimeSR'))

if (-not $installRoot.StartsWith($installParent + [System.IO.Path]::DirectorySeparatorChar)) {
    throw "Refusing to remove unexpected path: $installRoot"
}

$registryPaths = @(
    "HKCU:\Software\Microsoft\Edge\NativeMessagingHosts\$hostName",
    "HKCU:\Software\Google\Chrome\NativeMessagingHosts\$hostName"
)
foreach ($registryPath in $registryPaths) {
    if (Test-Path -LiteralPath $registryPath) {
        Remove-Item -LiteralPath $registryPath -Recurse -Force
    }
}
if (Test-Path -LiteralPath $installRoot) {
    Remove-Item -LiteralPath $installRoot -Recurse -Force
}

Write-Host 'AnimeSR native host removed.'
