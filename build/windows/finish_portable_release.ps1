param(
    [string]$Version = "0.5.0"
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$DistDir = Join-Path $Root "dist"
$PortableDir = Join-Path $DistDir "Pelak-Khan"
$PortableExe = Join-Path $PortableDir "Pelak-Khan.exe"
$ReleaseDir = Join-Path $Root "release"
$ZipPath = Join-Path $ReleaseDir "Pelak-Khan-Portable-Windows-x64-v$Version.zip"
$HashPath = Join-Path $ReleaseDir "Pelak-Khan-Portable-Windows-x64-v$Version.sha256.txt"

if (-not (Test-Path $PortableExe)) {
    throw "Portable build not found: $PortableExe`nRun build_portable.ps1 first."
}

New-Item -ItemType Directory -Path $ReleaseDir -Force | Out-Null
Remove-Item -Force $ZipPath, $HashPath -ErrorAction SilentlyContinue

Write-Host "Creating portable ZIP directly from dist (no release-stage copy)..." -ForegroundColor Yellow
$TarCommand = Get-Command "tar.exe" -ErrorAction SilentlyContinue
if (-not $TarCommand) {
    throw "Windows tar.exe was not found. Windows 10/11 normally includes it."
}

& $TarCommand.Source -a -c -f $ZipPath -C $DistDir "Pelak-Khan"
if ($LASTEXITCODE -ne 0 -or -not (Test-Path $ZipPath)) {
    throw "Failed to create portable ZIP with tar.exe."
}

$Hash = (Get-FileHash -Algorithm SHA256 $ZipPath).Hash.ToLowerInvariant()
"$Hash  $([IO.Path]::GetFileName($ZipPath))" | Set-Content $HashPath -Encoding ASCII

$FolderSizeBytes = (Get-ChildItem $PortableDir -Recurse -File | Measure-Object Length -Sum).Sum
$FolderSize = [math]::Round($FolderSizeBytes / 1MB, 1)
$ZipSize = [math]::Round((Get-Item $ZipPath).Length / 1MB, 1)

Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host " PORTABLE RELEASE READY" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Write-Host "Folder : $PortableDir" -ForegroundColor Green
Write-Host "ZIP    : $ZipPath" -ForegroundColor Green
Write-Host "SHA256 : $HashPath" -ForegroundColor Green
Write-Host "Folder : $FolderSize MB" -ForegroundColor Green
Write-Host "ZIP    : $ZipSize MB" -ForegroundColor Green
