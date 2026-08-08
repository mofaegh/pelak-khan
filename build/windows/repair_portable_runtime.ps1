param(
    [string]$Version = "0.5.0"
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$PortableDir = Join-Path $Root "dist\Pelak-Khan"
$PortableLauncher = Join-Path $PortableDir "apps\desktop\launcher.py"
$SourceLauncher = Join-Path $Root "apps\desktop\launcher.py"
$RuntimePython = Join-Path $PortableDir "runtime\python"
$ReleaseDir = Join-Path $Root "release"
$ZipPath = Join-Path $ReleaseDir "Pelak-Khan-Portable-Windows-x64-v$Version.zip"
$HashPath = Join-Path $ReleaseDir "Pelak-Khan-Portable-Windows-x64-v$Version.sha256.txt"

foreach ($Required in @($PortableDir, $SourceLauncher, (Join-Path $RuntimePython "python.exe"))) {
    if (-not (Test-Path $Required)) { throw "Required path not found: $Required" }
}

Write-Host "[1/3] Updating portable launcher source..." -ForegroundColor Yellow
New-Item -ItemType Directory -Path (Split-Path $PortableLauncher -Parent) -Force | Out-Null
Copy-Item -Force $SourceLauncher $PortableLauncher

Write-Host "[2/3] Running strengthened portable self-test..." -ForegroundColor Yellow
$OldPythonHome = $env:PYTHONHOME
$OldPythonPath = $env:PYTHONPATH
$OldNoUserSite = $env:PYTHONNOUSERSITE
$OldPortableRoot = $env:PELAK_PORTABLE_ROOT
try {
    $env:PYTHONHOME = $RuntimePython
    $env:PYTHONPATH = "$PortableDir;$PortableDir\src"
    $env:PYTHONNOUSERSITE = "1"
    $env:PELAK_PORTABLE_ROOT = $PortableDir
    & (Join-Path $RuntimePython "python.exe") -m apps.desktop.launcher --self-test
    if ($LASTEXITCODE -ne 0) { throw "Portable self-test failed." }
}
finally {
    $env:PYTHONHOME = $OldPythonHome
    $env:PYTHONPATH = $OldPythonPath
    $env:PYTHONNOUSERSITE = $OldNoUserSite
    $env:PELAK_PORTABLE_ROOT = $OldPortableRoot
}

Write-Host "[3/3] Recreating release ZIP and SHA-256..." -ForegroundColor Yellow
New-Item -ItemType Directory -Path $ReleaseDir -Force | Out-Null
Remove-Item -Force $ZipPath, $HashPath -ErrorAction SilentlyContinue
$TarCommand = Get-Command "tar.exe" -ErrorAction SilentlyContinue
if ($TarCommand) {
    & $TarCommand.Source -a -c -f $ZipPath -C (Split-Path $PortableDir -Parent) "Pelak-Khan"
    if ($LASTEXITCODE -ne 0) { throw "tar.exe failed while creating ZIP." }
}
else {
    Compress-Archive -Path $PortableDir -DestinationPath $ZipPath -CompressionLevel Optimal -Force
}
if (-not (Test-Path $ZipPath)) { throw "Portable ZIP was not generated." }
$Hash = (Get-FileHash -Algorithm SHA256 $ZipPath).Hash.ToLowerInvariant()
"$Hash  $([IO.Path]::GetFileName($ZipPath))" | Set-Content $HashPath -Encoding ASCII
$ZipSize = [math]::Round((Get-Item $ZipPath).Length / 1MB, 1)
Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host " PORTABLE RUNTIME REPAIRED" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Write-Host "ZIP    : $ZipPath" -ForegroundColor Green
Write-Host "SHA256 : $HashPath" -ForegroundColor Green
Write-Host "ZIP    : $ZipSize MB" -ForegroundColor Green
