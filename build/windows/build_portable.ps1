param(
    [string]$Version = "0.5.0"
)

$ErrorActionPreference = "Stop"
$ExpectedVersion = "0.5.0"
if ($Version -ne $ExpectedVersion) {
    throw "This source tree is Pelak-Khan v$ExpectedVersion. Build it with -Version $ExpectedVersion."
}

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$VenvPython = Join-Path $Root ".venv\Scripts\python.exe"
$VenvSite = Join-Path $Root ".venv\Lib\site-packages"
$Detector = Join-Path $Root "models\runtime\detector_v1.pt"
$Ocr = Join-Path $Root "models\runtime\ocr_v1.pt"
$Frontend = Join-Path $Root "apps\frontend\index.html"
$Launcher = Join-Path $Root "apps\desktop\launcher.py"
$Bootstrap = Join-Path $Root "apps\desktop\bootstrap.py"
$Icon = Join-Path $Root "build\windows\assets\pelak-khan.ico"

$DistDir = Join-Path $Root "dist"
$PortableDir = Join-Path $DistDir "Pelak-Khan"
$RuntimePython = Join-Path $PortableDir "runtime\python"
$RuntimeSite = Join-Path $RuntimePython "Lib\site-packages"
$PortableExe = Join-Path $PortableDir "Pelak-Khan.exe"
$BuildWork = Join-Path $Root "build-output\portable-runtime"
$BootstrapDist = Join-Path $BuildWork "bootstrap-dist"
$ReleaseDir = Join-Path $Root "release"
$ZipPath = Join-Path $ReleaseDir "Pelak-Khan-Portable-Windows-x64-v$Version.zip"
$HashPath = Join-Path $ReleaseDir "Pelak-Khan-Portable-Windows-x64-v$Version.sha256.txt"

function Invoke-Robocopy {
    param(
        [Parameter(Mandatory=$true)][string]$Source,
        [Parameter(Mandatory=$true)][string]$Destination,
        [string[]]$Extra = @()
    )

    New-Item -ItemType Directory -Path $Destination -Force | Out-Null
    & robocopy $Source $Destination /E /COPY:DAT /DCOPY:DAT /R:1 /W:1 /NFL /NDL /NJH /NJS /NP @Extra | Out-Host
    $Code = $LASTEXITCODE
    if ($Code -ge 8) {
        throw "Robocopy failed ($Code): $Source -> $Destination"
    }
}

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host " Pelak-Khan PORTABLE RUNTIME BUILDER v$Version" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Packaging mode: bundled Python runtime + lightweight EXE launcher" -ForegroundColor DarkCyan

foreach ($Required in @($VenvPython, $VenvSite, $Detector, $Ocr, $Frontend, $Launcher, $Bootstrap, $Icon)) {
    if (-not (Test-Path $Required)) {
        throw "Required file not found: $Required"
    }
}

Push-Location $Root
try {
    $Bits = (& $VenvPython -c "import struct; print(struct.calcsize('P') * 8)").Trim()
    if ($Bits -ne "64") {
        throw "A 64-bit Python environment is required to build the Windows x64 portable release."
    }

    $BasePython = (& $VenvPython -c "import sys; print(sys.base_prefix)").Trim()
    if (-not (Test-Path (Join-Path $BasePython "python.exe"))) {
        throw "Base Python installation was not found: $BasePython"
    }

    Write-Host "[1/9] Preparing build/runtime dependencies..." -ForegroundColor Yellow
    & $VenvPython -m pip install --upgrade "pyinstaller>=6.10,<7" "pystray>=0.19.5,<1" "lap>=0.5.12"
    if ($LASTEXITCODE -ne 0) { throw "Dependency preparation failed." }

    Write-Host "[2/9] Cleaning old portable output..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force $PortableDir -ErrorAction SilentlyContinue
    Remove-Item -Recurse -Force $BuildWork -ErrorAction SilentlyContinue
    New-Item -ItemType Directory -Path $PortableDir -Force | Out-Null
    New-Item -ItemType Directory -Path $ReleaseDir -Force | Out-Null
    Remove-Item -Force $ZipPath, $HashPath -ErrorAction SilentlyContinue

    Write-Host "[3/9] Copying relocatable Python standard runtime..." -ForegroundColor Yellow
    $ExcludedBaseDirs = @(
        "/XD",
        (Join-Path $BasePython "Lib\site-packages"),
        (Join-Path $BasePython "Scripts"),
        (Join-Path $BasePython "Tools"),
        (Join-Path $BasePython "Doc"),
        (Join-Path $BasePython "share"),
        (Join-Path $BasePython "include"),
        (Join-Path $BasePython "libs")
    )
    Invoke-Robocopy -Source $BasePython -Destination $RuntimePython -Extra $ExcludedBaseDirs

    if (-not (Test-Path (Join-Path $RuntimePython "python.exe"))) {
        throw "Bundled python.exe was not created."
    }
    if (-not (Test-Path (Join-Path $RuntimePython "pythonw.exe"))) {
        throw "Bundled pythonw.exe was not created."
    }

    Write-Host "[4/9] Copying Pelak-Khan runtime packages..." -ForegroundColor Yellow
    Invoke-Robocopy -Source $VenvSite -Destination $RuntimeSite

    # Editable-install .pth files contain absolute developer-machine paths.
    # Pelak-Khan source is shipped explicitly and PYTHONPATH is set by bootstrap.exe.
    Get-ChildItem $RuntimeSite -Filter "__editable__*" -Force -ErrorAction SilentlyContinue |
        Remove-Item -Recurse -Force -ErrorAction SilentlyContinue

    Write-Host "[5/9] Copying application source, frontend, models, and assets..." -ForegroundColor Yellow
    Invoke-Robocopy -Source (Join-Path $Root "apps") -Destination (Join-Path $PortableDir "apps") -Extra @("/XD", "__pycache__")
    Invoke-Robocopy -Source (Join-Path $Root "src") -Destination (Join-Path $PortableDir "src") -Extra @("/XD", "__pycache__")

    if (Test-Path (Join-Path $Root "configs")) {
        Invoke-Robocopy -Source (Join-Path $Root "configs") -Destination (Join-Path $PortableDir "configs") -Extra @("/XD", "__pycache__")
    }

    $PortableModels = Join-Path $PortableDir "models\runtime"
    New-Item -ItemType Directory -Path $PortableModels -Force | Out-Null
    Copy-Item -Force $Detector (Join-Path $PortableModels "detector_v1.pt")
    Copy-Item -Force $Ocr (Join-Path $PortableModels "ocr_v1.pt")

    $PortableAssets = Join-Path $PortableDir "build\windows\assets"
    New-Item -ItemType Directory -Path $PortableAssets -Force | Out-Null
    Copy-Item -Force $Icon (Join-Path $PortableAssets "pelak-khan.ico")

    Write-Host "[6/9] Building lightweight Pelak-Khan.exe launcher..." -ForegroundColor Yellow
    New-Item -ItemType Directory -Path $BootstrapDist -Force | Out-Null
    $BootstrapWork = Join-Path $BuildWork "bootstrap-work"
    $BootstrapSpec = Join-Path $BuildWork "bootstrap-spec"
    New-Item -ItemType Directory -Path $BootstrapWork -Force | Out-Null
    New-Item -ItemType Directory -Path $BootstrapSpec -Force | Out-Null

    & $VenvPython -m PyInstaller `
        --noconfirm `
        --clean `
        --onefile `
        --windowed `
        --noupx `
        --name "Pelak-Khan" `
        --icon $Icon `
        --distpath $BootstrapDist `
        --workpath $BootstrapWork `
        --specpath $BootstrapSpec `
        $Bootstrap
    if ($LASTEXITCODE -ne 0) { throw "Lightweight launcher build failed." }

    $BuiltBootstrap = Join-Path $BootstrapDist "Pelak-Khan.exe"
    if (-not (Test-Path $BuiltBootstrap)) { throw "Launcher EXE was not generated." }
    Copy-Item -Force $BuiltBootstrap $PortableExe

    Write-Host "[7/9] Preparing portable data and documentation..." -ForegroundColor Yellow
    $DataDir = Join-Path $PortableDir "data"
    New-Item -ItemType Directory -Path $DataDir -Force | Out-Null

    @"
Pelak-Khan Portable Data
========================

This folder travels with the portable application.
Database, captured images, processed videos, backups, and logs are stored here.
To move Pelak-Khan to another computer, copy the complete Pelak-Khan folder.
"@ | Set-Content (Join-Path $DataDir "README.txt") -Encoding UTF8

    @"
Pelak-Khan Portable v$Version
==============================

GitHub Repository:
https://github.com/mofaegh/pelak-khan

Developer:
https://github.com/mofaegh

HOW TO USE
1. Extract the ZIP completely.
2. Double-click Pelak-Khan.exe.
3. The local web application opens in your default browser.
4. Keep Pelak-Khan running in the Windows system tray while using it.
5. Exit from the tray icon when finished.

NO INSTALLATION REQUIRED
The target computer does NOT need Python, pip, PyTorch, FastAPI,
Uvicorn, OpenCV, or Ultralytics installed.

PORTABLE DATA
All writable user data is stored in the data folder beside Pelak-Khan.exe.
Copy the complete Pelak-Khan folder to move the application and its history.

RECOMMENDED SYSTEM
Windows 10/11 x64.
"@ | Set-Content (Join-Path $PortableDir "README-PORTABLE.txt") -Encoding UTF8

    $Version | Set-Content (Join-Path $PortableDir "VERSION.txt") -Encoding ASCII

    Write-Host "[8/9] Running bundled-runtime self-test..." -ForegroundColor Yellow
    $OldPythonHome = $env:PYTHONHOME
    $OldPythonPath = $env:PYTHONPATH
    $OldNoUserSite = $env:PYTHONNOUSERSITE
    try {
        $env:PYTHONHOME = $RuntimePython
        $env:PYTHONPATH = "$PortableDir;$PortableDir\src"
        $env:PYTHONNOUSERSITE = "1"
        & (Join-Path $RuntimePython "python.exe") -m apps.desktop.launcher --self-test
        if ($LASTEXITCODE -ne 0) { throw "Bundled Python runtime self-test failed." }
    }
    finally {
        $env:PYTHONHOME = $OldPythonHome
        $env:PYTHONPATH = $OldPythonPath
        $env:PYTHONNOUSERSITE = $OldNoUserSite
    }

    $SelfTest = Start-Process -FilePath $PortableExe -ArgumentList "--self-test" -WorkingDirectory $PortableDir -Wait -PassThru
    if ($SelfTest.ExitCode -ne 0) {
        throw "Pelak-Khan.exe self-test failed with exit code $($SelfTest.ExitCode)."
    }

    Write-Host "[9/9] Creating release ZIP and SHA-256..." -ForegroundColor Yellow

    # Do NOT copy the portable directory into another staging directory here.
    # Deep package trees (for example Jupyter static assets inherited from a
    # developer Python installation) can exceed the classic Windows path limit
    # when an extra staging prefix is added. The portable folder has already
    # passed its runtime self-test, so archive it directly from dist.
    Remove-Item -Force $ZipPath -ErrorAction SilentlyContinue

    $TarCommand = Get-Command "tar.exe" -ErrorAction SilentlyContinue
    if ($TarCommand) {
        & $TarCommand.Source -a -c -f $ZipPath -C $DistDir "Pelak-Khan"
        if ($LASTEXITCODE -ne 0) {
            throw "tar.exe failed while creating the portable ZIP."
        }
    }
    else {
        # Fallback for unusual Windows environments without bsdtar.
        Compress-Archive -Path $PortableDir -DestinationPath $ZipPath -CompressionLevel Optimal -Force
    }

    if (-not (Test-Path $ZipPath)) { throw "Portable ZIP was not generated." }

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
    Write-Host ""
    Write-Host "Architecture: bundled relocatable Python runtime (AI libraries are NOT frozen by PyInstaller)." -ForegroundColor DarkGreen
    Write-Host "IMPORTANT: Test the ZIP on another Windows 10/11 x64 PC without Python before publishing." -ForegroundColor Yellow
}
finally {
    Pop-Location
}
