from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_submodules

ROOT = Path(SPECPATH).resolve().parents[1]
ICON = ROOT / "build" / "windows" / "assets" / "pelak-khan.ico"

ultra_datas, ultra_bins, ultra_hidden = collect_all("ultralytics")
tv_datas, tv_bins, tv_hidden = collect_all("torchvision")

hiddenimports = list(dict.fromkeys(
    ultra_hidden
    + tv_hidden
    + collect_submodules("uvicorn")
    + collect_submodules("pystray")
    + [
        "pelak_khan",
        "pelak_khan.runtime",
        "pelak_khan.ocr",
        "pelak_khan.tracking",
        "pelak_khan.storage",
        "pelak_khan.postprocessing",
        "apps.backend.main",
    ]
))

datas = [
    (str(ROOT / "apps" / "frontend"), "apps/frontend"),
    (str(ROOT / "models" / "runtime" / "detector_v1.pt"), "models/runtime"),
    (str(ROOT / "models" / "runtime" / "ocr_v1.pt"), "models/runtime"),
    (str(ICON), "build/windows/assets"),
] + ultra_datas + tv_datas

binaries = ultra_bins + tv_bins

analysis = Analysis(
    [str(ROOT / "apps" / "desktop" / "launcher.py")],
    pathex=[str(ROOT), str(ROOT / "src")],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["tkinter", "pytest", "IPython", "jupyter", "notebook"],
    noarchive=False,
)

pyz = PYZ(analysis.pure)

exe = EXE(
    pyz,
    analysis.scripts,
    [],
    exclude_binaries=True,
    name="Pelak-Khan",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(ICON),
)

collect = COLLECT(
    exe,
    analysis.binaries,
    analysis.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="Pelak-Khan",
)
