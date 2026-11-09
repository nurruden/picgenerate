# -*- mode: python ; coding: utf-8 -*-
# 在 Windows 上于项目根目录执行：pyinstaller packaging/generatepic.spec

a = Analysis(
    ["generatepic/web.py"],
    pathex=["."],
    binaries=[],
    datas=[
        ("generatepic/templates/index.html", "generatepic/templates"),
        (".env.example", "."),
    ],
    hiddenimports=["PIL.Image", "PIL.ImageOps"],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="generatepic",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="generatepic",
)
