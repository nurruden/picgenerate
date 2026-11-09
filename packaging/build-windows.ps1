# 在 Windows 上、项目根目录的上一级不用管。直接运行：
#   powershell -ExecutionPolicy Bypass -File packaging\build-windows.ps1
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

python -m pip install --upgrade pip
python -m pip install pyinstaller
python -m PyInstaller packaging\generatepic.spec --noconfirm --clean

$iscc = @(
    "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe",
    "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
    "$env:ProgramFiles\Inno Setup 6\ISCC.exe"
) | Where-Object { Test-Path $_ } | Select-Object -First 1

if ($iscc) {
    & $iscc "packaging\installer.iss"
    Write-Host "安装包: dist\generatepic-setup.exe"
} else {
    Write-Host "已生成 dist\generatepic\generatepic.exe"
    Write-Host "安装 Inno Setup 6 后重新运行本脚本，才会得到 generatepic-setup.exe"
}
