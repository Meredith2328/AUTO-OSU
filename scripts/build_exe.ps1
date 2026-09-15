# Build the Windows one-folder executable and the release zip.
#
#   powershell -ExecutionPolicy Bypass -File scripts/build_exe.ps1            # CPU build (default)
#   powershell -ExecutionPolicy Bypass -File scripts/build_exe.ps1 -Venv .venv-gpu -Suffix cuda
#
# Requirements: a virtual environment with the project installed (pip install -e .[gui,build])
# and the two model files in models/ (or run `python -m autoosu --download` first).
param(
    [string]$Venv = ".venv",
    [string]$Suffix = "cpu",
    [switch]$NoZip
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$py = Join-Path $Venv "Scripts\python.exe"
if (-not (Test-Path $py)) { throw "python not found at $py" }

$version = (& $py -c "import autoosu; print(autoosu.__version__)").Trim()
Write-Host "== AUTO-OSU $version ($Suffix) with $py"

& $py -m PyInstaller --noconfirm --clean AUTO-OSU.spec
if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed" }

$dist = Join-Path $root "dist\AUTO-OSU"
New-Item -ItemType Directory -Force (Join-Path $dist "models") | Out-Null
foreach ($m in @("rhythm_v0.pt", "coord_v0.pt")) {
    $src = Join-Path $root "models\$m"
    if (Test-Path $src) { Copy-Item $src (Join-Path $dist "models\$m") -Force }
    else { Write-Warning "models\$m missing: the app will offer to download it on first run" }
}
Copy-Item (Join-Path $root "README.md") $dist -Force
Copy-Item (Join-Path $root "README.zh-CN.md") $dist -Force
Copy-Item (Join-Path $root "LICENSE") $dist -Force
Write-Host "== dist ready: $dist"

if (-not $NoZip) {
    $zip = Join-Path $root "dist\AUTO-OSU-$version-win64-$Suffix.zip"
    if (Test-Path $zip) { Remove-Item $zip }
    Compress-Archive -Path $dist -DestinationPath $zip -CompressionLevel Optimal
    $mb = [math]::Round((Get-Item $zip).Length / 1MB)
    Write-Host "== zip: $zip ($mb MB)"
}
