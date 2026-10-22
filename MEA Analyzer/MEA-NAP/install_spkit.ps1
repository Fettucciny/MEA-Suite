# =============================================================================
#  install_spkit.ps1  --  install spkit into the MEA-NAP venv and verify it
# -----------------------------------------------------------------------------
#  Run it one of these ways (PowerShell):
#     powershell -ExecutionPolicy Bypass -File .\install_spkit.ps1
#  or right-click the file  ->  "Run with PowerShell".
#
#  If you already know your venv python (the SAME python.exe you launch the GUI
#  with), just set $py on the next line and it skips the auto-search.
# =============================================================================

$ErrorActionPreference = "Stop"

# ---- 1. venv python: set it here, or leave "" to auto-search -----------------
$py = ""

if (-not $py) {
    Write-Host "Searching for your venv python (.venv\Scripts\python.exe)..."
    $roots = @(
        "C:\Users\Yikan\Nextcloud2\MEA Data\New MEA 06172026",
        "C:\Users\Yikan\Nextcloud2\MEA-NAP-main",
        "C:\Users\Yikan"
    )
    foreach ($r in $roots) {
        if (Test-Path $r) {
            $hit = Get-ChildItem -Path $r -Recurse -Depth 4 -Filter "python.exe" `
                    -ErrorAction SilentlyContinue |
                   Where-Object { $_.FullName -match "\\(\.venv|venv|env)\\Scripts\\python\.exe$" } |
                   Select-Object -First 1
            if ($hit) { $py = $hit.FullName; break }
        }
    }
}

if (-not $py -or -not (Test-Path $py)) {
    Write-Host ""
    Write-Host "Could not auto-find the venv python." -ForegroundColor Yellow
    Write-Host "Open this script, set the line:  `$py = 'C:\path\to\.venv\Scripts\python.exe'"
    Write-Host "(use the same python.exe you launch mea_nap_gui.py with) and re-run."
    exit 1
}
Write-Host "Using python: $py" -ForegroundColor Cyan

# ---- 2. install spkit --------------------------------------------------------
& $py -m pip install --upgrade spkit

# ---- 3. verify it imported and the MEA submodule loads -----------------------
& $py -c "import spkit, spkit.mea; print('OK - spkit', spkit.__version__, '| spkit.mea loaded')"

Write-Host ""
Write-Host "Done. If you saw 'OK - spkit ...' above, the SpKit EGM tab is ready to use." -ForegroundColor Green
