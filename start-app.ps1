$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot
$pythonPath = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (!(Test-Path -LiteralPath $pythonPath)) {
    throw "Create .venv and install requirements.txt first. See README.md."
}
Write-Host "Open http://127.0.0.1:8000 (leave this terminal running)."
& $pythonPath -m uvicorn app.main:app --host 127.0.0.1 --port 8000
