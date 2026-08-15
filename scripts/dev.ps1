<#
.SYNOPSIS
    Developer command helper for DocuResearch AI (Windows PowerShell equivalent of the Makefile).
.EXAMPLE
    ./scripts/dev.ps1 venv
    ./scripts/dev.ps1 install-dev
    ./scripts/dev.ps1 test
    ./scripts/dev.ps1 mock
#>
param(
    [Parameter(Position = 0, Mandatory = $true)]
    [ValidateSet("venv", "install", "install-dev", "test", "lint", "typecheck", "mock", "run", "clean")]
    [string]$Command,

    [string]$Topic = "The rise of India's smartphone revolution"
)

switch ($Command) {
    "venv"        { python -m venv .venv }
    "install"     { pip install -r requirements.txt }
    "install-dev" { pip install -r requirements-dev.txt }
    "test"        { pytest -v }
    "lint"        { ruff check src tests }
    "typecheck"   { mypy src }
    "mock"        { python -m docuresearch --mock }
    "run"         { python -m docuresearch --topic "$Topic" }
    "clean" {
        Get-ChildItem -Recurse -Directory -Include "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache" |
            Remove-Item -Recurse -Force
    }
}
