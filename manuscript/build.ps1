$ErrorActionPreference = 'Stop'
$paperRoot = $PSScriptRoot
$texCommand = Get-Command xelatex -ErrorAction SilentlyContinue
$bibCommand = Get-Command bibtex -ErrorAction SilentlyContinue
if (-not $texCommand) {
    $texExecutable = Join-Path $env:LOCALAPPDATA 'Programs/MiKTeX/miktex/bin/x64/xelatex.exe'
} else { $texExecutable = $texCommand.Source }
if (-not $bibCommand) {
    $bibExecutable = Join-Path $env:LOCALAPPDATA 'Programs/MiKTeX/miktex/bin/x64/bibtex.exe'
} else { $bibExecutable = $bibCommand.Source }
if (-not (Test-Path -LiteralPath $texExecutable)) { throw 'XeLaTeX was not found.' }
if (-not (Test-Path -LiteralPath $bibExecutable)) { throw 'BibTeX was not found.' }
Push-Location $paperRoot
try {
    New-Item -ItemType Directory -Force -Path 'build' | Out-Null
    & $texExecutable -interaction=nonstopmode -halt-on-error -output-directory=build main.tex
    if ($LASTEXITCODE -ne 0) { throw 'First XeLaTeX pass failed. See build/main.log.' }
    & $bibExecutable build/main
    if ($LASTEXITCODE -ne 0) { throw 'BibTeX failed. See build/main.blg.' }
    for ($paperPass = 0; $paperPass -lt 2; $paperPass++) {
        & $texExecutable -interaction=nonstopmode -halt-on-error -output-directory=build main.tex
        if ($LASTEXITCODE -ne 0) { throw 'XeLaTeX failed. See build/main.log.' }
    }
    Copy-Item -LiteralPath 'build/main.pdf' -Destination 'paper_chinese_draft.pdf' -Force
    Write-Output 'Built paper_chinese_draft.pdf'
} finally { Pop-Location }
