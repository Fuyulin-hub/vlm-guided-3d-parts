$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    New-Item -ItemType Directory -Force -Path 'build' | Out-Null
    & xelatex -interaction=nonstopmode -halt-on-error -output-directory=build main.tex
    if ($LASTEXITCODE -ne 0) { throw 'XeLaTeX failed. Inspect build/main.log.' }
    & bibtex build/main
    if ($LASTEXITCODE -ne 0) { throw 'BibTeX failed. Inspect build/main.blg.' }
    for ($pass = 0; $pass -lt 2; $pass++) {
        & xelatex -interaction=nonstopmode -halt-on-error -output-directory=build main.tex
        if ($LASTEXITCODE -ne 0) { throw 'XeLaTeX failed. Inspect build/main.log.' }
    }
    Copy-Item -LiteralPath 'build/main.pdf' -Destination 'paper_ijcv_draft.pdf' -Force
    Write-Output 'Built paper_ijcv_draft.pdf'
} finally { Pop-Location }
