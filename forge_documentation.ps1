#!/usr/bin/env pwsh

# This forge script builds the documentation stack for Lezargus. We 
# leverage Sphinx to build the main systems. If a valid LaTeX installation 
# is present, we also build the LaTeX PDF version of the documentation.
# In order for us to build the documentation, we need to build the software
# itself first. This is because of the traversal of the apidoc.

# Building the Lezargus software.
./forge_lezargus.ps1

# Requirements for Sphinx documentation.
pip install sphinx sphinx-book-theme

# Building the API documentation.
sphinx-apidoc -f -e -P -o ./docs/source/code/ ./src/lezargus/

# Building the rest of the documentation.
sphinx-build -a -b html ./docs/source/ ./docs/build/html/


# Checking if a local LaTeX install is present. This is really quick and 
# dirty and there are likely better methods to this.
$PdfTeXVersionBlob = pdftex --version
# The blob contains a lot of other version and license information we do not
# need.
$PdfTeXVersion = $PdfTeXVersionBlob[0]
if ([string]::IsNullOrEmpty($PdfTeXVersion)) {
    Write-Error -Message (
        "LaTeX (TeXLive) is not detected, likely not installed. " +
        "Required for LaTeX documentation build."
    )
}
else {
    Write-Host "Using $PdfTeXVersion"
    # Building the LaTeX documentation.
    sphinx-build -M latexpdf ./docs/source/ ./docs/build/latex/
}
