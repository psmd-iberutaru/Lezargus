#!/usr/bin/env pwsh

# This forge script builds the documentation stack for Lezargus for the 
# LaTeX engine. As this creates a lot of files, requires the need for a local
# LaTeX install, and is seldom used, we split it up into its own forge script
# from the main documentation script.

# In order to build the LaTeX form of the documentation, the main form of 
# the documentation needs to be built first.
./forge_documentation.ps1

# Ensuring that a local LaTeX install is present. This is really quick and 
# dirty.
$PdfTeXVersion = pdftex --version
if ([string]::IsNullOrEmpty($PdfTeXVersion)) {
    Write-Error -Message "LaTeX (TeXLive) is likely installed. Please install TeXLive"
}
else {
    Write-Host "Using $PythonVersion"
}

# Ensuring that Sphinx is installed.
pip install sphinx

# Building the LaTeX documentation.
sphinx-build -M latexpdf ./docs/source/ ./docs/build/latex/
