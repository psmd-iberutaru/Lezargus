#!/usr/bin/env pwsh

# This forge script builds installs the needed requirements for all of 
# other forge scripts in Lezargus.

# We stop if there is an error.
$ErrorActionPreference = "Stop"

# We need to make sure there is a compatible Python install.
$PythonVersion = python --version
# If there is no Python in the first place...
if ([string]::IsNullOrEmpty($PythonVersion)) {
    Write-Error -Message "Python is not installed. Please install Python 3.14+."
}
else {
    Write-Host "Using $PythonVersion"
}
# We check if the Python version is more than 3.14+ as requited.
$null, $Version = $PythonVersion.split(" ")
$MajorVersion, $MinorVersion, $PatchVersion = $Version.split(".")
if (-not ($MajorVersion -ge 3 -and $MinorVersion -ge 14)) {
    Write-Error -Message "Python version must be 3.14+. Current version $PythonVersion"
}

# We need to make sure we have all of the required Python packages.
# Python hatch can allow us to bootstrap everything else.
python -m ensurepip
pip install hatch