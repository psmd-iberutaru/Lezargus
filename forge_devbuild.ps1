#!/usr/bin/env pwsh

# This forge script builds a new development version of Lezargus. 
# Warning!: For the sake of simplicity, it also installs this new development 
# version of Lezargus onto the system. This forge script is mostly for
# testing and development purposes. A complete distribution-ready build is 
# invoked using `forge_lezargus.ps1`.

# We need to make sure have the right requirements.
./forge_requirements.ps1

# Using Hatch, we clean up the current environment.
hatch clean

# Upgrading to a new dev version and building it.
hatch version dev
hatch build

# Installing the new version. 
# We need the new filename as Windows does not expand wildcards natively.
$glob = Get-ChildItem .\dist -Recurse -Filter "lezargus-*.whl"
$wheel_file = $glob.FullName
# Installing
pip install $wheel_file

