#!/usr/bin/env pwsh

# This forge script builds the documentation stack for Lezargus. We 
# leverage Sphinx to build the main systems.

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
