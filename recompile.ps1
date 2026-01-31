#!/usr/bin/env sh
echo --% >/dev/null;: ' | out-null
<#'

# Warning: The above text should never be edited. It allows this script to be 
# run on common UNIX shells and Powershell. 
# Credit to https://stackoverflow.com/a/67292076
#
# This script rebuilds and reinstalls Lezargus, for development purposes only.
# It does the following via Hatch:
# - It deletes any old build artifacts. 
# - It updates the version to the next dev level.
# - It builds the new project.
# - It force reinstalls the updated project via pip.


### ### ### ### ### ### ### ### ### ### ### ### ### ### 
# This is a shell script doing the documented procedure.
# # # # # # # # # # # # # # # # # # # # # # # # # # # # 

# Compile it...
pyinstaller --clean --collect-all lezargus  ./src/lezargus/__main__.py

# Clean up any files after.
rm __main__.spec
rm -rf ./build

# We need to exit early to prevent the Powershell section from mucking things.
exit #>



### ### ### ### ### ### ### ### ### ### ### ### ### ### ### 
# This part is a Powershell script doing the documented task.
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

