#!/usr/bin/env pwsh

# This forge script builds needed auxiliary data files which Lezargus uses.
# As most of the files are generated using Jupyter Notebooks, we just 
# run each of the note books.

# We stop if we encounter an error.
$ErrorActionPreference = "Stop"

# We need to make sure we have the requirements to do this first.
./forge_requirements.ps1
# Jupyter requirements.
pip install jupyter nbconvert 
# Formatting requirements.
pip install "black[jupyter]"
# Actual script requirements, in excess of Lezargus.
pip install pycpd

# First, entering into the auxiliary directory to process the data.
Set-Location "./auxiliary"

# We want to have pretty files.
black *.ipynb --ipynb

# We remove all of the files before hand.
# Remove-Item "./products/" -Recurse -Force

# Generating the data files, if the script fails, we don't want to continue 
# and mess up the other files.
python -m jupyter nbconvert --to notebook --execute "./atmosphere_psg.ipynb"
if ( -not ($?)) { Exit }
python -m jupyter nbconvert --to notebook --execute "./constants.ipynb"
if ( -not ($?)) { Exit }
#python -m jupyter nbconvert --to notebook --execute "./gemini_data.ipynb"
if ( -not ($?)) { Exit }
python -m jupyter nbconvert --to notebook --execute "./irtf_optics.ipynb"
if ( -not ($?)) { Exit }
python -m jupyter nbconvert --to notebook --execute "./photometric_filters.ipynb"
if ( -not ($?)) { Exit }
python -m jupyter nbconvert --to notebook --execute "./spectre_calibrations.ipynb"
if ( -not ($?)) { Exit }
python -m jupyter nbconvert --to notebook --execute "./spectre_detectors.ipynb"
if ( -not ($?)) { Exit }
python -m jupyter nbconvert --to notebook --execute "./spectre_dispersion.ipynb"
if ( -not ($?)) { Exit }
python -m jupyter nbconvert --to notebook --execute "./spectre_optics.ipynb"
if ( -not ($?)) { Exit }
python -m jupyter nbconvert --to notebook --execute "./standard_stars.ipynb"
if ( -not ($?)) { Exit }


# Copy the files.
Copy-Item -Path "./products/*" -Destination "./../src/lezargus/data/_files/" -Recurse

# Remove all other unimportant files.
Remove-Item "./*.nbconvert.ipynb" -Recurse -Force

# And going back to the main directory.
Set-Location ".."