# First, entering into the auxiliary directory to process the data.
Set-Location "./auxiliary"

# We want to have pretty files.
black *.ipynb --ipynb

# We remove all of the files before hand.
Remove-Item "./products/*" -Recurse -Force

# Generating the data files, if the script fails, we don't want to continue 
# and mess up the other files.
# We use `if ( -not ($?)) { Exit }` so the script exits on failure.
python -m jupyter nbconvert --to notebook --execute "./constants.ipynb"
if ( -not ($?)) { Exit }
python -m jupyter nbconvert --to notebook --execute "./dispersion_spots.ipynb"
if ( -not ($?)) { Exit }
python -m jupyter nbconvert --to notebook --execute "./standard_stars.ipynb"
if ( -not ($?)) { Exit }
python -m jupyter nbconvert --to notebook --execute "./photometric_filters.ipynb"
if ( -not ($?)) { Exit }
python -m jupyter nbconvert --to notebook --execute "./psg_data.ipynb"
if ( -not ($?)) { Exit }
#python -m jupyter nbconvert --to notebook --execute "./gemini_data.ipynb"
if ( -not ($?)) { Exit }
python -m jupyter nbconvert --to notebook --execute "./irtf_telescope.ipynb"
if ( -not ($?)) { Exit }
python -m jupyter nbconvert --to notebook --execute "./optic_efficiencies.ipynb"
if ( -not ($?)) { Exit }

# Copy the files.
Robocopy.exe "./products/" "./../src/lezargus/data/_files" /MIR

# Remove all other unimportant files.
Remove-Item "./*.nbconvert.ipynb" -Recurse -Force

# And going back to the main directory.
Set-Location ".."