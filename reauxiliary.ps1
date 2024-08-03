# First, entering into the auxiliary directory to process the data.
Set-Location "./auxiliary"

# We want to have pretty files.
black *.ipynb --ipynb

# We remove all of the files before hand.
Remove-Item "./products/*" -Recurse -Force

# Generating the data files.
python -m jupyter nbconvert --to notebook --execute "./standard_stars.ipynb"
python -m jupyter nbconvert --to notebook --execute "./photometric_filters.ipynb"
python -m jupyter nbconvert --to notebook --execute "./psg_data.ipynb"
#python -m jupyter nbconvert --to notebook --execute "./gemini_data.ipynb"
python -m jupyter nbconvert --to notebook --execute "./irtf_telescope.ipynb"

# Copy the files.
Robocopy.exe "./products/" "./../src/lezargus/data/_files" /MIR

# Remove all other unimportant files.
Remove-Item "./*.nbconvert.ipynb" -Recurse -Force

# And going back to the main directory.
Set-Location ".."