"""Supplemental data needed for Lezargus is stored here.

This directory comprises of all types of data which is required for Lezargus.
No code should be placed in directory. To read and write the data from this
directory, see lezargus/library/data.py. We have this information in a
__init__.py file just so it is conveniently traced by the documentation build
script.

The following conventions are used for the data files:

- CSV files : We use pipe-delimitated files. Pipes form natural column line for
              quick reading. Comments are delimitated by # and serve to be a
              header for documentation.
- FITS files : FITS files listed are written by and loaded by the Lezargus
               data containers. The FITS files within this directory should
               only be spectra, image, or cube data. Any other data is likely
               better in a different file format.
"""
