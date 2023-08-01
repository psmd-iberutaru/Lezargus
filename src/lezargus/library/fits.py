"""FITS file reading, writing, and other manipulations."""

import copy

import numpy as np
import astropy.io.fits
import astropy.table

from lezargus import library
from lezargus.library import logging
from lezargus.library import hint

def read_fits_header(filename: str, extension: int | str = 0) -> hint.Header:
    """Read a FITS file header.
    
    This reads the header of fits files only. This should be used only if
    there is no data. Really, this is just a wrapper around Astropy, but it 
    is made for consistency and to avoid the usage of the convince functions.

    Parameters
    ----------
    filename : str
        The filename that the fits image file is at.
    extension : int or str, default = 0
        The fits extension that is desired to be opened.

    Returns
    -------
    header : Astropy Header
        The header of the fits file.
    """
    with astropy.io.fits.open(filename) as hdul:
        hdu = hdul[extension].copy()
        header = hdu.header
        data = hdu.data
    # Check that the data does not exist, so the data read should be none.
    if data is not None:
        logging.warning(warning_type=logging.DataLossWarning, message="Non-empty data is detected for the FITS file {f}, only the header is being read and processed.".format(f=filename))
    return header


def read_fits_image_file(
    filename: str, extension: int | str = 0
) -> tuple[hint.Header, hint.Array | None]:
    """Read a FITS image file.
    
    This reads fits files, assuming that the fits file is an image. It is a
    wrapper function around the Astropy functions.

    Parameters
    ----------
    filename : str
        The filename that the fits image file is at.
    extension : int or str, default = 0
        The fits extension that is desired to be opened.

    Returns
    -------
    header : Astropy Header
        The header of the fits file.
    data : Array
        The data image of the fits file. If the reading of the array does not
        produce a data array, it returns None.
    """
    with astropy.io.fits.open(filename) as hdul:
        hdu = hdul[extension].copy()
        header = hdu.header
        data = hdu.data

    # Check that the data really is an image.
    if not isinstance(data, np.ndarray):
        logging.warning(warning_type=logging.DataLossWarning, message="The FITS file {f} is expected to be an image-based FITS file, but its data is not that of an image. The return data is None.".format(f=filename))

    return header, data


def read_fits_table_file(
    filename: str, extension: int|str = 0
) -> tuple[hint.Header, hint.Table | None]:
    """This reads fits files, assuming that the fits file is a binary table.
    It is a wrapper function around the astropy functions.

    Parameters
    ----------
    filename : str
        The filename that the fits image file is at.
    extension : int or str, default = 0
        The fits extension that is desired to be opened.

    Returns
    -------
    header : Astropy Header
        The header of the fits file.
    table : Astropy Table
        The data table of the fits file. If the reading of the array does not
        produce a data table, it returns None.
    """
    with astropy.io.fits.open(filename) as hdul:
        hdu = hdul[extension].copy()
        header = hdu.header
        data = hdu.data
    # Check that the data really is table-like.
    if not isinstance(data, (astropy.table.Table, astropy.io.fits.FITS_rec)):
        logging.warning(warning_type=logging.DataLossWarning, message="The FITS file {f} is expected to be a table-based FITS file, but its data is not that of an table. The returned table is None.".format(f=filename))
        table = None
    else:
        # The return is specified to be an astropy table.
        table = astropy.table.Table(data)
    return header, table


def write_fits_image_file(
    filename: str, header: hint.Header, data: hint.Array, overwrite: bool = False
) -> None:
    """Write FITS image file.
    
    This writes fits image files to disk. Acting as a wrapper around the
    fits functionality of astropy.

    Parameters
    ----------
    filename : str
        The filename that the fits image file will be written to.
    header : Astropy Header
        The header of the fits file.
    data : array-like
        The data image of the fits file.
    overwrite : boolean, default = False
        Decides if to overwrite the file if it already exists.

    Returns
    -------
    None
    """
    # Type checking, ensuring that this function is being used for images only.
    if not isinstance(header, (dict, astropy.io.fits.Header)):
        logging.error(error_type=logging.InputError, message="The input header for the FITS file {f} is not an appropriate header, defaulting to an empty header.".format(f=filename))
        logging.warning(warning_type=logging.DataLossWarning, message="The input header data, as follows, was ignored because it is not compatible: \n {hd}".format(hd=header))
        header = None
    if not isinstance(data, np.ndarray):
        logging.error(error_type=logging.InputError, message="The input data for the FITS file {f} is not an appropriate data array. There is nothing to save, defaulting to a FITS file without any data.".format(f=filename))
        logging.warning(warning_type=logging.DataLossWarning, message="The input data, as follows, was ignored because it is not compatible: \n {dt}".format(dt=data))
        data = None
    else:
        saving_data = np.array(data)
    # Create the image and add the header.
    hdu = astropy.io.fits.PrimaryHDU(data=saving_data, header=header)
    # Write.
    hdu.writeto(filename, overwrite=overwrite)
    return None

def write_fits_table_file(
    filename: str, header: hint.Header, table: hint.Table, overwrite: bool = False
) -> None:
    """This writes fits table files to disk. Acting as a wrapper around the
    fits functionality of Astropy.

    Parameters
    ----------
    filename : string
        The filename that the fits image file will be written to.
    header : Astropy Header
        The header of the fits file.
    table : Astropy Table
        The data table of the table file.
    overwrite : boolean, default = False
        Decides if to overwrite the file if it already exists.

    Returns
    -------
    None
    """
    # Type checking, ensuring that this function is being used for tables only.
    if not isinstance(header, (dict, astropy.io.fits.Header)):
        logging.error(error_type=logging.InputError, message="The input header for the FITS file {f} is not an appropriate header, defaulting to an empty header.".format(f=filename))
        logging.warning(warning_type=logging.DataLossWarning, message="The input header data, as follows, was ignored because it is not compatible: \n {hd}".format(hd=header))
        header = None
    if not isinstance(table, (astropy.table.Table, astropy.io.fits.FITS_rec)):
        logging.error(error_type=logging.InputError, message="The input data for the FITS file {f} is not an appropriate table. There is nothing to save, defaulting to a FITS file without any table data.".format(f=filename))
        logging.warning(warning_type=logging.DataLossWarning, message="The input data, as follows, was ignored because it is not compatible: \n {tb}".format(tb=table))
        table = None
    # Create the table data
    binary_table = astropy.io.fits.BinTableHDU(data=table, header=header)
    # Write.
    binary_table.writeto(filename, overwrite=overwrite)
    return None