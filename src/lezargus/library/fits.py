"""FITS file reading, writing, and other manipulations."""

import copy
import os

import astropy.io.fits
import astropy.table
import numpy as np

from lezargus import library
from lezargus.library import hint
from lezargus.library import logging

# This is order and defaults for the header parameters relevant to Lezargus.
# It is structured as {key:(default, comment)}.
# See TODO for more information on the headers.
_LEZARGUS_HEADER_KEYWORDS_DICTIONARY = {
    # The beginning keyword to notate the start of Lezargus.
    "LZ_BEGIN": (True, "LZ: True if Lezargus processed."),
    # Metadata on the format of the cube itself.
    "LZ_FITSF": (None, "LZ: FITS cube format."),
    "LZ_WTABN": ("WCS-TAB", "LZ: WCS bintable extension name."),
    "LZ_UIMGN": ("UNCERTAINTY", "LZ: Uncertainty image extension name."),
    "LZ_MIMGN": ("MASK", "LZ: Mask image extension name."),
    "LZ_FIMGN": ("FLAGS", "LZ: Flag image extension name."),
    # Instrument information for whatever data Lezargus is reducing.
    "LZI_INST": (None, "LZ: Specified instrument."),
    "LZI_PXPS": (None, "LZ: Pixel plate scale, deg/pix."),
    "LZI_SLPS": (None, "LZ: Slice plate scale, deg/slice."),
    # Information about the object itself.
    "LZO_NAME": (None, "LZ: Object name."),
    "LZOPK__X": (None, "LZ: PSF peak X index."),
    "LZOPK__Y": (None, "LZ: PSF peak Y index."),
    "LZOPK_RA": (None, "LZ: PSF peak RA value, degrees."),
    "LZOPKDEC": (None, "LZ: PSF peak DEC value, degrees."),
    "LZO_ROTA": (None, "LZ: Rotation angle, degrees"),
    # Units on the data.
    "LZDWUNIT": (None, "LZ: The wavelength unit."),
    "LZDFUNIT": (None, "LZ: The flux/data unit."),
    "LZDUUNIT": (None, "LZ: The uncertainty unit."),
    # The world coordinate system entries.
    "LZWBEGIN": (False, "LZ: Begin WCS; True if present."),
    "LZW__END": (None, "LZ: End WCS entries."),
    # The ending keyword to notate the end of Lezargus.
    "LZ___END": (False, "LZ: True if Lezargus finished."),
}


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
        logging.warning(
            warning_type=logging.DataLossWarning,
            message=(
                "Non-empty data is detected for the FITS file {f}, only the"
                " header is being read and processed.".format(f=filename)
            ),
        )
    return header


def read_lezargus_fits_file(
    filename: str,
) -> tuple[
    hint.Header,
    hint.ndarray,
    hint.ndarray,
    hint.ndarray,
    hint.Unit,
    hint.Unit,
    hint.ndarray,
    hint.ndarray,
]:
    """Read in a Lezargus fits file.

    This function reads in a Lezargus FITS file and parses it based on the
    convention of Lezargus. See TODO for the specification. However, we do
    not construct the actual classes here and instead leave that to the class
    reader and writers of the container themselves so we can reuse error
    reporting code there.

    In general, it is advisable to use the reading and writing class
    functions of the container instance you want.

    Parameters
    ----------
    filename : str
        The filename of the FITS file to read.

    Returns
    -------
    header : Header
        The header of the Lezargus FITS file.
    wavelength : Array
        The wavelength information of the file.
    data : Array
        The data array of the Lezargus FITS file.
    uncertainty : Array
        The uncertainty in the data.
    wavelength_unit : Unit
        The unit of the wavelength array.
    data_unit : Unit
        The unit of the data.
    mask : Array
        The mask of the data.
    flags : Array
        The noted flags for each of the data points.
    """
    # We first need to check if the file even exists to read.
    if not os.path.isfile(filename):
        logging.critical(
            critical_type=logging.FileError,
            message=(
                "We cannot read the Lezargus FITS file {fl}, it does not exist."
                .format(fl=filename)
            ),
        )
    else:
        logging.info(
            message=f"Reading Lezargus FITS file {filename}.",
        )

    # This is a small wrapper function to make

    # Opening the file itself.
    with astropy.io.fits.open(filename) as raw_hdul:
        hdul = copy.deepcopy(raw_hdul)
        # The header information that we actually care about is in the primary
        # extension.
        header = hdul["PRIMARY"].header
        # The wavelength information is kept in the wavelength extension.
        # For the wavelength unit, we try Lezargus input first, then FITS
        # standard.
        wave_table = hdul[header["LZ_WTABN"]]
        wavelength = np.ravel(wave_table.data["WAVELENGTH"])
        wavelength_unit = header.get("LZ_WUNIT", None)
        if wavelength_unit is None:
            wavelength_unit = header.get("CUNIT3", None)
        # The data is stored in the primary extension. The Lezargus axis
        # convention and some visualization conventions have the axis reversed;
        # we convert between these.
        # For the data unit, we try Lezargus input first, then FITS
        # standard.
        data = hdul["PRIMARY"].data.T
        data_unit = header.get("LZ_FUNIT", None)
        if data_unit is None:
            data_unit = header.get("BUNIT", None)
        # The uncertainty is stored in its own extension, We transform it like
        # the data itself.
        uncertainty = hdul[header["LZ_UIMGN"]].data.T
        # Masks and flags are stored in their own extensions as well.
        mask = hdul[header["LZ_MIMGN"]].data.T
        flags = hdul[header["LZ_FIMGN"]].data.T
    # All done.
    return (
        header,
        wavelength,
        data,
        uncertainty,
        wavelength_unit,
        data_unit,
        mask,
        flags,
    )


def write_lezargus_fits_file(
    filename: str,
    header: hint.Header,
    wavelength: hint.ndarray,
    data: hint.ndarray,
    uncertainty: hint.ndarray,
    wavelength_unit: hint.Unit,
    data_unit: hint.Unit,
    uncertainty_unit: hint.Unit,
    mask: hint.ndarray,
    flags: hint.ndarray,
    overwrite: bool = False,
) -> None:
    """Write to a Lezargus fits file.

    This function reads in a Lezargus FITS file and parses it based on the
    convention of Lezargus. See TODO for the specification. However, we do
    not construct the actual classes here and instead leave that to the class
    reader and writers of the container themselves so we can reuse error
    reporting code there.

    In general, it is advisable to use the reading and writing class
    functions of the container instance you want.

    Parameters
    ----------
    filename : str
        The filename of the FITS file to write to.
    header : Header
        The header of the Lezargus FITS file.
    wavelength : Array
        The wavelength information of the file.
    data : Array
        The data array of the Lezargus FITS file.
    uncertainty : Array
        The uncertainty in the data.
    wavelength_unit : Unit
        The unit of the wavelength array.
    data_unit : Unit
        The unit of the data.
    uncertainty_unit : Unit
        The unit of the uncertainty of the data.
    mask : Array
        The mask of the data.
    flags : Array
        The noted flags for each of the data points.
    overwrite : bool, default = False
        If True, overwrite the file upon conflicts.

    Returns
    -------
    None
    """
    # We test if the file already exists.
    if os.path.isfile(filename):
        if overwrite:
            logging.warning(
                warning_type=logging.FileWarning,
                message=(
                    "The FITS file {fl} already exists, overwriting as"
                    " overwrite is True.".format(fl=filename)
                ),
            )
        else:
            logging.critical(
                critical_type=logging.FileError,
                message=(
                    "The FITS file {fl} already exists. Overwrite is False."
                    .format(fl=filename)
                ),
            )

    # We first compile the header. The unit information is kept in the header
    # as well.
    header = astropy.io.fits.Header(header)
    lezargus_header = create_lezargus_fits_header(
        header=header,
        entries={
            "LZDWUNIT": wavelength_unit,
            "LZDFUNIT": data_unit,
            "LZDUUNIT": uncertainty_unit,
        },
    )
    # We purge the old header of all Lezargus keys as we will add them back
    # in bulk in order. We do not want duplicate cards.
    for keydex in lezargus_header:
        header.remove(keydex, ignore_missing=True, remove_all=True)
    header.extend(lezargus_header, update=True)

    # First we write the main data to the array.
    data_hdu = astropy.io.fits.PrimaryHDU(data, header=header)
    # Now the WCS binary table, most relevant for the wavelength axis. Special
    # care must be made to format the data correctly. Namely, the wavelength
    # index and axis must all fit in a row of a column; see TODO.
    n_wave = len(wavelength)
    wave_index = astropy.io.fits.Column(
        name="WAVEINDEX",
        array=np.arange(n_wave),
        format=f"{n_wave}J",
        dim=f"(1,{n_wave})",
    )
    wave_value = astropy.io.fits.Column(
        name="WAVELENGTH",
        array=wavelength,
        format=f"{n_wave}E",
        dim=f"(1,{n_wave})",
    )
    wcstab_hdu = astropy.io.fits.BinTableHDU.from_columns(
        [wave_index, wave_value],
        name=header["LZ_WTABN"],
    )
    # The uncertainty of the observation stored in its own extension as well.
    uncertainty_hdu = astropy.io.fits.ImageHDU(
        uncertainty.T,
        name=header["LZ_UIMGN"],
    )
    # The mask and flags are also stored in their own HDUs.
    mask_hdu = astropy.io.fits.ImageHDU(mask.T, name=header["LZ_MIMGN"])
    flags_hdu = astropy.io.fits.ImageHDU(flags.T, name=header["LZ_FIMGN"])

    # Compiling it all together and writing it to disk.
    hdul = astropy.io.fits.HDUList(
        [data_hdu, wcstab_hdu, uncertainty_hdu, mask_hdu, flags_hdu],
    )
    hdul.writeto(filename, overwrite=overwrite)


def create_lezargus_fits_header(
    header: hint.Header,
    entries: dict = None,
) -> hint.Header:
    """Create a Lezargus header.

    This function creates an ordered Lezargus header from a header containing
    both Lezargus keywords and non-Lezargus keywords. We only include the
    relevant headers. WCS header information is also extracted and added as
    we consider it within our domain even though it does not follow the
    keyword naming convention (as WCS keywords must follow WCS convention).

    Additional header entries may be provided as a last-minute overwrite. We
    also operate on a copy of the header to prevent conflicts.

    Parameters
    ----------
    header : Astropy Header
        The header which the entries will be added to.
    entries : dictionary, default = None
        The new entries to the header. By default, None means nothing is
        to be overwritten at the last minute.

    Returns
    -------
    lezargus_header : Astropy Header
        The header which Lezargus entries have been be added to. The order
        of the entries are specified.
    """
    # Working on a copy of the header just in case.
    header_copy = copy.deepcopy(header)
    lezargus_header = astropy.io.fits.Header()
    # Type checking and providing the default as documented.
    entries = dict(entries) if entries is not None else {}

    # Defaults values are used, unless overwritten by the provided entries or
    # the provided header, in that order.
    for keydex, itemdex in _LEZARGUS_HEADER_KEYWORDS_DICTIONARY.items():
        # Extracting the default values and the comment.
        defaultdex, commentdex = itemdex
        # We attempt to get a value, either from the supplied header or the
        # entries provided, to override our default.
        if keydex in entries:
            # We first check for a new value provided.
            valuedex = entries[keydex]
        elif keydex in header_copy:
            # Then if a value already existed in the old header, there is
            # nothing to change or a default to add.
            valuedex = header_copy[keydex]
        else:
            # Otherwise, we just use the default.
            valuedex = defaultdex

        # We type check as FITS header files are picky about the object types
        # they get FITS headers really only support some specific basic types.
        valuedex = (
            library.conversion.convert_to_allowable_fits_header_data_types(
                input_data=valuedex,
            )
        )
        lezargus_header[keydex] = (valuedex, commentdex)

    # We construct the WCS header from the Lezargus header if one does not
    # already exist. We insert it in the WCS section of the header.
    if not header.get("LZWBEGIN", False):
        logging.info(
            message=(
                "A WCS header is already present, skipping unnecessary"
                " extraction and instantiation."
            ),
        )
    else:
        # We put the WCS header into the correct ordered location. We expect
        # that the WCS headers are generated by these functions so the order
        # is relatively self-contained.
        wcs_header = create_wcs_header_from_lezargus_header(header=header_copy)
        for keydex in wcs_header:
            # We needed to break it up like this so we can also grab the
            # header comments, which may or may not exist for any given card.
            valuedex = wcs_header[keydex]
            commentdex = wcs_header.comments[keydex]
            # We place it in the order we expect, but we want to avoid
            # duplicate cards where possible.
            if keydex in lezargus_header:
                # The key for this card already exists, just replace it inplace.
                lezargus_header[keydex] = (valuedex, commentdex)
            else:
                # We want to put it within the WCS section of Lezargus.
                lezargus_header.insert(
                    "LZW__END",
                    (keydex, valuedex, commentdex),
                    after=False,
                )

    # All done.
    return lezargus_header


def create_wcs_header_from_lezargus_header(header: hint.Header) -> hint.Header:
    """Create WCS header keywords from Lezargus header.

    See the FITS standard for more information.

    Parameters
    ----------
    header : Header
        The Lezargus header from which we will derive a WCS header from.

    Returns
    -------
    wcs_header : Header
        The WCS header.
    """
    # If the header provided is not a Lezargus header, we cannot extract
    # the WCS information from it.
    if not header.get("LZ_BEGIN", False):
        logging.error(
            error_type=logging.InputError,
            message=(
                "A WCS header cannot be reasonably derived from a header"
                " without Lezargus keys, this is likely to fail."
            ),
        )

    # Getting the WCS data from the header...
    # If there is already WCS info present in the Lezargus header, then we
    # just extract it from the header.
    has_wcs = header.get("LZWBEGIN", False)
    if has_wcs:
        logging.info(
            message=(
                "Inputted Lezargus header already has a WCS, extracting it."
            ),
        )
    # Coordinate standard.
    wcsaxes = header.get("WCSAXES", None) if has_wcs else 3
    radesys = header.get("RADESYS", None) if has_wcs else "ICRS"
    # The WCS RA axis information.
    ctype1 = header.get("CTYPE1", None) if has_wcs else "RA---TAN"
    crpix1 = (
        header.get("CRPIX1", None) if has_wcs else header.get("LZOPK__X", None)
    )
    crval1 = (
        header.get("CRVAL1", None) if has_wcs else header.get("LZOPK_RA", None)
    )
    cunit1 = header.get("CUNIT1", None) if has_wcs else "deg"
    cdelt1 = (
        header.get("CDELT1", None) if has_wcs else header.get("LZI_PXSC", None)
    )
    # The WCS DEC axis information.
    ctype2 = header.get("CTYPE2", None) if has_wcs else "DEC--TAN"
    crpix2 = (
        header.get("CRPIX2", None) if has_wcs else header.get("LZOPK__Y", None)
    )
    crval2 = (
        header.get("CRVAL2", None) if has_wcs else header.get("LZOPKDEC", None)
    )
    cunit2 = header.get("CUNIT2", None) if has_wcs else "deg"
    cdelt2 = (
        header.get("CDELT2", None) if has_wcs else header.get("LZI_SLPS", None)
    )
    # Rotation is stored via the second axis WCS rotation parameter.
    crota2 = (
        header.get("CROTA2", None) if has_wcs else header.get("LZO_ROTA", None)
    )
    # The wavelength WCS is constructed using a table format. We just create
    # the metadata for it here.
    ctype3 = header.get("CTYPE3", None) if has_wcs else "WAVE-TAB"
    crpix3 = header.get("CRPIX3", None) if has_wcs else 1
    crval3 = header.get("CRVAL3", None) if has_wcs else 1
    cdelt3 = header.get("CDELT3", None) if has_wcs else 1
    cunit3 = (
        header.get("CUNIT3", None) if has_wcs else header.get("LZDWUNIT", None)
    )
    ps3_0 = header.get("PS3_0", None) if has_wcs else "WCS-TAB"
    ps3_1 = header.get("PS3_1", None) if has_wcs else "WAVELENGTH"
    ps3_2 = header.get("PS3_2", None) if has_wcs else "WAVEINDEX"

    # We start with a blank header.
    wcs_header = astropy.io.fits.Header()
    # Adding the data, along with the header comments.
    # For more information about these specific keywords, specific to
    # Lezargus, see TODO.
    wcs_header["WCSAXES"] = (wcsaxes, "WCS axis count.")
    wcs_header["RADESYS"] = (radesys, "Reference frame.")
    wcs_header["CTYPE1"] = (ctype1, "Axis 1 type code.")
    wcs_header["CRPIX1"] = (crpix1, "Axis 1 reference pixel.")
    wcs_header["CRVAL1"] = (crval1, "Axis 1 reference value.")
    wcs_header["CUNIT1"] = (cunit1, "Axis 1 unit.")
    wcs_header["CDELT1"] = (cdelt1, "Axis 1 step-size; unit/pix.")
    wcs_header["CTYPE2"] = (ctype2, "Axis 2 type code.")
    wcs_header["CRPIX2"] = (crpix2, "Axis 2 reference pixel.")
    wcs_header["CRVAL2"] = (crval2, "Axis 2 reference value.")
    wcs_header["CUNIT2"] = (cunit2, "Axis 2 unit.")
    wcs_header["CDELT2"] = (cdelt2, "Axis 2 step-size; unit/pix.")
    wcs_header["CROTA2"] = (crota2, "Axis 2 (image) rotation.")
    wcs_header["CTYPE3"] = (ctype3, "Axis 3 type code.")
    wcs_header["CRPIX3"] = (crpix3, "Axis 3 reference pixel.")
    wcs_header["CRVAL3"] = (crval3, "Axis 3 reference value.")
    wcs_header["CDELT3"] = (cdelt3, "Axis 3 step-size.")
    wcs_header["CUNIT3"] = (cunit3, "Axis 3 unit.")
    wcs_header["PS3_0"] = (ps3_0, "Axis 3, lookup table extension.")
    wcs_header["PS3_1"] = (ps3_1, "Axis 3, table column name.")
    wcs_header["PS3_2"] = (ps3_2, "Axis 3, index array column name.")
    # All done.
    return wcs_header