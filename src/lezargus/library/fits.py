"""FITS file reading, writing, and other manipulations."""

# isort: split
# Import required to remove circular dependencies from type checking.
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lezargus.library import hint
# isort: split

import copy
import os

import astropy.io.fits
import astropy.table
import numpy as np

import lezargus
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
    "LZIFNAME": (None, "LZ: Initial pathless filename."),
    # Information about the object itself.
    "LZO_NAME": (None, "LZ: Object name."),
    "LZOPK__X": (None, "LZ: PSF peak X index."),
    "LZOPK__Y": (None, "LZ: PSF peak Y index."),
    "LZOPK_RA": (None, "LZ: PSF peak RA value, deg."),
    "LZOPKDEC": (None, "LZ: PSF peak DEC value, deg."),
    "LZO_ROTA": (None, "LZ: Rotation angle, deg."),
    "LZO_AIRM": (None, "LZ: Airmass."),
    # Synthetic photometric magnitudes derived from the spectrum. This is a
    # helpful place to put photometry measurements.
    "LZPM_J_U": (None, "LZ: Johnson U magnitude."),
    "LZPU_J_U": (None, "LZ: Johnson U uncertainty."),
    "LZPM_J_B": (None, "LZ: Johnson B magnitude."),
    "LZPU_J_B": (None, "LZ: Johnson B uncertainty."),
    "LZPM_J_V": (None, "LZ: Johnson V magnitude."),
    "LZPU_J_V": (None, "LZ: Johnson V uncertainty."),
    "LZPM_G_G": (None, "LZ: Gaia G magnitude."),
    "LZPU_G_G": (None, "LZ: Gaia G uncertainty."),
    "LZPM_GGB": (None, "LZ: Gaia GB magnitude."),
    "LZPU_GGB": (None, "LZ: Gaia GB uncertainty."),
    "LZPM_GGR": (None, "LZ: Gaia GR magnitude."),
    "LZPU_GGR": (None, "LZ: Gaia GR uncertainty."),
    "LZPM_2_J": (None, "LZ: 2MASS J magnitude."),
    "LZPU_2_J": (None, "LZ: 2MASS J uncertainty."),
    "LZPM_2_H": (None, "LZ: 2MASS H magnitude."),
    "LZPU_2_H": (None, "LZ: 2MASS H uncertainty."),
    "LZPM_2Ks": (None, "LZ: 2MASS Ks magnitude."),
    "LZPU_2Ks": (None, "LZ: 2MASS Ks uncertainty."),
    # Units and scales on the data.
    "LZDWUNIT": (None, "LZ: The wavelength unit."),
    "LZDFUNIT": (None, "LZ: The flux/data unit."),
    "LZDUUNIT": (None, "LZ: The uncertainty unit, same as data."),
    "LZDSPECS": (None, "LZ: Spectral resolution, angstrom/pixel."),
    "LZDPIXPS": (None, "LZ: Pixel plate scale, arcsec/pixel."),
    "LZDSLIPS": (None, "LZ: Slice plate scale, arcsec/slice."),
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
                "Non-empty data is detected for the FITS file {filename}, only"
                " the header is being read and processed."
            ),
        )
    return header


def read_lezargus_fits_file(
    filename: str,
) -> tuple[
    hint.Header,
    hint.NDArray,
    hint.NDArray,
    hint.NDArray,
    hint.Unit,
    hint.Unit,
    float,
    float,
    hint.NDArray,
    hint.NDArray,
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
    wavelength : ndarray
        The wavelength information of the file.
    data : ndarray
        The data array of the Lezargus FITS file.
    uncertainty : ndarray
        The uncertainty in the data.
    wavelength_unit : Unit
        The unit of the wavelength array.
    data_unit : Unit
        The unit of the data.
    spectral_scale : float
        The spectral scale of the FITS file, in SI units.
    pixel_scale : float
        The plate pixel scale of the FITS file, in SI units.
    slice_scale : float
        The slice pixel scale of the FITS file, in SI units.
    mask : ndarray
        The mask of the data.
    flags : ndarray
        The noted flags for each of the data points.

    """
    # We first need to check if the file even exists to read.
    if not os.path.isfile(filename):
        logging.critical(
            critical_type=logging.FileError,
            message=(
                f"We cannot read the Lezargus FITS file {filename}, it does not"
                " exist."
            ),
        )
    else:
        logging.info(
            message=f"Reading Lezargus FITS file {filename}.",
        )

    # Opening the file itself.
    with astropy.io.fits.open(filename) as raw_hdul:
        # The hdul object.
        hdul = copy.deepcopy(raw_hdul)
        # The header information that we actually care about is in the primary
        # extension.
        header = hdul["PRIMARY"].header
        # The wavelength information is kept in the wavelength extension.
        # For the wavelength unit, we try Lezargus input first, then FITS
        # standard.
        wave_table = hdul[header["LZ_WTABN"]]
        wavelength = np.ravel(wave_table.data["WAVELENGTH"])
        wavelength_unit_str = header.get("LZDWUNIT", None)
        if wavelength_unit_str is None:
            wavelength_unit_str = header.get("CUNIT3", None)
        wavelength_unit = lezargus.library.conversion.parse_astropy_unit(
            unit_string=wavelength_unit_str,
        )
        # The data is stored in the primary extension. The Lezargus axis
        # convention and some visualization conventions have the axis reversed;
        # we convert between these.
        # For the data unit, we try Lezargus input first, then FITS
        # standard.
        data = hdul["PRIMARY"].data.T
        data_unit_str = header.get("LZDFUNIT", None)
        if data_unit_str is None:
            data_unit_str = header.get("BUNIT", None)
        data_unit = lezargus.library.conversion.parse_astropy_unit(
            unit_string=data_unit_str,
        )
        # We also attempt to get any stored spectral and spatial information.
        # Note, the headers and the input specify different scales so we need
        # to convert. Header units are arcsec per pixel/slice, we want radians
        # per pixel/slice.
        spectral_scale_raw = header.get("LZDSPECS", None)
        if spectral_scale_raw is None:
            spectral_scale = None
        else:
            spectral_scale = lezargus.library.conversion.convert_units(
                value=spectral_scale_raw,
                value_unit="angstrom pix^-1",
                result_unit="meter pix^-1",
            )
        pixel_scale_raw = header.get("LZDPIXPS", None)
        if pixel_scale_raw is None:
            pixel_scale = None
        else:
            pixel_scale = lezargus.library.conversion.convert_units(
                value=pixel_scale_raw,
                value_unit="arcsec pix^-1",
                result_unit="rad pix^-1",
            )
        slice_scale_raw = header.get("LZDSLIPS", None)
        if slice_scale_raw is None:
            slice_scale = None
        else:
            slice_scale = lezargus.library.conversion.convert_units(
                value=slice_scale_raw,
                value_unit="arcsec pix^-1",
                result_unit="rad pix^-1",
            )

        # The uncertainty is stored in its own extension, We transform it like
        # the data itself.
        uncertainty = hdul[header["LZ_UIMGN"]].data.T
        # Masks and flags are stored in their own extensions as well. The mask
        # is actually written as an integer because FITS does not support
        # writing boolean values. We need to convert it back into the more
        # familiar data type.
        mask_int = hdul[header["LZ_MIMGN"]].data.T
        mask = np.array(mask_int, dtype=bool)
        flags = hdul[header["LZ_FIMGN"]].data.T
    # All done.
    return (
        header,
        wavelength,
        data,
        uncertainty,
        wavelength_unit,
        data_unit,
        spectral_scale,
        pixel_scale,
        slice_scale,
        mask,
        flags,
    )


def write_lezargus_fits_file(
    filename: str,
    header: hint.Header,
    wavelength: hint.NDArray,
    data: hint.NDArray,
    uncertainty: hint.NDArray,
    wavelength_unit: hint.Unit,
    data_unit: hint.Unit,
    spectral_scale: float,
    pixel_scale: float,
    slice_scale: float,
    mask: hint.NDArray,
    flags: hint.NDArray,
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
    wavelength : ndarray
        The wavelength information of the file.
    data : ndarray
        The data array of the Lezargus FITS file.
    uncertainty : ndarray
        The uncertainty in the data.
    wavelength_unit : Unit
        The unit of the wavelength array.
    data_unit : Unit
        The unit of the data.
    spectral_scale : float
        The spectral resolution scale of the FITS file, in SI units.
    pixel_scale : float
        The plate pixel scale of the FITS file, in SI units.
    slice_scale : float
        The slice pixel scale of the FITS file, in SI units.
    mask : ndarray
        The mask of the data.
    flags : ndarray
        The noted flags for each of the data points.
    overwrite : bool, default = False
        If True, overwrite the file upon conflicts.

    Returns
    -------
    None

    """
    # We test if the file already exists.
    filename = os.path.abspath(filename)
    if os.path.isfile(filename):
        if overwrite:
            logging.warning(
                warning_type=logging.FileWarning,
                message=(
                    f"The FITS file {filename} already exists, overwriting as"
                    " overwrite is True."
                ),
            )
        else:
            logging.critical(
                critical_type=logging.FileError,
                message=(
                    f"The FITS file {filename} already exists. Overwrite is"
                    " False."
                ),
            )

    # We need to convert the pixel/slice scale from the input radian per unit
    # to the FITS header degree per unit.
    if spectral_scale is None:
        spectral_scale_angstrom = None
    else:
        spectral_scale_angstrom = lezargus.library.conversion.convert_units(
            value=spectral_scale,
            value_unit="meter pix^-1",
            result_unit="angstrom pix^-1",
        )
    if pixel_scale is None:
        pixel_scale_arcsec = None
    else:
        pixel_scale_arcsec = lezargus.library.conversion.convert_units(
            value=pixel_scale,
            value_unit="rad pix^-1",
            result_unit="arcsec pix^-1",
        )
    if slice_scale is None:
        slice_scale_arcsec = None
    else:
        slice_scale_arcsec = lezargus.library.conversion.convert_units(
            value=slice_scale,
            value_unit="rad pix^-1",
            result_unit="arcsec pix^-1",
        )

    # We first compile the header. The unit information is kept in the header
    # as well. However, it is best to work on a copy.
    header = create_fits_header(input_dict=header.copy())
    lezargus_header = create_lezargus_fits_header(
        header=header,
        entries={
            "LZDWUNIT": str(wavelength_unit),
            "LZDFUNIT": str(data_unit),
            "LZDUUNIT": str(data_unit),
            "LZDSPECS": spectral_scale_angstrom,
            "LZDPIXPS": pixel_scale_arcsec,
            "LZDSLIPS": slice_scale_arcsec,
        },
    )
    # We purge the old header of all Lezargus keys as we will add them back
    # in bulk in order. We do not want duplicate cards.
    for keydex in lezargus_header:
        header.remove(keydex, ignore_missing=True, remove_all=True)
    header.extend(lezargus_header, update=True)

    # First we write the main data to the array.
    data_hdu = astropy.io.fits.PrimaryHDU(data.T, header=header)
    # Now the WCS binary table, most relevant for the wavelength axis. Special
    # care must be made to format the data correctly. Namely, the wavelength
    # index and axis must all fit in a row of a column; see TODO.
    n_wave = len(wavelength)
    wave_index = astropy.io.fits.Column(
        name="WAVEINDEX",
        array=np.arange(n_wave).reshape(1, 1, n_wave),
        format=f"{n_wave}J",
        dim=f"(1,{n_wave})",
    )
    wave_value = astropy.io.fits.Column(
        name="WAVELENGTH",
        array=wavelength.reshape(1, 1, n_wave),
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
    # The mask and flags are also stored in their own HDUs. Masks are usually
    # a boolean, but FITS does not support that so we need to convert.
    mask_uint8 = np.array(mask, dtype=np.uint8)
    mask_hdu = astropy.io.fits.ImageHDU(mask_uint8.T, name=header["LZ_MIMGN"])
    flags_hdu = astropy.io.fits.ImageHDU(flags.T, name=header["LZ_FIMGN"])

    # Compiling it all together and writing it to disk.
    hdul = astropy.io.fits.HDUList(
        [data_hdu, wcstab_hdu, uncertainty_hdu, mask_hdu, flags_hdu],
    )
    hdul.writeto(filename, overwrite=overwrite)


def create_fits_header(
    input_dict: dict | hint.Header | None = None,
) -> hint.Header:
    """Create a FITS header provided dictionary input.

    This function creates a FITS header from provided input cards in the
    form of a dictionary. This function mostly exists to properly sanitize
    input data to better conform to the FITS standard.

    Parameter
    ---------
    input_dict : dict, default = None
        The input dictionary to create a FITS header from. If it is None, the
        input is considered blank.


    Returns
    -------
    output_header : Astropy Header
        The header made from the input.

    """
    # If it is a header, there is nothing to do.
    if isinstance(input_dict, astropy.io.fits.Header):
        return input_dict
    # Otherwise, we first need to check if there is input.
    input_dict = input_dict if input_dict is not None else {}

    # We sort through every record and fix the issues with the dictionary.
    corrected_cards = []
    for keydex, valuedex in input_dict.items():
        # The header keys are usually capitalized.
        key = str(keydex).upper()

        value = lezargus.library.sanitize.fix_fits_header_value(
            input_data=valuedex,
        )

        # Saving the corrected record.
        carddex = astropy.io.fits.Card(key, value)
        corrected_cards.append(carddex)

    # Building the header from the corrected records.
    output_header = astropy.io.fits.Header(corrected_cards)
    return output_header


def create_lezargus_fits_header(
    header: hint.Header,
    entries: dict | None = None,
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
    entries : dict, default = None
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
        valuedex = lezargus.library.sanitize.fix_fits_header_value(
            input_data=valuedex,
        )
        lezargus_header[keydex] = (valuedex, commentdex)

    # We construct the WCS header from the Lezargus header if one does not
    # already exist. We insert it in the WCS section of the header.
    if header.get("LZWBEGIN", False):
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
            lezargus_header["LZWBEGIN"] = True

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
    if header.get("LZ_BEGIN", False):
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
