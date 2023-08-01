"""Data file functions.

This file deals with the loading in and saving of data files which are in 
the /data/ directory of Lezargus. Moreover, the contents of the data 
are accessed using attributes of this module.
"""

import os
import numpy as np
import astropy.coordinates
import astropy.units

from lezargus import library
from lezargus.library import logging
from lezargus.library import hint


# We need to get the actual directory of the data.
MODULE_DATA_DIRECTORY = os.path.join(library.config.MODULE_INSTALLATION_PATH, "data")

class LezargusStarData():
    """A class which packages star spectra data from the data directory.

    Lezargus star data instances contain all of the relevant data of specific
    stars which are important for data simulation or data reduction purposes.

    Attributes
    ----------
    wavelength : Array
        The wavelength of the spectra of the star, in units of micrometers.
    flux : Array
        The flux of the star, in units of FLAM.
    error : Array
        The error on the flux of the star, in units of FLAM.

    ra : str
        The RA of the star.
    dec : str
        The DEC of the star.
    skycoord : Astropy SkyCoord
        An object of the coordinates.
    parallax : float
        The parallax of the star, in milliarcseconds.
    spectral_type : str
        The spectral type of the star.
    
    B_mag : float
        The B-band magnitude of the star.
    B_mag_error : float
        The error on the B-band magnitude of the star.
    V_mag : float
        The V-band magnitude of the star.
    B_mag_error : float
        The error on the V-band magnitude of the star.
    J_mag : float
        The J-band magnitude of the star, usually from 2MASS.
    J_mag_error : float
        The error on the J-band magnitude of the star, usually from 2MASS.
    H_mag : float
        The H-band magnitude of the star, usually from 2MASS.
    H_mag_error : float
        The error on the H-band magnitude of the star, usually from 2MASS.
    Ks_mag : float
        The K-short band magnitude of the star, usually from 2MASS.
    Ks_mag_error : float
        The error on the Ks-band magnitude of the star, usually from 2MASS.
    """

    def __init__(self, filename:str) -> None:
        """Load the Lezargus star file.
        
        Parameters
        ----------
        filename : str
            The filename of the Lezargus star file to load.

        Returns
        -------
        None 
        """
        # Load the file itself.
        header, data = library.fits.read_fits_image_file(filename=filename)
        # The spectral data.
        self.wavelength = data[0]
        self.flux = data[1]
        self.error = data[2]

        # The Header data.
        self.ra = header.get("RA", None)
        self.dec = header.get("DEC", None)
        if self.ra is None:
            logging.error(error_type=logging.DevelopmentError, message="The Lezargus star data file {fl} does not have a RA.".format(fl=filename))
        if self.dec is None:
            logging.error(error_type=logging.DevelopmentError, message="The Lezargus star data file {fl} does not have a DEC.".format(fl=filename))
        if self.ra is not None and self.dec is not None:
            # Creating a coordinate object.
            self.skycoord = astropy.coordinates.SkyCoord(self.ra, self.dec, frame="ICRS", unit=(astropy.units.hourangle, astropy.units.deg))
        else:
            logging.error(error_type=logging.DevelopmentError, message="The Lezargus star data file {fl} cannot have its RA and DEC parsed into an Astropy SkyCoord object.")
        # Parallax, in milliarcsec.
        self.parallax = header.get("PLX", None)
        if self.parallax is None:
            logging.warning(warning_type=logging.DevelopmentWarning, message="The Lezargus star data file {fl} does not have a parallax.".format(fl=filename))
        # Spectral type.
        self.spectral_type = header.get("SPTYPE", None)
        if self.spectral_type is None:
            logging.warning(warning_type=logging.DevelopmentWarning, message="The Lezargus star data file {fl} does not have a spectral type.".format(fl=filename))
        
        # Magnitudes, the filter names here are case sensitive and there are 
        # filter version of the lowercase letters and we do not want to 
        # confuse users.
        filter_names = ["B", "V", "J", "H", "Ks"]
        for filterdex in filter_names:
            # Making standard filter names per the FITS file. By convention for
            # the FITS file headers, the filter names are described by at most
            # two characters with left justified with a trailing underscore if 
            # required.
            temp_filter = filterdex.upper() if len(filterdex) > 1 else filterdex.upper() + "_"
            if 1 <= len(temp_filter) < 3:
                logging.error(error_type=logging.DevelopmentError, message="The filter name {flt} does not follow the convention of being maximally two characters long with a trailing underscore where needed.".format(flt=temp_filter))
            # Getting both the magnitude and errors.
            mag = header.get("{fil}_MAG".format(fil=filterdex.upper()), None)
            mag_err = header.get("{fil}_M_ERR".format(fil=filterdex.upper()), None)
            if mag is None and mag_err is None:
                logging.error(error_type=logging.DevelopmentError, message="The Lezargus star data file {fl} does not have any data for the filter band {flt}.".format(fl=filename, flt=temp_filter))
                mag = np.nan
                mag_err = np.nan
            elif mag is None and mag_err is not None:
                logging.warning(warning_type=logging.DataLossWarning, message="The Lezargus star data file {fl} has, for the filter {flt}, an entry with no magnitude measurement but an error on it. We null both values.")
                mag = np.nan
                mag_err = np.nan
            else:
                # All good.
                pass
            # Finally applying the values.
            mag_variable_name = "{flt}_mag".format(flt=filterdex)
            err_variable_name = "{flt}_mag_error".format(flt=filterdex)
            setattr(self, mag_variable_name, mag)
            setattr(self, err_variable_name, mag_err)
            # All done.
            return None
# Loading the stars.
VEGA = LezargusStarData(filename=library.path.merge_pathname(directory=MODULE_DATA_DIRECTORY, filename="star_vega", extension="fits"))
