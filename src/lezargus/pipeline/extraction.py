"""Extraction code to extract data from raw image arrays.

Image arrays hold all of the data needed. However, often, we need to extract
from the images specific parts, like spectral slices or dispersion regions.
This module holds all of the extraction classes for a wide array of
extractions needed.
"""

# isort: split
# Import required to remove circular dependencies from type checking.
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lezargus.library import hint
# isort: split


import os

import astropy.table
import numpy as np

import lezargus
from lezargus.library import logging


class SpectreExtractor:
    """Extract SPECTRE data and traces.

    This is the primary class for extracting SPECTRE slices. This module does
    not process the data any more than it requires to pull out a slice's
    array based on the location of the slice. The location of the slice is
    determined either by the flat field itself or archival positions.
    """

    image: hint.LezargusImage
    """The image which we are extracting slices from. We operate on this
    copy as a de-facto read-only object and thus changing this changes the
    extraction."""

    slice_corners: hint.Table
    """Corner coordinates of each of the slices; as arranged in a table. The
    corners are defined as an (x, y) pair based on the labeled slice and the
    location of the corner per the table."""

    channel: hint.Literal["visible", "nearir", "midir"]  # noqa: F821, UP037
    """The specific channel of the three channels of SPECTRE which the image
    is in. The channel is needed to define the initial conditions for
    finding the location of the slices."""

    def __init__(
        self: SpectreExtractor,
        image: hint.LezargusImage,
        channel: str,
    ) -> None:
        """Initialize the SPECTRE extractor class.

        Parameters
        ----------
        image : LezargusImage
            The image array we are working with to extract.
        channel : str
            The channel that the image exists in.


        Returns
        -------
        None

        """
        # Applying the image, and other parameters.
        self.image = image

        # Select the channel that this instance is simulating, and assign the
        # other parameters per the correct channel.
        channel = channel.casefold()
        if channel == "visible":
            self.channel = "visible"
        elif channel == "nearir":
            self.channel = "nearir"
        elif channel == "midir":
            self.channel = "midir"
        else:
            logging.error(
                error_type=logging.InputError,
                message=(
                    f"Channel name input {channel} does not match: visible,"
                    " nearir, midir."
                ),
            )

        # All done.

    def __calculate_initial_slice_corners_simulation(
        self: hint.Self,
    ) -> hint.Table:
        """Derive the slice corners from the SPECTRE simulation defaults.

        One option to derive the slice corners is to use the simulation to
        predict where the slice corners should be. We don't actually need to
        boot the simulation up as the predefined slice pattern should give
        us the information.

        Parameters
        ----------
        None

        Returns
        -------
        initial_slice_corners : Table
            The initial slice corners as derived from the SPECTRE simulation
            slice pattern.

        """
        # Number of slices...
        n_slices = lezargus.data.CONST_SPECTRE_SLICES
        # And the disperser we are working with.
        dispersion_spectre = lezargus.data.DISPERSION_SPECTRE

        # Depending on the channel we are in, the blue and red wavelength ends
        # of the slice range differs.
        # These are nominal values and are not perfect, but are good enough
        # for determining the initial slice corners. The units should be SI,
        # but we make it easier to read.
        if self.channel == "visible":
            min_wave = 0.400 * 1e-6
            max_wave = 0.850 * 1e-6
        elif self.channel == "nearir":
            min_wave = 0.850 * 1e-6
            max_wave = 2.400 * 1e-6
        elif self.channel == "midir":
            min_wave = 2.400 * 1e-6
            max_wave = 4.200 * 1e-6
        else:
            min_wave = np.nan
            max_wave = np.nan
            logging.error(
                error_type=logging.InputError,
                message=(
                    f"Channel name input {self.channel} does not match:"
                    " visible, nearir, midir."
                ),
            )

        # Deriving the corners for each slice.
        slice_index_list = []
        top_left_corners = []
        top_right_corners = []
        bot_left_corners = []
        bot_right_corners = []
        for slicedex in range(n_slices):
            # The slice index is not 0 indexed, but is instead 1-36.
            slice_index = slicedex + 1
            # The top coordinates are defined by the upper level of the
            # blue-most wavelength end.
            top_left = dispersion_spectre.get_slice_dispersion_pixel(
                channel=self.channel,
                slice_=slice_index,
                location="top_left",
                wavelength=min_wave,
            )
            top_right = dispersion_spectre.get_slice_dispersion_pixel(
                channel=self.channel,
                slice_=slice_index,
                location="top_right",
                wavelength=min_wave,
            )
            # And the bottom coordinates are defined by the lower level of the
            # red-most wavelength end.
            bot_left = dispersion_spectre.get_slice_dispersion_pixel(
                channel=self.channel,
                slice_=slice_index,
                location="bot_left",
                wavelength=max_wave,
            )
            bot_right = dispersion_spectre.get_slice_dispersion_pixel(
                channel=self.channel,
                slice_=slice_index,
                location="bot_right",
                wavelength=max_wave,
            )

            # Adding to the corner collection.
            slice_index_list.append(slice_index)
            top_left_corners.append(top_left)
            top_right_corners.append(top_right)
            bot_left_corners.append(bot_left)
            bot_right_corners.append(bot_right)

        # Converting and splitting the coordinates per the convention of the
        # table.
        top_left_x, top_left_y = np.array(top_left_corners).transpose()
        top_right_x, top_right_y = np.array(top_right_corners).transpose()
        bot_left_x, bot_left_y = np.array(bot_left_corners).transpose()
        bot_right_x, bot_right_y = np.array(bot_right_corners).transpose()

        # And creating the table.
        table_columns = {
            "slice": slice_index_list,
            "top_left_x": top_left_x,
            "top_left_y": top_left_y,
            "top_right_x": top_right_x,
            "top_right_y": top_right_y,
            "bot_left_x": bot_left_x,
            "bot_left_y": bot_left_y,
            "bot_right_x": bot_right_x,
            "bot_right_y": bot_right_y,
        }
        initial_slice_corners = astropy.table.Table(table_columns)

        # All done.
        return initial_slice_corners

    def __calculate_initial_slice_corners_table(
        self: hint.Self,
        filename: str,
    ) -> hint.Table:
        """Derive the slice corners from a table containing the coordinates.

        If there exists already a file table with the corners laid out,
        we can use that instead.

        Parameters
        ----------
        filename : str
            The filename of the file which has the table which we will read in
            for the slice corners.

        Returns
        -------
        initial_slice_corners : Table
            The initial slice corners as derived from reading the file.

        """
        # We need to make sure the file actually exists.
        if not os.path.exists(filename):
            logging.critical(
                critical_type=logging.FileError,
                message=(
                    f"The initial slice corners file {filename} does not exist."
                ),
            )

        # Otherwise, we attempt to read in the table.
        try:
            raw_table = astropy.table.Table.read(
                filename,
                comment="#",
                format="ascii.mrt",
            )
        except ZeroDivisionError:
            logging.error(
                error_type=logging.InputError,
                message=f"Cannot parse {filename} as a valid Astropy Table.",
            )

        # Number of slices...
        n_slices = lezargus.data.CONST_SPECTRE_SLICES

        # From the raw table, we pull the needed data based on the conventions
        # provided here. The best way to verify that the table works is just
        # by transcribe it for all of the data we need. If it fails to provide
        # said data, it is bad; if it can provide the data, who cares if other
        # parts are out of specification.
        slice_index_list = []
        top_left_corners = []
        top_right_corners = []
        bot_left_corners = []
        bot_right_corners = []
        # We structure the read and write code similar to pulling it from the
        # simulation because copy and paste is easy.
        for slicedex in range(n_slices):
            # The slice index is not 0 indexed, but is instead 1-36.
            slice_index = slicedex + 1

            # Here, we attempt to get the data from the table.
            try:
                # The row where the data should be.
                raw_row = raw_table[raw_table["slice"]]
                # And getting the corners.
                top_left = (raw_row["top_left_x"], raw_row["top_left_y"])
                top_right = (raw_row["top_right_x"], raw_row["top_right_y"])
                bot_left = (raw_row["bot_left_x"], raw_row["bot_left_y"])
                bot_right = (raw_row["bot_right_x"], raw_row["bot_right_y"])
            except KeyError:
                logging.error(
                    error_type=logging.InputError,
                    message=(
                        f"Slice corner file {filename} failed to provide"
                        f" expected data for slice {slice_index}."
                    ),
                )

            # Adding to the corner collection.
            slice_index_list.append(slice_index)
            top_left_corners.append(top_left)
            top_right_corners.append(top_right)
            bot_left_corners.append(bot_left)
            bot_right_corners.append(bot_right)

        # Converting and splitting the coordinates per the convention of the
        # table.
        top_left_x, top_left_y = np.array(top_left_corners).transpose()
        top_right_x, top_right_y = np.array(top_right_corners).transpose()
        bot_left_x, bot_left_y = np.array(bot_left_corners).transpose()
        bot_right_x, bot_right_y = np.array(bot_right_corners).transpose()

        # And creating the table.
        table_columns = {
            "slice": slice_index_list,
            "top_left_x": top_left_x,
            "top_left_y": top_left_y,
            "top_right_x": top_right_x,
            "top_right_y": top_right_y,
            "bot_left_x": bot_left_x,
            "bot_left_y": bot_left_y,
            "bot_right_x": bot_right_x,
            "bot_right_y": bot_right_y,
        }
        initial_slice_corners = astropy.table.Table(table_columns)

        # All done.
        return initial_slice_corners

    def __calculate_initial_slice_corners_flat(
        self: hint.Self,
        flat_array: hint.NDArray,
    ) -> hint.Table:
        """Hi."""
