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
import copy

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

    def _calculate_initial_slice_corners_simulation(
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
        spectre_disperser = lezargus.data.DISPERSION_SPECTRE

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
            top_left = spectre_disperser.get_slice_dispersion_pixel(
                channel=self.channel,
                slice_=slice_index,
                location="top_left",
                wavelength=min_wave,
            )
            top_right = spectre_disperser.get_slice_dispersion_pixel(
                channel=self.channel,
                slice_=slice_index,
                location="top_right",
                wavelength=min_wave,
            )
            # And the bottom coordinates are defined by the lower level of the
            # red-most wavelength end.
            bot_left = spectre_disperser.get_slice_dispersion_pixel(
                channel=self.channel,
                slice_=slice_index,
                location="bottom_left",
                wavelength=max_wave,
            )
            bot_right = spectre_disperser.get_slice_dispersion_pixel(
                channel=self.channel,
                slice_=slice_index,
                location="bottom_right",
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

        # And creating the table. For some reason, the corner coordinates have 
        # unneeded dimensions.
        table_columns = {
            "slice": slice_index_list,
            "top_left_x": top_left_x,
            "top_left_y": top_left_y,
            "top_right_x": top_right_x,
            "top_right_y": top_right_y,
            "bottom_left_x": bot_left_x,
            "bottom_left_y": bot_left_y,
            "bottom_right_x": bot_right_x,
            "bottom_right_y": bot_right_y,
        }
        table_columns = {keydex:np.squeeze(valuedex) for keydex, valuedex in table_columns.items()}

        initial_slice_corners = astropy.table.Table(table_columns)

        # All done.
        return initial_slice_corners

    def _calculate_initial_slice_corners_table(
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
                bot_left = (raw_row["bottom_left_x"], raw_row["bottom_left_y"])
                bot_right = (raw_row["bottom_right_x"], raw_row["bottom_right_y"])
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

        # And creating the table. For some reason, the corner coordinates have 
        # unneeded dimensions.
        table_columns = {
            "slice": slice_index_list,
            "top_left_x": top_left_x,
            "top_left_y": top_left_y,
            "top_right_x": top_right_x,
            "top_right_y": top_right_y,
            "bottom_left_x": bot_left_x,
            "bottom_left_y": bot_left_y,
            "bottom_right_x": bot_right_x,
            "bottom_right_y": bot_right_y,
        }
        table_columns = {keydex:np.squeeze(valuedex) for keydex, valuedex in table_columns.items()}
        initial_slice_corners = astropy.table.Table(table_columns)

        # All done.
        return initial_slice_corners

    def _calculate_initial_slice_corners_flat(
        self: hint.Self,
        flat_array: hint.NDArray,
    ) -> hint.Table:
        """Derive the slice corners from a flat field image.

        This method determines the slice corners via corner detection of the 
        flat field image. We use other initial corner methods (table first, 
        then simulation) to determine the order of the points. Corner detection
        algorithms do not typically keep and named order to the points found.

        Parameters
        ----------
        flat_array : NDArray
            The array containing the flat field image data. The initial corners
            are determined from this array.

        Returns
        -------
        initial_slice_corners : Table
            The initial slice corners as derived from reading the file.

        """
        # If needed, thresholding of the array should be done here.
        threshold_array = flat_array

        # Now, we determine the corners.
        n_slices = lezargus.data.CONST_SPECTRE_SLICES
        n_corners = n_slices * 4
        raw_corners = lezargus.library.transform.corner_detection(array=threshold_array, max_corners=n_corners, quality_level=0.001, minimum_distance=3)
        # It is probably easier to have it as separate values.
        raw_corner_x, raw_corner_y = np.transpose(raw_corners)

        # The corners are unordered so we use the simulation corners to help us
        # determine which corners are which. We attempt to the table first.
        try:
            table_filename = None
            labeled_corners = self._calculate_initial_slice_corners_table(filename=table_filename)
        except Exception:
            # Using the simulation instead as something is wrong with the file.
            labeled_corners = self._calculate_initial_slice_corners_simulation()

        # Assuming the closest found corner to the simulation corner is the 
        # correct way to go. We go through all corners and slices.
        corner_names = ["top_left", "top_right", "bottom_left", "bottom_right"]
        # We find the point and just repopulate the labeled corner table.
        initial_slice_corners = copy.deepcopy(labeled_corners)
        for slicedex in range(n_slices):
            # The slices are 1-based indexed.
            slice_index = slicedex  + 1
            for cornerdex in corner_names:
                # The expected location for this specific corner.
                labeled_rowdex = labeled_corners[labeled_corners["slice"] == slice_index]
                expect_x = np.array(labeled_rowdex[f"{cornerdex}_x"])
                expect_y = np.array(labeled_rowdex[f"{cornerdex}_y"])
                # The (Euclidean) separation.
                separation = (raw_corner_x - expect_x)**2 + (raw_corner_y - expect_y)**2
                # And whichever point is the minimum is likely the matching
                # point.
                min_sep_index = np.argmin(separation)
                matched_x = raw_corner_x[min_sep_index]
                matched_y = raw_corner_y[min_sep_index]

                # Applying the values to the current table.
                initial_slice_corners[f"{cornerdex}_x"][slicedex] = matched_x
                initial_slice_corners[f"{cornerdex}_y"][slicedex] = matched_y

        # All done.
        return initial_slice_corners