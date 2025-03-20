"""Container classes to hold spectral dispersion patterning and pixel locations.

These class serves as a simpler interface for the spectral dispersion,
displaying where light should end up on a detector due to dispersive and
imaging elements.

Note that the dispersion classes are unique to each instrument and its usage
must be done carefully.
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


class SpectreDispersionPattern:
    """The dispersion pattern for SPECTRE.

    This dispersion pattern class should only be used for the SPECTRE
    instrument! Any other instrument is unsupported with this class.
    """

    dispersion_table: hint.Table
    """The dispersion table detailing the spectral dispersion and pattern of
    the light on a detector. The format of this table should pass internal
    checks and so we do not suggest changing this table manually."""

    visible_pixel_size: float
    """The pixel size of the visible channel detector, taken from the data
    constants."""

    nearir_pixel_size: float
    """The pixel size of the near IR channel detector, taken from the data
    constants."""

    midir_pixel_size: float
    """The pixel size of the mid IR channel detector, taken from the data
    constants."""

    def __init__(
        self: SpectreDispersionPattern,
        dispersion_table: hint.Table,
    ) -> None:
        """Initialize the dispersion pattern class.

        This class should not be invoked directly. Please read in a dispersion
        table file via :py:meth:`read_dispersion_table`. The format of the
        dispersion table is a bit specific, see
        :py:meth:`verify_dispersion_table`.

        Parameters
        ----------
        dispersion_table : Astropy Table
            The dispersion table which holds the locations of the dispersive
            points. If the table doesn't match the expected format, an error
            is logged.

        Returns
        -------
        None

        """
        # We need to make sure the dispersion table can pass checks.
        if not self.verify_dispersion_table(table=dispersion_table):
            logging.error(
                error_type=logging.InputError,
                message="Dispersion table provided failed verification.",
            )
        self.dispersion_table = dispersion_table
        # All done.

    @staticmethod
    def verify_dispersion_table(table: hint.Table) -> bool:
        """Verify that the provided dispersion table has the expected fields.

        We pull the required data for dispersion determination from the table,
        and so the format of the table needs to adhere to specific
        specifications and assumptions.

        Parameters
        ----------
        table : Astropy Table
            The table we are testing.

        Returns
        -------
        verification : bool
            If True, the table passes all of our verification tests and should
            be good enough for this class.

        """
        # Assuming a good table...
        verification = True

        # Needs to be an Astropy table.
        if not isinstance(table, astropy.table.Table):
            logging.error(
                error_type=logging.InputError,
                message=(
                    "Provided dispersion table is not an Astropy table, but"
                    f" is: {type(table)}"
                ),
            )
            verification = False

        # We just make sure the columns have the expected names.
        expected_columns = [
            "channel",
            "slice",
            "wavelength",
            "center_x",
            "center_y",
            "top_x",
            "top_y",
            "bottom_x",
            "bottom_y",
            "left_x",
            "left_y",
            "right_x",
            "right_y",
        ]
        for coldex in expected_columns:
            if coldex not in table.colnames:
                logging.error(
                    error_type=logging.InputError,
                    message=(
                        f"Expected column, {coldex}, does not exist in provided"
                        " dispersion table. (Case sensitive.)"
                    ),
                )
                verification = False

        # We check to make sure there are all three channels, with the expected
        # names.
        channel_column = table.columns.get("channel", None)
        if channel_column is None:
            logging.error(
                error_type=logging.InputError,
                message=(
                    "Channel column in table does not exist, cannot test for"
                    " channel entries."
                ),
            )
            verification = False
        else:
            # Checking.
            channel_names = tuple(set(map(str, channel_column)))
            expected_channel_names = ["visible", "nearir", "midir"]
            for channeldex in expected_channel_names:
                if channeldex not in channel_names:
                    logging.error(
                        error_type=logging.InputError,
                        message=(
                            f"Expected channel data, {channeldex}, does not"
                            " exist in provided dispersion table."
                        ),
                    )
                    verification = False

        # All done.
        return verification

    @classmethod
    def read_dispersion_table(cls: type[hint.Self], filename: str) -> hint.Self:
        """Read a dispersion table file and create the dispersion class.

        Parameters
        ----------
        filename : str
            The filename of the dispersion table we are reading in.

        Returns
        -------
        dispersion_class : SpectreDispersionPattern
            The dispersion class.

        """
        # We need to make sure the file even exists in the first place.
        if not os.path.exists(filename):
            logging.error(
                error_type=logging.FileError,
                message=f"Dispersion table file does not exit: {filename}",
            )

        # Reading the file.
        raw_dispersion_table = astropy.table.Table.read(
            filename,
            format="ascii.mrt",
        )

        # We rely on the main instantiation to checking.
        dispersion_class = cls(dispersion_table=raw_dispersion_table)
        return dispersion_class

    def _query_dispersion_table(
        self: hint.Self,
        channel: str,
        slice_: int,
        column: str,
    ) -> hint.NDArray:
        """Query the dispersion table, filtering for the right data.

        This is a helper function for querying the dispersion table. It
        serves mostly to act as a wrapper around the filtering mechanisms
        as we pull data out of the table. It also has a few checks. We always
        do all wavelengths.

        Parameters
        ----------
        channel : str
            The name of the channel which you are getting the dispersion
            coordinate for; the map for all three are a little different. Must
            be one of: `visible`, `nearir`, `midir`.
        slice_ : int
            The slice index to indicate which slice you are getting. Must be
            within 1-36 (or whatever count of slices there are); this is not
            zero indexed.
        column : str
            The name of the data column of the table you want to query or
            # filter to. Available options:

                - `wavelength`: The wavelength column.
                - `center_x` or `center_y`: The X or Y data of the center.
                - `top_x` or `top_y`: The X or Y data of the top.
                - `bottom_x` or `bottom_y`: The X or Y data of the bottom.
                - `left_x` or `left_y`: The X or Y data of the left.
                - `right_x` or `right_y`: The X or Y data of the right.

        Returns
        -------
        data : NDarray
            The queried and filtered data from the table. Please note, we do
            preserve unit objects in the query. As this is an internal
            function, the known convention of the input table is enough.

        """
        # We just filter by boolean indexing. This could be optimized in the
        # future if it is an issue.
        # Channel...
        channel = channel.casefold()
        if channel not in ["visible", "nearir", "midir"]:
            logging.error(
                error_type=logging.DevelopmentError,
                message=f"Channel input not supported: {channel}",
            )
        channel_filter = self.dispersion_table["channel"] == channel
        # Slice
        n_slices = 36
        if not 1 <= slice_ <= n_slices:
            logging.error(
                error_type=logging.DevelopmentError,
                message=(
                    f"Slice input not within supported range [1, 36]: {slice_}"
                ),
            )
        slice_filter = self.dispersion_table["slice"] == slice_

        # A simple check to make sure the column data exists.
        column_data = self.dispersion_table.columns.get(column, None)
        if column_data is None:
            logging.critical(
                critical_type=logging.DevelopmentError,
                message=(
                    "Column provided is not a column in the dispersion table:"
                    f" {column}"
                ),
            )

        # Otherwise, we try and access the data.
        full_filter = channel_filter & slice_filter
        data = np.array(column_data[full_filter])
        return data

    def get_slice_dispersion_coordinate(
        self: hint.Self,
        channel: str,
        slice_: int,
        location: str,
        wavelength: hint.NDArray,
    ) -> list[tuple[float, float]]:
        """Get the coordinate location on the detector, in linear units.

        Note, this function returns the coordinates in meters, consider using
        py:meth:`get_slice_dispersion_pixel` for the pixel coordinates.

        Parameters
        ----------
        channel : str
            The name of the channel which you are getting the dispersion
            coordinate for; the map for all three are a little different. Must
            be one of: `visible`, `nearir`, `midir`.
        slice_ : int
            The slice index to indicate which slice you are getting. Must be
            within 1-36 (or whatever count of slices there are); this is not
            zero indexed.
        location : str
            The location of the area you want relative to the center of the
            slice. Acceptable terms:

                - `center`: The center of the slice.
                - `left` or `right`: The direct left and right from the center.
                - `top` or `bottom`: The direct top and bottom from the center.
                - `top_left` or `top_right`: The top left and right corners.
                - `bottom_left` or `bottom_right`: The bottom corners.

        wavelength : float | NDArray
            The wavelength(s) to get the coordinates for the dispersion for,
            required as this is to model spectral dispersion using its
            spatial displacement effects.

        Returns
        -------
        coordinate_pairs : list
            The list of the (X, Y) coordinate pairs, parallel with the provided
            wavelength array.

        """
        # We need to determine the location desired, specifying which points we
        # use to interpolate.
        location = location.casefold()
        # The normal points...
        if location == "center":
            interp_wave = self._query_dispersion_table(
                channel=channel,
                slice_=slice_,
                column="wavelength",
            )
            interp_x = self._query_dispersion_table(
                channel=channel,
                slice_=slice_,
                column="center_x",
            )
            interp_y = self._query_dispersion_table(
                channel=channel,
                slice_=slice_,
                column="center_y",
            )
        elif location == "left":
            interp_wave = self._query_dispersion_table(
                channel=channel,
                slice_=slice_,
                column="wavelength",
            )
            interp_x = self._query_dispersion_table(
                channel=channel,
                slice_=slice_,
                column="left_x",
            )
            interp_y = self._query_dispersion_table(
                channel=channel,
                slice_=slice_,
                column="left_y",
            )
        elif location == "right":
            interp_wave = self._query_dispersion_table(
                channel=channel,
                slice_=slice_,
                column="wavelength",
            )
            interp_x = self._query_dispersion_table(
                channel=channel,
                slice_=slice_,
                column="right_x",
            )
            interp_y = self._query_dispersion_table(
                channel=channel,
                slice_=slice_,
                column="right_y",
            )
        elif location == "top":
            interp_wave = self._query_dispersion_table(
                channel=channel,
                slice_=slice_,
                column="wavelength",
            )
            interp_x = self._query_dispersion_table(
                channel=channel,
                slice_=slice_,
                column="top_x",
            )
            interp_y = self._query_dispersion_table(
                channel=channel,
                slice_=slice_,
                column="top_y",
            )
        elif location == "bottom":
            interp_wave = self._query_dispersion_table(
                channel=channel,
                slice_=slice_,
                column="wavelength",
            )
            interp_x = self._query_dispersion_table(
                channel=channel,
                slice_=slice_,
                column="bottom_x",
            )
            interp_y = self._query_dispersion_table(
                channel=channel,
                slice_=slice_,
                column="bottom_y",
            )
        # These points are the corners, which we just assume a rectangle and
        # switch up the X and Y points accordingly.
        elif location == "top_left":
            interp_wave = self._query_dispersion_table(
                channel=channel,
                slice_=slice_,
                column="wavelength",
            )
            interp_x = self._query_dispersion_table(
                channel=channel,
                slice_=slice_,
                column="left_x",
            )
            interp_y = self._query_dispersion_table(
                channel=channel,
                slice_=slice_,
                column="top_y",
            )
        elif location == "top_right":
            interp_wave = self._query_dispersion_table(
                channel=channel,
                slice_=slice_,
                column="wavelength",
            )
            interp_x = self._query_dispersion_table(
                channel=channel,
                slice_=slice_,
                column="right_x",
            )
            interp_y = self._query_dispersion_table(
                channel=channel,
                slice_=slice_,
                column="top_y",
            )
        elif location == "bottom_left":
            interp_wave = self._query_dispersion_table(
                channel=channel,
                slice_=slice_,
                column="wavelength",
            )
            interp_x = self._query_dispersion_table(
                channel=channel,
                slice_=slice_,
                column="left_x",
            )
            interp_y = self._query_dispersion_table(
                channel=channel,
                slice_=slice_,
                column="bottom_y",
            )
        elif location == "bottom_right":
            interp_wave = self._query_dispersion_table(
                channel=channel,
                slice_=slice_,
                column="wavelength",
            )
            interp_x = self._query_dispersion_table(
                channel=channel,
                slice_=slice_,
                column="right_x",
            )
            interp_y = self._query_dispersion_table(
                channel=channel,
                slice_=slice_,
                column="bottom_y",
            )
        # The input was likely not supported.
        else:
            interp_wave = interp_x = interp_y = np.array([np.nan, np.nan])
            logging.error(
                error_type=logging.InputError,
                message=(
                    f"Location provided, {location}, does not seem to match"
                    " provided options, see documentation."
                ),
            )

        # Now that we have the data we can interpolate to the wavelengths
        # desired.
        interpolator_x = lezargus.library.interpolate.Linear1DInterpolate(
            x=interp_wave,
            v=interp_x,
            extrapolate=True,
        )
        interpolator_y = lezargus.library.interpolate.Linear1DInterpolate(
            x=interp_wave,
            v=interp_y,
            extrapolate=True,
        )
        # And we get the values.
        wavelength = np.atleast_1d(wavelength)
        coordinate_x = interpolator_x(x=wavelength)
        coordinate_y = interpolator_y(x=wavelength)
        # Creating the pairs.

        coordinate_pairs = list(zip(coordinate_x, coordinate_y, strict=True))
        return coordinate_pairs

    def get_slice_dispersion_pixel(
        self: hint.Self,
        channel: str,
        slice_: int,
        location: str,
        wavelength: hint.NDArray,
    ) -> list[tuple[float, float]]:
        """Get the coordinate location on the detector, in pixels.

        Note, this function returns the coordinates in pixel units. We convert
        from the linear units from the original function
        py:meth:`get_slice_dispersion_coordinate` to pixel units based on the
        pixel scale of each detector within the channel. This is mostly a
        convenience function and the main function is the predominant source
        of the documentation.

        Parameters
        ----------
        channel : str
            The name of which of the three channels being queried.
        slice_ : int
            The slice index, 1-36 inclusive, for the slice being queried.
        location : str
            The location of the area you want relative to the center of the
            slice. Acceptable terms:

                - `center`: The center of the slice.
                - `left` or `right`: The direct left and right from the center.
                - `top` or `bottom`: The direct top and bottom from the center.
                - `top_left` or `top_right`: The top left and right corners.
                - `bottom_left` or `bottom_right`: The bottom corners.

        wavelength : float | NDArray
            The wavelength(s) to get the coordinates for the dispersion.

        Returns
        -------
        pixel_coordinate_pairs : list
            The list of the (X, Y) coordinate pairs in pixel units, parallel
            with the provided wavelength array.

        """
        # We are mostly just converting the linear coordinates given the
        # pixel sizes so we need the linear coordinates first.
        coordinate_pairs = self.get_slice_dispersion_coordinate(
            channel=channel,
            slice_=slice_,
            location=location,
            wavelength=wavelength,
        )

        # We need to use the correct pixel size, depending on the channel.
        channel = channel.casefold()
        if channel == "visible":
            pixel_size = lezargus.data.CONST_VISIBLE_PIXEL_SIZE
            detector_size = lezargus.data.CONST_VISIBLE_DETECTOR_SIZE
        elif channel == "nearir":
            pixel_size = lezargus.data.CONST_NEARIR_PIXEL_SIZE
            detector_size = lezargus.data.CONST_NEARIR_DETECTOR_SIZE
        elif channel == "midir":
            pixel_size = lezargus.data.CONST_MIDIR_PIXEL_SIZE
            detector_size = lezargus.data.CONST_MIDIR_DETECTOR_SIZE
        else:
            logging.error(
                error_type=logging.InputError,
                message=(
                    f"Channel provided, {channel}, does not have a pixel size."
                ),
            )

        # Convert the coordinates to pixels.
        center_pixel_coordinate_pairs = [
            (xdex / pixel_size, ydex / pixel_size)
            for (xdex, ydex) in coordinate_pairs
        ]

        # The values computed are center-origin which is atypical for pixels.
        # We shift it so the origin is correctly in a corner.
        # We assume a square detector.
        pixel_coordinate_pairs = [
            (xdex + detector_size // 2, ydex + detector_size // 2)
            for (xdex, ydex) in center_pixel_coordinate_pairs
        ]

        return pixel_coordinate_pairs
