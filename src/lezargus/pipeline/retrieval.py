"""Retrieval and extraction code get slice data from raw image arrays.

Image arrays hold all of the data needed. However, often, we need to retrieve
the specific parts from the image data, like spectral slices or dispersion
regions. This module holds all of the retrieval classes for a wide array of
retrievals needed. We use the phrase "retrieval" here to refer to extracting
the data from the image arrays to the slices (or cube), due to the use of
extraction referring to spectra from cubes.
"""

# isort: split
# Import required to remove circular dependencies from type checking.
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lezargus.library import hint
# isort: split


import copy
import os

import astropy.table
import cv2
import numpy as np
import scipy.stats

import lezargus
from lezargus.library import logging


class SpectreRetrieval:
    """Retrieve SPECTRE data and traces.

    This is the primary class for retrieving SPECTRE slices. This module does
    not process the data any more than it requires to pull out a slice's
    array based on the location of the slice. The location of the slice is
    determined either by the flat field itself or archival positions.
    """

    image: hint.LezargusImage
    """The data image which we are retrieving slices from. We operate on this
    copy as a de-facto read-only object and thus changing this changes the
    retrieval."""

    flat_image: hint.LezargusImage
    """The slice flat field image which we use to determine how and where to
    retrieve the slices from the data image."""

    arc_image: hint.LezargusImage
    """The arc lamp field image which we use to determine the wavelength
    solution of the data image as retrieved. For the mid-ir region, the
    data image itself will also contribute to the determination of the
    wavelength solution."""

    slice_corners: hint.Table
    """Corner coordinates of each of the slices; as arranged in a table. The
    corners are defined as an (x, y) pair based on the labeled slice and the
    location of the corner per the table."""

    channel: hint.Literal["visible", "nearir", "midir"]  # noqa: F821, UP037
    """The specific channel of the three channels of SPECTRE which the image
    is in. The channel is needed to define the initial conditions for
    finding the location of the slices."""

    def __init__(
        self: SpectreRetrieval,
        flat_image: hint.LezargusImage,
        arc_image: hint.LezargusImage,
        channel: str,
        image: hint.LezargusImage | None = None,
    ) -> None:
        """Initialize the SPECTRE retrieval class.

        Parameters
        ----------
        flat_image : LezargusImage
            The flat field image used for retrieving the slices for both the
            main image data and the arc lamp data.
        arc_image : LezargusImage
            The arc lamp image used for creating the wavelength solution for
            each slice.
        channel : str
            The channel that the image exists in.
        image : LezargusImage, default = None
            The image array we are working with to retrieve. It does not need
            to be provided now as it can be provided later or during
            retrieval itself.


        Returns
        -------
        None

        """
        # Select the channel that this instance is retrieving, and assign the
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

        # Adding the flats and arc images. It is easy to just use the
        # built-in replacement functions here as the handle everything we need.
        self.replace_flat_image(new_flat_image=flat_image)
        self.replace_arc_image(new_arc_image=arc_image)

        # If provided, we can also store the data image.
        if image is not None:
            self.image = image

        # All done.

    def replace_flat_image(
        self: hint.Self,
        new_flat_image: hint.LezargusImage,
    ) -> None:
        """Replace the stored flat field image with the new one.

        This is a helper function to replace the stored flat field with a new
        flat field image. This function also calls the needed functions to
        recompute specific parameters for smooth operation, making this the
        preferred method of utilizing a few flat image.

        Parameters
        ----------
        new_flat_image : LezargusImage
            The new flat field image which we are using to replace.

        Returns
        -------
        None

        """
        # We add the old flat field with the new flat field.
        self.flat_image = new_flat_image

        # We also then recompute the slice corners, explicitly, we go through
        # all of the methods per the default.
        self.slice_corners = self.find_slice_corners(
            flat_image=new_flat_image,
            initial_method=None,
        )

        # All done.

    def replace_arc_image(
        self: hint.Self,
        new_arc_image: hint.LezargusImage,
    ) -> None:
        """Replace the stored arc line lamp image with the new one.

        This is a helper function to replace the stored arc line lamp image
        with a new arc line lamp image. This function also calls the needed
        functions to recompute specific parameters for smooth operation,
        making this the preferred method of utilizing a few flat image.

        Parameters
        ----------
        new_arc_image : LezargusImage
            The new new arc lamp image which we are using to replace.

        Returns
        -------
        None

        """
        # We add the old arc image with the new arc image.
        self.arc_image = new_arc_image

        # All done.

    def _calculate_initial_slice_corners_simulation(
        self: hint.Self,
    ) -> hint.Table | None:
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
        initial_slice_corners : Table | None
            The initial slice corners as derived from the SPECTRE simulation
            slice pattern. If None, then the calculation of the initial
            corners failed.

        """
        # Number of slices...
        n_slices = lezargus.data.CONST_SPECTRE_SLICES
        # And the simulation disperser we are working with.
        spectre_disperser = lezargus.data.SPECTRE_DISPERSION

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
        table_columns = {
            keydex: np.squeeze(valuedex)
            for keydex, valuedex in table_columns.items()
        }

        initial_slice_corners = astropy.table.Table(table_columns)

        # All done.
        return initial_slice_corners

    def _calculate_initial_slice_corners_file(
        self: hint.Self,
        filename: str,
    ) -> hint.Table | None:
        """Derive the slice corners from a file containing the coordinates.

        If there exists already a file table with the corners laid out,
        we can use that instead.

        Parameters
        ----------
        filename : str
            The filename of the file which has the table which we will read in
            for the slice corners.

        Returns
        -------
        initial_slice_corners : Table | None
            The initial slice corners as derived from reading the file. If
            None, then the calculation of the initial corners failed.

        """
        # We need to make sure the file actually exists.
        if not os.path.exists(filename):
            logging.error(
                error_type=logging.FileError,
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
        except FileNotFoundError as error:
            logging.error(
                error_type=logging.FileError,
                message=(
                    f"Cannot parse {filename} as a valid Astropy Table; the"
                    f" following error occurred: {type(error).__name__} :"
                    f" {error!s}"
                ),
            )
            return None

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
                bot_right = (
                    raw_row["bottom_right_x"],
                    raw_row["bottom_right_y"],
                )
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
        table_columns = {
            keydex: np.squeeze(valuedex)
            for keydex, valuedex in table_columns.items()
        }
        initial_slice_corners = astropy.table.Table(table_columns)

        # All done.
        return initial_slice_corners

    def _calculate_initial_slice_corners_flat(
        self: hint.Self,
        flat_array: hint.NDArray,
        use_harris: bool = False,
    ) -> hint.Table | None:
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
        use_harris : bool, default = False
            Argument passed to the corner detection algorithm. If True, we use
            the Harris corner detection method as opposed to the default
            Shi-Tomasi method.

        Returns
        -------
        initial_slice_corners : Table
            The initial slice corners as derived from reading the file. If
            None, then the calculation of the initial corners failed.

        """
        # If needed, thresholding of the array should be done here.
        threshold_array = flat_array

        # Now, we determine the corners.
        n_slices = lezargus.data.CONST_SPECTRE_SLICES
        n_corners = n_slices * 4
        # Quality level and minimum distance is mostly just heuristic, and
        # are just dummy levels. We need to change the data type of the arrays.
        threshold_array_float32 = np.asarray(threshold_array, dtype=np.float32)
        raw_corners = lezargus.library.transform.corner_detection(
            array=threshold_array_float32,
            max_corners=n_corners,
            quality_level=0.001,
            minimum_distance=3,
            use_harris=use_harris,
        )
        # It is probably easier to have it as separate values.
        raw_corner_x, raw_corner_y = np.transpose(raw_corners)

        # The corners are unordered so we use the table or simulation corners
        # to help us determine which corners are which. We attempt to the
        # table first.
        labeled_corners = None
        table_filename = "Dummy"
        labeled_corners = self._calculate_initial_slice_corners_file(
                filename=table_filename,
        )
        # If there is no labeled corners, that means something went wrong with
        # determining the labels from the file, and we go to a backup method.
        if labeled_corners is None:
            # We are using the simulation as a backup.
            logging.warning(
                warning_type=logging.AlgorithmWarning,
                message=(
                    "Using the labeled corner table from simulation as a"
                    " backup."
                ),
            )
            labeled_corners = self._calculate_initial_slice_corners_simulation()


        # If there still is no labeled corner table...we just cannot get a 
        # labeled table. There is no possible way to get the corner table.
        if labeled_corners is None:
            logging.error(
                error_type=logging.AlgorithmError,
                message=(
                    "No labeled corner table, not found via file or"
                    " simulation. Corner table from flat cannot be made."
                ),
            )
            return None

        # Assuming the closest found corner to the simulation corner is the
        # correct way to go. We go through all corners and slices.
        corner_names = ["top_left", "top_right", "bottom_left", "bottom_right"]
        # We find the point and just repopulate the labeled corner table.
        initial_slice_corners = copy.deepcopy(labeled_corners)
        for slicedex in range(n_slices):
            # The slices are 1-based indexed.
            slice_index = slicedex + 1
            for cornerdex in corner_names:
                # The expected location for this specific corner.
                labeled_rowdex = labeled_corners[
                    labeled_corners["slice"] == slice_index
                ]
                expect_x = np.array(labeled_rowdex[f"{cornerdex}_x"])
                expect_y = np.array(labeled_rowdex[f"{cornerdex}_y"])
                # The (Euclidean) separation.
                separation = (raw_corner_x - expect_x) ** 2 + (
                    raw_corner_y - expect_y
                ) ** 2
                # And whichever point is the minimum separation is likely the
                # matching point.
                min_sep_index = np.nanargmin(separation)
                matched_x = raw_corner_x[min_sep_index]
                matched_y = raw_corner_y[min_sep_index]

                # Applying the values to the current table.
                initial_slice_corners[f"{cornerdex}_x"][slicedex] = matched_x
                initial_slice_corners[f"{cornerdex}_y"][slicedex] = matched_y

        # All done.
        return initial_slice_corners

    def _refine_initial_slice_corners(
        self: hint.Self,
        initial_slice_corners: hint.Table,
        flat_array: hint.NDArray,
    ) -> hint.Table:
        """Refine the initial slice corners determined by other methods.

        Parameters
        ----------
        initial_slice_corners : Table
            The initial slice corner table which we will be refining.
        flat_array : NDArray
            The array containing the flat field image data. This array is what
            we use to refine the corners.

        Returns
        -------
        refined_slice_corners : Table
            The refined slice corners derived from the subpixel corner
            refinement of the initial slice corners.

        """
        # The subpixel refinement only cares about using raw data points.
        corner_points = self._corner_table_to_points(
            corner_table=initial_slice_corners,
        )

        # Converting the data type to what is expected to the subpixel
        # refinement.
        flat_array_float32 = np.asarray(flat_array, dtype=np.float32)

        # Some default settings for the subpixel refinement. These values
        # seem to work the vast majority of cases.
        search_radius = 5
        iteration_count = 1000
        refined_corner_points = (
            lezargus.library.transform.corner_detection_subpixel_refinement(
                array=flat_array_float32,
                initial_corners=corner_points,
                search_radius=search_radius,
                iterations=iteration_count,
            )
        )

        # It is easier to have it as separate values for the next step of
        # labeling.
        raw_refined_x, raw_refined_y = np.transpose(refined_corner_points)

        # The refined points does not have the labeling typical with the
        # corner tables, so we find the refined points and normal corner
        # points correspondence.
        # Assuming the closest found corner to the simulation corner is the
        # correct way to go. We go through all corners and slices.
        n_slices = lezargus.data.CONST_SPECTRE_SLICES
        corner_names = ["top_left", "top_right", "bottom_left", "bottom_right"]
        labeled_corners = initial_slice_corners
        # We find the point and just repopulate the labeled corner table.
        refined_slice_corners = copy.deepcopy(labeled_corners)
        for slicedex in range(n_slices):
            # The slices are 1-based indexed.
            slice_index = slicedex + 1
            for cornerdex in corner_names:
                # The expected location for this specific corner.
                labeled_rowdex = labeled_corners[
                    labeled_corners["slice"] == slice_index
                ]
                expect_x = np.array(labeled_rowdex[f"{cornerdex}_x"])
                expect_y = np.array(labeled_rowdex[f"{cornerdex}_y"])
                # The (Euclidean) separation.
                separation = (raw_refined_x - expect_x) ** 2 + (
                    raw_refined_y - expect_y
                ) ** 2
                # And whichever point is the minimum separation is likely the
                # matching point.
                min_sep_index = np.nanargmin(separation)
                matched_x = raw_refined_x[min_sep_index]
                matched_y = raw_refined_y[min_sep_index]

                # Applying the values to the current table.
                refined_slice_corners[f"{cornerdex}_x"][slicedex] = matched_x
                refined_slice_corners[f"{cornerdex}_y"][slicedex] = matched_y

        # All done.
        return refined_slice_corners

    def find_slice_corners(
        self: hint.Self,
        flat_image: hint.LezargusImage,
        initial_method: str | None = None,
    ) -> hint.Table:
        """Recompute the slice corners from an image flat.

        Parameters
        ----------
        flat_image : LezargusImage
            The flat field image for which we are determining the corners of.
        initial_method : str, default = None
            The method we get the initial slice corners. If provided, we only
            use the method provided; otherwise we try all of the methods in
            the following order (of accuracy and inversely of reliability):

            - "flat" : Initial corners determined by corner detection of the
            current provided flat field image.
            - "file" : Initial corners determined by the provided/stored file.
            - "simulation" : Initial corners determined based on where the
            simulation expects them to be.

        apply : bool = True
            If True, the new slice corners are applied to this class and
            overwrites the `slice_corners` table with the new set of corners.

        Returns
        -------
        slice_corners : Table
            The new recomputed slice corners.

        """
        # We only need the data from the flat image.
        flat_data = flat_image.data

        # We need to find the initial corners (we refine them later). We
        # go through the methods by accuracy.
        initial_corners = None
        # Via the flat field itself...
        if initial_corners is None or initial_method == "flat":
            try:
                initial_corners = self._calculate_initial_slice_corners_flat(
                    flat_array=flat_data,
                )
            except logging.UndiscoveredError as error:
                logging.error(
                    error_type=logging.AlgorithmError,
                    message=(
                        "Initial corner detection via the flat field failed"
                        f" with: {type(error).__name__} : {error!s}"
                    ),
                )
                initial_corners = None
            # We may need to move on to the next method.
            if initial_corners is None:
                logging.info(
                    message=(
                        "Flat field failed, moving on to next method of corner"
                        " retrieval: a stored corner file."
                    ),
                )
                
        # Via a stored corner file.
        if initial_corners is None or initial_method == "file":
            try:
                corner_filename = "Dummy"
                initial_corners = self._calculate_initial_slice_corners_file(
                    filename=corner_filename,
                )
            except logging.UndiscoveredError as error:
                logging.error(
                    error_type=logging.AlgorithmError,
                    message=(
                        "Initial corner detection via a file table failed"
                        f" with: {type(error).__name__} : {error!s}"
                    ),
                )
                initial_corners = None
            # We may need to move on to the next method.
            if initial_corners is None:
                logging.info(
                    message=(
                        "Corner file failed, moving on to next method of corner"
                        " retrieval: the simulation."
                    ),
                )

        # Via the simulation itself.
        if initial_corners is None or initial_method == "simulation":
            try:
                initial_corners = (
                    self._calculate_initial_slice_corners_simulation()
                )
            except logging.UndiscoveredError as error:
                logging.warning(
                    warning_type=logging.AlgorithmWarning,
                    message=(
                        "Initial corner detection via the simulation failed"
                        f" with: {type(error).__name__} : {error!s}"
                    ),
                )
                initial_corners = None
            # We may need to move on to the next method.
            if initial_corners is None:
                logging.info(
                    message=(
                        "The simulation failed, moving on to next method of"
                        " corner retrieval: rasing an error."
                    ),
                )

        # If the corners were still not found, something is wrong.
        if initial_corners is None:
            # The corner was not found, not sure why.
            logging.critical(
                critical_type=logging.AlgorithmError,
                message=(
                    "Initial corners cannot be found for the provided flat"
                    " field and method. There is no way to determine where the"
                    " slice corners are."
                ),
            )

        # Otherwise, we need to refine where the slice corners are as the
        # provided methods above mostly only give a rough estimate as to where
        # they are.
        refined_slice_corners = self._refine_initial_slice_corners(
            initial_slice_corners=initial_corners,
            flat_array=flat_data,
        )

        # Renaming based on the documentation.
        slice_corners = refined_slice_corners

        # All done. If the user wanted us to overwrite the current class.
        return slice_corners

    def _corner_table_to_points(
        self: hint.Self,
        corner_table: hint.Table,
    ) -> list(tuple):
        """Convert a table of slice corners to just the corner points.

        The table of slice corners has some labeling information which allows
        for the points to have slice indexes and position labels attached to
        them. Such a format is very custom, and most other image manipulation
        software only cares about point coordinates, so this function just
        converts the table to raw (x, y) point coordinates.

        Parameters
        ----------
        corner_table : Table
            The slice corner table which we are converting to just raw points.

        Returns
        -------
        point_coordinates : list
            A list of the coordinate points, given as an (x, y) tuple pair.

        """
        # A copy, just in case.
        corner_table = copy.deepcopy(corner_table)

        # The structures for the coordinates themselves.
        x_coordinate = []
        y_coordinate = []
        # It is just easy to go through the entire table and pull out the
        # points in "order", not that order should matter.
        corner_names = ["top_left", "top_right", "bottom_left", "bottom_right"]
        for cornerdex in corner_names:
            corner_x_values = np.array(corner_table[f"{cornerdex}_x"]).flatten()
            corner_y_values = np.array(corner_table[f"{cornerdex}_y"]).flatten()
            x_coordinate = x_coordinate + corner_x_values.tolist()
            y_coordinate = y_coordinate + corner_y_values.tolist()

        # Repackaging them per the documentation.
        point_coordinates = list(zip(x_coordinate, y_coordinate, strict=True))
        return point_coordinates

    def retrieve_slice(
        self: hint.Self,
        image: hint.LezargusImage,
        slice_index: int,
        buffer_width: int = 0,
        rebin: bool = True,
        force_width: int | None = None,
    ) -> hint.LezargusImage:
        """Fetch/retrieve the slice based on the slice corners.

        We extract a slice based on its corners as defined by the current
        flat field. The corners are used as initial anchors which are used to
        define the region we are extracting. Slice rotation and fractional
        pixel flux is handled as well.

        Parameters
        ----------
        image : LezargusImage
            The image which we are using to fetching the slice data from.
            Typically a data image, it can be a flat or an arc lamp image.
        slice_index : int
            The slice index which we are fetching from. This is a 1-indexed
            slice index number.
        buffer_width : int, default = 0
            The number of pixels to buffer on each side from the pixel corners.
            We default to not having a buffer but it is a bad idea.
        rebin : bool, default = True
            If True, we rebin each wavelength slice to take into account
            fractional pixel light and/or the forced width.
        force_width : int, default = None
            The width of the slice after trimming the buffer is determined
            automatically but may be forced to a specific width using this
            parameter.

        Returns
        -------
        fetched_slice : LezargusImage
            The image that contains the slice based on the corners detected
            by the predefined slice and further preprocessing.

        """
        # For the given slice, we need to find the corner.
        # Just making sure it exists, though this check is also done later
        # during rough extraction.
        if self.slice_corners is None:
            logging.error(
                error_type=logging.WrongOrderError,
                message=(
                    "There are no slice corners, we cannot fetch the slice"
                    " based on the slice corners. Replace the flat field."
                ),
            )

        # A buffer is highly suggested to avoid any data loss.
        if buffer_width <= 1:
            logging.warning(
                warning_type=logging.AccuracyWarning,
                message=(
                    f"The buffer {buffer_width} is small (i.e. <= 1) and may"
                    " result in edge artifacts in the retrieved slice."
                ),
            )

        # We are attempting to force a width without rebinning. This will
        # likely fail on the stacking of wavelength rows if there are
        # different slice row widths.
        if (not rebin) and (force_width is not None):
            logging.error(
                error_type=logging.AlgorithmError,
                message=(
                    "Without rebinning, we cannot force a slice to have a"
                    " specific width. For point-cloud based extraction, see"
                    " `retrieve_point_cloud`."
                ),
            )

        # We first rough retrieve the slice so we know what we are working
        # with. We also need the flat field itself.
        rough_slice_image = self.rough_retrieve_slice(
            image=image,
            slice_index=slice_index,
            buffer_width=buffer_width,
        )
        rough_flat_image = self.rough_retrieve_slice(
            image=self.flat_image,
            slice_index=slice_index,
            buffer_width=buffer_width,
        )

        # Once we have the slice image, we only really care about the data
        # array contained within.
        rough_slice_array = rough_slice_image.data
        rough_flat_array = rough_flat_image.data

        # The slice may be rotated, we attempt to find this rotation.
        slice_rotation = self.find_slice_flat_rotation(
            flat_slice=rough_flat_image,
        )
        # And try and fix this rotation.
        rotated_slice_array = lezargus.library.transform.rotate_2d(
            array=rough_slice_array,
            rotation=-slice_rotation,
            order=1,
            mode="nearest",
        )
        rotated_flat_array = lezargus.library.transform.rotate_2d(
            array=rough_flat_array,
            rotation=-slice_rotation,
            order=1,
            mode="nearest",
        )

        # Next, we trim out any of the buffer.
        trimmed_slice_array = self._trim_slice_buffer(
            slice_array=rotated_slice_array,
            flat_array=rotated_flat_array,
            rebin=rebin,
            force_width=force_width,
        )

        # Complete, now, we just need to repack everything as needed.
        fetched_slice = lezargus.library.container.LezargusImage(
            data=trimmed_slice_array,
            uncertainty=None,
            wavelength=image.wavelength,
            wavelength_unit=image.wavelength_unit,
            data_unit=image.data_unit,
            spectral_scale=image.spectral_scale,
            pixel_scale=image.pixel_scale,
            slice_scale=image.slice_scale,
            mask=None,
            flags=None,
            header=image.header,
        )
        # All done.
        return fetched_slice

    def rough_retrieve_slice(
        self: hint.Self,
        image: hint.LezargusImage,
        slice_index: int,
        buffer_width: int = 0,
    ) -> hint.LezargusImage:
        """Fetch/retrieve the rough slice based on the slice corners.

        This function just excises the image based on the corners, plus some
        pixel width buffer (which is used for somethings). It is not an
        advanced retrieval of the slice (but it is a part in the whole thing).

        In the event of the corner pixels not being completely square
        (i.e. the slice itself is rotated or crooked), please lean for the
        largest bounding box by increasing the buffer.

        Parameters
        ----------
        image : LezargusImage
            The image which we are using to fetching the slice data from.
            Typically a data image, it can be a flat or an arc lamp image.
        slice_index : int
            The slice index which we are fetching from. This is a 1-indexed
            slice index number.
        buffer_width : int, default = 0
            The number of pixels to buffer on each side from the pixel corners.
            We default to not having a buffer.

        Returns
        -------
        fetched_slice : LezargusImage
            The image that contains the slice based on the corners detected
            by the predefined slice, plus the buffer provided.

        """
        # For the given slice, we need to find the corner.
        # Just making sure it exists.
        if self.slice_corners is None:
            logging.error(
                error_type=logging.WrongOrderError,
                message=(
                    "There are no slice corners, we cannot fetch the slice"
                    " based on the slice corners. Replace the flat field."
                ),
            )
        # Otherwise, we just need to search the table for the slice corner
        # coordinates.
        slice_corner_row = self.slice_corners[
            self.slice_corners["slice"] == slice_index
        ]
        if len(slice_corner_row) == 0:
            logging.error(
                error_type=logging.InputError,
                message=(
                    f"Invalid slice index {slice_index}, no matching entry in"
                    " slice corner table."
                ),
            )

        # Fetching the corner coordinates, and removing any excess dimensions.
        top_left_corner = np.array(
            [slice_corner_row["top_left_x"], slice_corner_row["top_left_y"]],
        ).squeeze()
        top_right_corner = np.array(
            [slice_corner_row["top_right_x"], slice_corner_row["top_right_y"]],
        ).squeeze()
        bottom_left_corner = np.array(
            [
                slice_corner_row["bottom_left_x"],
                slice_corner_row["bottom_left_y"],
            ],
        ).squeeze()
        bottom_right_corner = np.array(
            [
                slice_corner_row["bottom_right_x"],
                slice_corner_row["bottom_right_y"],
            ],
        ).squeeze()

        # Adding the buffer specified, taking into account the coordinates
        # and which corner is actually being modified.
        top_left_corner += np.array([-buffer_width, buffer_width])
        top_right_corner += np.array([buffer_width, buffer_width])
        bottom_left_corner += np.array([-buffer_width, -buffer_width])
        bottom_right_corner += np.array([buffer_width, -buffer_width])

        # Of course, the corners actually do not matter so much as the furthest
        # right and left, and top and bottom.
        left_edge = np.floor(
            np.min([top_left_corner[0], bottom_left_corner[0]]),
        )
        right_edge = np.ceil(
            np.max([top_right_corner[0], bottom_right_corner[0]]),
        )
        top_edge = np.ceil(np.max([top_left_corner[1], top_left_corner[1]]))
        bottom_edge = np.floor(
            np.min([bottom_left_corner[1], bottom_left_corner[1]]),
        )

        # Ensuring integers.
        left_edge = int(left_edge)
        right_edge = int(right_edge)
        top_edge = int(top_edge)
        bottom_edge = int(bottom_edge)

        # Provided the corner edges, we can now sub-image the image array.
        fetched_slice = image.subimage(
            x_span=[left_edge, right_edge],
            y_span=[bottom_edge, top_edge],
        )

        # All done.
        return fetched_slice

    def _trim_slice_buffer(
        self: hint.Self,
        slice_array: hint.NDArray,
        flat_array: hint.NDArray,
        rebin: bool = True,
        force_width: int | None = None,
    ) -> hint.NDArray:
        """Trim the left-right buffer from a rough retrieval slice.

        Rough retrieval of the slices involve a buffer provided to have some
        background and to ensure no pixels are lost. We trim out the buffer
        here using thresholding to return only the relevant data. This function
        will not trim any top and bottom buffer, but as those are in the
        spectral dimension, edge effects and spectral stitching algorithms
        should handle that later on.

        Parameters
        ----------
        slice_array : NDArray
            The rough retrieved slice data array with the buffer built in.
        flat_array : NDArray
            The rough retrieved slice flat array. The flat is used to determine
            which parts are relevant data and which parts are background.
        rebin : bool, default = True
            If True, we rebin each wavelength slice to take into account
            fractional pixel light.
        force_width : int, default = None
            The width of the slice after trimming is determined automatically
            but may be forced to a specific width using this parameter.

        Returns
        -------
        trim_slice : NDArray
            The trimmed slice data array.

        """
        # Ensuring they are arrays.
        slice_array = np.asarray(slice_array)
        flat_array = np.asarray(flat_array)

        # Where we store the output. We keep track of the widths just in case
        # there is a mismatch; which happens more often than not.
        trim_slice_rows = []
        raw_row_widths = []

        # To trim the array, we just loop over each row (over wavelength) and
        # trim the rows wavelength by wavelength, then assemble them to a
        # proper clean array of similar dimensions.
        for index in range(flat_array.shape[0]):
            # Getting the rows at this given wavelength.
            slice_row = slice_array[index, :]
            flat_row = flat_array[index, :]
            # Trimming the buffer from this wavelength row.
            trim_row = self._trim_slice_buffer_wavelength_row(
                slice_row=slice_row,
                flat_row=flat_row,
                rebin=rebin,
                force_width=force_width,
            )
            trim_width = trim_row.size

            # Done, assembling the outputs.
            trim_slice_rows.append(trim_row)
            raw_row_widths.append(trim_width)

        # The main row width is just the one that is the most common, and will
        # result in the least amount of lost and excess data. We do not really
        # care about the count for the mode.
        raw_row_widths = np.array(raw_row_widths, dtype=int)
        main_row_width, __ = scipy.stats.mode(
            raw_row_widths,
            axis=None,
            nan_policy="omit",
        )
        main_row_width = int(main_row_width)

        # Attempt to see if we can assemble all of the rows into one single
        # image. This really only works if all of the row widths are the same.
        if np.all(raw_row_widths == main_row_width):
            trim_slice = np.array(trim_slice_rows, dtype=slice_array.dtype)
        else:
            # There is a mismatch in the widths of the rows. We try again,
            # and force the width to the most sensible value we have so far
            # found.
            trim_slice = self._trim_slice_buffer(
                slice_array=slice_array,
                flat_array=flat_array,
                rebin=rebin,
                force_width=main_row_width,
            )

        # All done.
        return trim_slice

    def _trim_slice_buffer_wavelength_row(
        self: hint.Self,
        slice_row: hint.NDArray,
        flat_row: hint.NDArray,
        rebin: bool = True,
        force_width: int | None = None,
    ) -> hint.NDArray:
        """Trim the slice wavelength row from a rough retrieval flat row.

        Rough retrieval of the slices involve a buffer provided to have some
        background and to ensure no pixels are lost. We trim out the buffer
        here using thresholding to return only the relevant data. The
        threshold is determined automatically; but the width of the
        important parts of a slice wavelength row can be forced.

        Parameters
        ----------
        slice_row : NDArray
            The 1D slice wavelength row data which we are going to trim.
        flat_row : NDArray
            The 1D slice wavelength row flat data which we will use to
            determine where the important parts of the data is.
        rebin : bool, default = True
            If True, we rebin each wavelength slice to take into account
            fractional pixel light.
        force_width : int, default = None
            If provided, the threshold value is adjusted to ensure that the
            width/size of the row is as provided. If None, the width is
            based on automatic thresholding.

        Returns
        -------
        trim_slice_row : NDArray
            The trimmed slice data row, with the size determined based on if
            it was forced or not.

        """
        # We check if the force width was provided, ensuring its type.
        force_width = int(force_width) if force_width is not None else None
        slice_row = np.ravel(slice_row)
        flat_row = np.ravel(flat_row)

        # Ensure the lengths are the same.
        if slice_row.shape != flat_row.shape:
            logging.error(
                error_type=logging.InputError,
                message=(
                    f"Slice row shape {slice_row.shape} and flat row shape"
                    f" {flat_row.shape} do not match."
                ),
            )

        # Determine the threshold value. The ISODATA algorithm is a good
        # starting point; however, it can be a little high for the edge
        # pixels so we adjust it a little.
        correction_factor = 1 / 2
        raw_threshold = self.__isodata_threshold_algorithm(array=flat_row)
        threshold = raw_threshold * correction_factor

        # The finding the pixels which are to be included using the threshold.
        thresh_mask = flat_row >= threshold
        thresh_slice_row = slice_row[thresh_mask]
        thresh_flat_row = flat_row[thresh_mask]

        # Now to check if we should rebin the row based on the edge pixel
        # flux values to attempt to take into account fractional pixel light.
        if rebin:
            # Rebinning, and the overall width of the slice is either forced,
            # or not.
            rebin_slice_width = (
                thresh_slice_row.size if force_width is None else force_width
            )
            trim_slice_row = self._trim_slice_rebin_row(
                slice_row=thresh_slice_row,
                flat_row=thresh_flat_row,
                slice_width=rebin_slice_width,
            )
        else:
            # No rebinning.
            trim_slice_row = thresh_slice_row

        # All done.
        return trim_slice_row

    def _trim_slice_rebin_row(
        self: hint.Self,
        slice_row: hint.NDArray,
        flat_row: hint.NDArray,
        slice_width: int,
    ) -> hint.NDArray:
        """Rebin a trimmed slice to adapt for fractional pixel effects.

        Rough retrieval of the slices involve a buffer provided to have some
        background and to ensure no pixels are lost. We trim out the buffer
        here using thresholding to return only the relevant data. As the
        actual slice may not line up with the pixels, this function adapts and
        tries to extract information from partial pixels.

        Parameters
        ----------
        slice_row : NDArray
            The rough retrieved slice data array wavelength row. This is the
            data we are rebinning.
        flat_row : NDArray
            The rough retrieved slice flat array wavelength row. The flat is
            used to determine which parts are relevant data and which parts
            are background and the fractional pixel light we are going to
            rebin.
        slice_width : int
            The width of the slice which we are rebinning to.

        Returns
        -------
        rebin_slice : NDArray
            The rebinned trimmed slice data array.

        """
        # To do...
        logging.error(
            error_type=logging.ToDoError,
            message=(
                "Trim slice rebinning needs to be done. Doing a bad"
                " interpolation job."
            ),
        )
        # Just simple dumb interpolation to "get it to work".
        raw_index = np.arange(flat_row.size) + 1
        raw_flux = slice_row
        out_index = np.linspace(min(raw_index), max(raw_index), slice_width)
        out_flux = np.interp(out_index, raw_index, raw_flux)
        slice_row = out_flux

        return slice_row

    @staticmethod
    def __isodata_threshold_algorithm(array: hint.NDArray) -> float:
        """Compute the ISODATA threshold value.

        This is a wrapper function for the ISODATA algorithm for determining
        a proper threshold value. It is a simple and likely stable algorithm
        suited for determining the slice for each wavelength row. We only
        handle the case for 2 groups: the data and the background.

        For more information on the ISODATA algorithm, see:
        - https://doi.org/10.1109/TSMC.1980.4308400
        -

        Parameters
        ----------
        array : NDArray
            The array which we use to calculate the threshold value.

        Returns
        -------
        threshold : float
            The ISODATA threshold value.

        """
        # Ensuring it is an array.
        array = np.asarray(array)

        # The algorithm is iterative, being similar to 1D k-means clustering.
        # We do not want it to loop forever though; it really should only
        # go a few times.
        current_iterations = 0
        max_iterations = 100
        # Initial conditions.
        threshold = None
        current_threshold = float((np.nanmin(array) + np.nanmax(array)) / 2)
        # Doing the loop.
        while current_iterations <= max_iterations:
            current_iterations += 1
            # Splitting it into two groups.
            data_half = array[array >= current_threshold]
            background_half = array[array < current_threshold]
            # Recomputing the new threshold and checking if it changed, thus
            # requiring another iteration.
            data_average = np.nanmedian(data_half)
            background_average = np.nanmedian(background_half)
            new_threshold = (data_average + background_average) / 2
            if np.isclose(current_threshold, new_threshold):
                # No change in groups, partition threshold found.
                threshold = new_threshold
                break
            current_threshold = new_threshold

        # Check to see if we need to warn about a threshold not found.
        if threshold is None:
            logging.warning(
                warning_type=logging.AlgorithmWarning,
                message=(
                    "Retrieval ISODATA threshold did not converge in"
                    f" {current_iterations} iterations."
                ),
            )
            # Giving it the most recent version, the best we can do.
            threshold = current_threshold

        # All done.
        return threshold

    def find_slice_flat_rotation(
        self: hint.Self,
        flat_slice: hint.LezargusImage,
    ) -> float:
        """Find the image rotation of a slice image.

        A slice image (i.e. a sub-image of the lamp flat) is often rotated
        relative to the main array. We find the rotation of the slice by
        finding the bounding rotating rectangle with minimum area using OpenCV.

        Parameters
        ----------
        flat_slice : LezargusImage
            The slice flat image data which we are going to find the
            rotation for.

        Returns
        -------
        rotation : float
            The rotation of the slice image, in radians. This value is
            typically small.

        """
        # We only need the data array to determine the flat slice rotation,
        # the other parts of the data are not too useful.
        flat_array = np.asarray(flat_slice.data, copy=True)

        # Finding the rotation is slightly iterative, we continue to calculate
        # and rotate until we zero in on no rotation at all. We want an vague
        # upper limit though for the computations, just in case.
        current_array = flat_array
        rotation_steps = []
        zero_rotation = False
        current_iterations = 0
        max_iterations = 10
        # Doing it...
        while not zero_rotation and (current_iterations <= max_iterations):
            # Step in iteration.
            current_iterations += 1
            # Attempt to find the next rotation iteration.
            current_rotation = self._rough_find_slice_flat_rotation(
                flat_array=current_array,
            )
            rotation_steps.append(current_rotation)

            # Check if done, otherwise a next iteration is needed.
            if np.isclose(current_rotation, 0.0):
                zero_rotation = True
            else:
                # Fix the current rotation for the next iteration.
                zero_rotation = False
                current_array = lezargus.library.transform.rotate_2d(
                    current_array,
                    rotation=-current_rotation,
                    mode="nearest",
                    order=1,
                )

        # If the stop was the iterations and not the zero rotations, might
        # be an issue.
        if not zero_rotation:
            logging.warning(
                warning_type=logging.AccuracyWarning,
                message=(
                    "Could not converge on slice flat rotation in"
                    f" {current_iterations} iterations."
                ),
            )

        # Combine all of the rotation steps for the final resulting rotation.
        rotation = float(np.nansum(rotation_steps))
        return rotation

    def _rough_find_slice_flat_rotation(
        self: hint.Self,
        flat_array: hint.NDArray,
    ) -> float:
        """Find the rough image rotation of a slice flat image.

        We find the rotation of the slice flat by finding the bounding rotating
        rectangle with minimum area using OpenCV. However, this only finds the
        rough rotation, there is a multiplicity and degeneracy of valid
        rectangles, so the full rotation needs to be found iteratively best
        handled by the parent function :py:func:`find_slice_rotation`.

        Parameters
        ----------
        flat_array : NDArray
            The slice flat array data which we are going to find the
            rotation for.

        Returns
        -------
        rotation : float
            The rotation of the slice image, in radians. This value is
            typically small. 

        """
        # The axis conventions of OpenCV/images and Numpy arrays require the
        # transposition of the array to match.
        flat_array = np.flip(flat_array).T

        # In order to use bounding rectangles in OpenCV, we need a binary
        # image which we get by thresholding. OpenCV takes very specific
        # data types so we need to convert.
        flat_array_uint16 = np.array(flat_array, dtype=np.uint16)

        # Thresholding, the Otsu method works best with two "peaks" in the
        # histogram which is kind of what the flats look like. As it
        # automatically determines the thresholding, we do not need to
        # supply one.
        threshold_fill_value = 255
        threshold_partition = np.nan
        threshold_partition, binary_array = cv2.threshold(
            flat_array_uint16,
            threshold_partition,
            threshold_fill_value,
            cv2.THRESH_OTSU,
        )
        # Finding the contours of the slice itself. The contour function
        # requires another retyping. There really only should be one contour.
        # We do not care about any hierarchy in the contours here as there
        # should be only one.
        binary_array_uint8 = np.array(binary_array, dtype=np.uint8)
        contours, __ = cv2.findContours(
            binary_array_uint8,
            cv2.RETR_LIST,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        # Checking if we have just the expected one contour.
        if len(contours) == 1:
            pass
        else:
            logging.error(
                error_type=logging.AlgorithmError,
                message=(
                    "There should only be 1 valid contour for a slice flat"
                    f" lamp, there are {len(contours)}. Otsu's method is"
                    " currently unreliable."
                ),
            )
            logging.warning(
                warning_type=logging.AlgorithmWarning,
                message=(
                    "Switching from Otsu threshold to Edge border threshold."
                ),
            )
            # We try again with thresholds defined by the very edge of the
            # arrays...
            # The edge of the slice establishes our floor...
            threshold_data = np.array(
                [
                    flat_array_uint16[0, :].ravel().tolist()
                    + flat_array_uint16[-1, :].ravel().tolist()
                    + flat_array_uint16[:, 0].ravel().tolist()
                    + flat_array_uint16[:, -1].ravel().tolist(),
                ],
                dtype=flat_array_uint16.dtype,
            )
            # ...and with an offset to ensure that the background
            # (i.e. <10% of the flat) is the background.
            background_percentage = 10
            threshold_offset = np.nanpercentile(flat_array_uint16, 75) * (
                background_percentage / 100
            )
            # The new partition.
            threshold_partition = int(
                np.nanmedian(threshold_data) + threshold_offset,
            )
            threshold_partition, binary_array = cv2.threshold(
                flat_array_uint16,
                threshold_partition,
                threshold_fill_value,
                cv2.THRESH_BINARY,
            )
            # And we try again.
            binary_array_uint8 = np.array(binary_array, dtype=np.uint8)
            contours, __ = cv2.findContours(
                binary_array_uint8,
                cv2.RETR_LIST,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            if len(contours) != 1:
                logging.error(
                    error_type=logging.AlgorithmError,
                    message=(
                        "Cannot find appropriate contours via thresholding."
                        " Rotation found is likely inaccurate."
                    ),
                )
                logging.warning(
                    warning_type=logging.AccuracyWarning,
                    message=(
                        "Rotation inaccurate due to poor contours found."
                        " Increasing buffer width may help."
                    ),
                )
        # The main contour we will be using, even if there are more than one.
        contour = contours[0]

        # Finding the minimum area bounding rectangle, allowing for rotation.
        # We use the rectangle defined to find the rotation.
        rotated_rectangle = cv2.minAreaRect(contour)
        rotation_degree = float(rotated_rectangle[2])

        # The rotation has degeneracies across the 90 degree rotations of a
        # rectangle. It should, by that nature as well, be no more than 90
        # degrees in either direction
        right_angle = 90.0
        if np.isclose(rotation_degree, -right_angle):
            # By the OpenCV documentation, this is the 0-rotation case as
            # It cannot be 0 itself due to the [-90, 0) defined range.
            rotation_degree = 0.0
        elif np.isclose(rotation_degree, +right_angle):
            # The positive case is what usually happens but is confusing
            # to Sparrow.
            rotation_degree = 0.0
        elif np.abs(rotation_degree) >= right_angle:
            # The angle should still be zero but may have not been caught due
            # to floating point issues. We warn just in case it is something
            # else.
            logging.warning(
                warning_type=logging.AccuracyWarning,
                message=(
                    f"Slice rotation magnitude |{rotation_degree}| is 90"
                    " degrees or more; most likely it is actually 0."
                ),
            )
            rotation_degree = rotation_degree % right_angle
        elif rotation_degree > 0:
            logging.error(
                error_type=logging.AlgorithmError,
                message=(
                    f"Initial slice rotation output from OpenCV should be"
                    f" negative due to the range: {rotation_degree}"
                ),
            )
            logging.error(
                error_type=logging.DevelopmentError,
                message=(
                    "OpenCV function minAreaRect may have changed, code needs"
                    " updating."
                ),
            )

        # OpenCV has the bound of angles between 0 and 90; for a negative
        # rotation, the rotation is read from 90. We assume a 45 degree angle
        # is the main demarcation between a positive or negative rotation.
        if np.abs(rotation_degree) <= right_angle / 2:
            signed_rotation_degree = rotation_degree
        elif np.abs(rotation_degree) >= right_angle / 2:
            signed_rotation_degree = rotation_degree + right_angle
        else:
            logging.error(
                error_type=logging.LogicFlowError,
                message="Rotation angle octant has no covering case.",
            )
            signed_rotation_degree = np.nan

        # Converting to radians, the unit convention for angles.
        rotation = np.deg2rad(signed_rotation_degree)

        # All done.
        return rotation
