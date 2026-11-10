"""Make functions to create the detectors and detector simulators.

This module contains functions which, provided a specific detector
specification, outputs a detector simulator.
"""

# isort: split
# Import required to remove circular dependencies from type checking.
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lezargus.library import hint
# isort: split

import astropy.table
import numpy as np
import yaml

import lezargus
from lezargus.data._make import functionality
from lezargus.library import logging


def make_detector(
    channel: hint.Literal["visible", "nearir", "midir"],  # noqa: F821, UP037
) -> hint.DetectorArray:
    """Load a detector files programmatically to make the detector.

    Note, the format of the detector specification files are
    very specific. User usage of the this function is discouraged.

    Parameters
    ----------
    channel : str
        The channel name for the channel the detector is in that we want to
        make.

    Returns
    -------
    detector_array : DetectorArray
        The detector simulation object.

    """
    # Ensuring correct channel name.
    if channel not in ["visible", "nearir", "midir"]:
        logging.error(
            error_type=logging.InputError,
            message=(
                f"Detector creation channel {channel} is not a valid channel."
            ),
        )
        logging.critical(
            critical_type=logging.DevelopmentError,
            message=(
                "The wrong channel was coded in to the detector _make code."
            ),
        )

    # First, initial detector information needed to make the class.
    detector_yaml_filename = functionality.find_data_filename(
        basename=f"{channel}_detector_properties.yaml",
    )
    with open(detector_yaml_filename) as yaml_file:
        detector_properties = yaml.safe_load(yaml_file)

    # Creating the detector.
    detector_array = lezargus.library.container.DetectorArray(
        detector_shape=detector_properties["detector_shape"],
        pixel_size=detector_properties["pixel_size"],
        detector_gain=detector_properties["gain"],
        nondestructive_reads=detector_properties["nondestructive_reads"],
    )

    # Adding read noise values.
    detector_array.add_read_noise(
        read_noise=detector_properties["read_noise_mean"],
        read_noise_stddev=detector_properties["read_noise_stddev"],
    )

    # Adding cosmic ray values.
    detector_array.add_cosmic_ray_rates(
        cosmic_ray_rate=detector_properties["cosmic_ray_rate"],
        cosmic_ray_value=detector_properties["cosmic_ray_value"],
    )

    # Adding bias frames.
    bias_mean_filename = functionality.find_data_filename(
        basename=f"{channel}_bias_frame_mean.npy",
    )
    bias_stddev_filename = functionality.find_data_filename(
        basename=f"{channel}_bias_frame_stddev.npy",
    )
    bias_mean_frame = np.load(bias_mean_filename)
    bias_stddev_frame = np.load(bias_stddev_filename)
    # Using raw_image functionality would make the compiled build too big.
    detector_array.add_bias_frame(
        bias_frame_mean=bias_mean_frame,
        bias_frame_stddev=bias_stddev_frame,
    )

    # Adding flat frames.
    flat_mean_filename = functionality.find_data_filename(
        basename=f"{channel}_flat_frame_mean.npy",
    )
    flat_stddev_filename = functionality.find_data_filename(
        basename=f"{channel}_flat_frame_stddev.npy",
    )
    flat_mean_frame = np.load(flat_mean_filename)
    flat_stddev_frame = np.load(flat_stddev_filename)
    # Using raw_image functionality would make the compiled build too big.
    detector_array.add_flat_frame(
        flat_frame_mean=flat_mean_frame,
        flat_frame_stddev=flat_stddev_frame,
    )

    # Adding dark current frames.
    dark_current_mean_filename = functionality.find_data_filename(
        basename=f"{channel}_dark_current_frame_mean.npy",
    )
    dark_current_stddev_filename = functionality.find_data_filename(
        basename=f"{channel}_dark_current_frame_stddev.npy",
    )
    dark_current_mean_frame = np.load(dark_current_mean_filename)
    dark_current_stddev_frame = np.load(dark_current_stddev_filename)
    # Using raw_image functionality would make the compiled build too big.
    detector_array.add_dark_current_frame(
        dark_current_frame_mean=dark_current_mean_frame,
        dark_current_frame_stddev=dark_current_stddev_frame,
    )

    # Adding linearity function.
    linearity_data_filename = functionality.find_data_filename(
        basename=f"{channel}_linearity_function.dat",
    )
    linearity_data_table = astropy.table.Table.read(
        linearity_data_filename,
        format="ascii.mrt",
    )
    linearity_function = lezargus.library.interpolate.Linear1DInterpolate(
        x=linearity_data_table["input"],
        v=linearity_data_table["output"],
        extrapolate=False,
    )
    detector_array.add_linearity_function(linearity_function=linearity_function)

    # Adding quantum efficiency (QE) function.
    detect_type = {
        "visible": "ccd",
        "nearir": "h2rg",
        "midir": "h2rg",
    }.get(channel)
    efficiency_data_filename = functionality.find_data_filename(
        basename=f"{channel}_{detect_type}_quantum_efficiency.dat",
    )
    efficiency_data_table = astropy.table.Table.read(
        efficiency_data_filename,
        format="ascii.mrt",
    )
    efficiency_function = lezargus.library.interpolate.Linear1DInterpolate(
        x=efficiency_data_table["wavelength"],
        v=efficiency_data_table["efficiency"],
        extrapolate=False,
    )
    detector_array.add_efficiency_function(
        efficiency_function=efficiency_function,
    )

    # Adding hot pixel map.
    hot_pixel_map_filename = functionality.find_data_filename(
        basename=f"{channel}_hot_pixel_map.npy",
    )
    hot_pixel_map = np.load(hot_pixel_map_filename)
    hot_pixel_value = detector_properties["hot_pixel_value"]
    detector_array.add_hot_pixel_map(
        hot_pixel_map=hot_pixel_map,
        hot_pixel_value=hot_pixel_value,
    )

    # Adding dead pixel map.
    dead_pixel_map_filename = functionality.find_data_filename(
        basename=f"{channel}_dead_pixel_map.npy",
    )
    dead_pixel_map = np.load(dead_pixel_map_filename)
    dead_pixel_value = detector_properties["dead_pixel_value"]
    detector_array.add_dead_pixel_map(
        dead_pixel_map=dead_pixel_map,
        dead_pixel_value=dead_pixel_value,
    )

    # Finally, recalculating the detector, just in case.
    detector_array.recalculate_detector()
    return detector_array
