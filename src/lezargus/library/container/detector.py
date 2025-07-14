"""Simulation code for modeling a detector.

We constrain ourselves to model CCD and H2RG detectors; or, more specifically,
we design the model off of simple CCD detectors and constrain H2RG parameters
to model that of a CCD-like detector.
"""

# isort: split
# Import required to remove circular dependencies from type checking.
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lezargus.library import hint
# isort: split


import time

import numpy as np
import scipy.stats

import lezargus
from lezargus.library import logging


class DetectorArray:
    """The detector class."""

    detector_shape: tuple
    """The shape of the detector, described as a tuple shape of the number 
    of pixels."""

    pixel_size: float
    """The length or the size of the pixels, in meters. We assume the pixels 
    are square and so only one measurement is needed."""

    detector_gain: float = 1
    """The detector gain value."""

    nondestructive_reads: int = 1
    """The number of nondestructive reads made to obtain a given observation.
    This value is intrinsic to the detector and it mostly affects read noise.
    Though this is valid for only H2RG-based detectors, for CCDs a value of 1 
    is appropriate."""

    raw_bias_frames: list[hint.NDArray] | None = None
    """The list of bias frames added to the detector to derive simulated 
    bias values from. If None, should no bias frames be provided, we default 
    to a zero bias frame for calculations. The bias is in ADUs."""

    bias_frame: hint.NDArray
    """The averaged bias frame for the detector. This value is typically 
    computed from the provided raw bias frames."""

    bias_stddev_frame: hint.NDArray
    """The standard deviation of the average bias frame. This value is 
    typically computed from the provided raw bias frames."""

    raw_flat_frames: list[hint.NDArray] | None = None
    """The list of flat frames added to the detector to derive simulated 
    flat values from. If None, should no flat frames be provided, we default 
    to a unity flat frame for calculations. The flat is unit-less, as an 
    efficiency factor."""

    flat_frame: hint.NDArray
    """The averaged flat frame for the detector. This value is typically 
    computed from the provided raw flat frames."""

    flat_stddev_frame: hint.NDArray
    """The standard deviation of the average flat frame. This value is 
    typically computed from the provided raw flat frames."""

    raw_dark_current_frames: list[hint.NDArray] | None = None
    """The list of dark current frames added to the detector to derive 
    simulated dark current values from. If None, should no dark current 
    frames be provided, we default to a zero dark current frame for 
    calculations. The dark current is in ADUs per second."""

    dark_current_frame: hint.NDArray
    """The averaged dark current frame for the detector. This value is 
    typically computed from the provided raw dark current frames."""

    dark_current_stddev_frame: hint.NDArray
    """The standard deviation of the average dark current frame. This value is 
    typically computed from the provided raw dark current frames."""

    raw_linearity_function: (
        hint.Callable[[hint.NDArray], hint.NDArray] | None
    ) = None
    """The added linearity of the detector. If not provided, this is None, and 
    we default to a linear linearity function."""

    raw_efficiency_function: (
        hint.Callable[[hint.NDArray], hint.NDArray] | None
    ) = None
    """The added efficiency function of the detector. This is the function 
    which determines the efficiency of the detector at a provided wavelength. 
    If not provided, this is None, and we default to a perfect, 1, efficiency 
    function."""

    read_noise: float = 0
    """The read noise of the detector's pixels. If not otherwise provided, 
    we default to a zero read noise detector."""

    read_noise_stddev: float = 0
    """The standard deviation of the read noise of the detector's pixels. If 
    not otherwise provided, we default to a zero read noise detector (thus 
    the standard deviation is also zero)."""

    raw_hot_pixel_maps: list[hint.NDArray] | None = None
    """The list of raw hot pixel maps added to the detector to derive 
    the simulated hot pixel map. If None, we assume no hot pixels at all.
    """

    hot_pixel_map: hint.NDArray
    """The hot pixel map of the detector. Where True, the pixel is considered 
    a "hot" pixel, and will have its value substituted by the typical hot 
    pixel value. """

    hot_pixel_value: float = +np.inf
    """The value of a hot pixel, where the hot pixel map determines which are 
    hot. The value is in ADUs. By default, we just use infinity, but there 
    should be much better values."""

    raw_dead_pixel_maps: list[hint.NDArray] | None = None
    """The list of raw dead pixel maps added to the detector to derive 
    the simulated dead pixel map. If None, we assume no dead pixels at all.
    """

    dead_pixel_map: hint.NDArray
    """The dead pixel map of the detector. Where True, the pixel is considered 
    a "dead" pixel, and will have its value substituted by the typical dead 
    pixel value. If None, we assume no dead pixels at all."""

    dead_pixel_value: float = 0
    """The value of a dead pixel, where the hot pixel map determines which are 
    hot. The value is in ADUs. By default, we assume 0, but there should be 
    much better values."""

    cosmic_ray_rate: float = 0
    """The rate that cosmic rays hit the surface of the Earth/detector. 
    This is used to compute a cosmic ray pixel map for where cosmic rays 
    hit the pixels. This is in cosmic rays per second per square meter."""

    cosmic_ray_value: float = +np.inf
    """The detector value for when a cosmic ray hits the detector. Though 
    we assume a near typical high value, given the variance of cosmic rays."""

    random_state_seed: int | None = None
    """The random state seed for all of the functions which need to have 
    a random number generator used. This is mostly for generating new 
    detector frames. If None, which is suggested, we always use a different 
    random state."""

    def __init__(
        self: DetectorArray,
        detector_shape: tuple,
        pixel_size: float,
        detector_gain: float = 1,
        nondestructive_reads: int = 1,
        random_state_seed: int | None = None,
    ) -> None:
        """Initialize the detector simulator.

        Parameters
        ----------
        detector_shape : tuple
            The shape of the detector, in pixels. Typically they are squares
            so the shape may be the same.
        pixel_size : float
            The length or size of a pixel, in meters. Note, we assume square
            pixels.
        detector_gain : float, default = 1
            The gain value of the detector, stored for later usage.
        nondestructive_reads : int, default = 1
            The number of nondestructive reads the detector does for typical
            exposures. If a CCD, use 1.
        random_state_seed : int, default = None
            The random state seed for the random number generator. If None,
            we just pick one at random.

        """
        # Storing the values.
        self.detector_shape = detector_shape
        self.pixel_size = pixel_size
        self.detector_gain = detector_gain
        self.nondestructive_reads = nondestructive_reads

        # The random state seed.
        if random_state_seed is not None:
            # Just a warning...
            logging.warning(
                warning_type=logging.AccuracyWarning,
                message=(
                    "The random state seed is statically set. Detector"
                    " simulations are no longer random."
                ),
            )
            self.random_state_seed = random_state_seed
        else:
            self.random_state_seed = None

        # All done.

    def get_random_state(self: hint.Self) -> int:
        """Get the random state seed for processing.

        This function acts as a middleman where if the random state is set,
        we just use that, otherwise, we make sure that random numbers can
        continue to be random numbers.

        Parameters
        ----------
        None

        Returns
        -------
        random_state : int
            The random state to use.

        """
        # If the random state is provided, then the user likely wants
        # repeatable data.
        if self.random_state_seed is not None:
            random_state = self.random_state_seed
        else:
            # Otherwise, we can emulate random numbers by supplying random seeds
            # with each function call. The overhead is minimal and this makes
            # sure that we can keep the RNG mode up-to-date. Scipy is likely to
            # shift how they do RNG numbers to keep up with Numpy's generator
            # changes.
            min_seed = 0
            max_seed = time.time_ns() % (2**32 - 1)
            random_generator = np.random.default_rng()
            random_integer = random_generator.integers(min_seed, max_seed)
            random_state = random_integer
        return random_state

    def add_bias_frame(self: hint.Self, bias_frame: hint.NDArray) -> None:
        """Add a detector bias frame to the simulated detector.

        Parameters
        ----------
        bias_frame : NDArray
            The bias frame which we are going to add.

        Returns
        -------
        None

        """
        # We just need to check it is of proper shape.
        input_bias_frame = np.array(bias_frame, copy=True)
        if input_bias_frame.shape != self.detector_shape:
            # It is wrong, we do not do anything as we cannot add it.
            logging.error(
                error_type=logging.InputError,
                message=(
                    "Input bias frame shape"
                    f" {input_bias_frame.shape} mismatches the detector shape"
                    f" {self.detector_shape}. Will not add frame."
                ),
            )
            return

        if self.raw_bias_frames is None:
            # This is the first frame.
            self.raw_bias_frames = [
                input_bias_frame,
            ]
        else:
            # We just add it to the rest.
            self.raw_bias_frames.append(input_bias_frame)

        # All done.
        return

    def add_flat_frame(self: hint.Self, flat_frame: hint.NDArray) -> None:
        """Add a detector flat frame to the simulated detector.

        Parameters
        ----------
        flat_frame : NDArray
            The flat frame which we are going to add.

        Returns
        -------
        None

        """
        # We just need to check it is of proper shape.
        input_flat_frame = np.array(flat_frame, copy=True)
        if input_flat_frame.shape != self.detector_shape:
            # It is wrong, we do not do anything as we cannot add it.
            logging.error(
                error_type=logging.InputError,
                message=(
                    "Input flat frame shape"
                    f" {input_flat_frame.shape} mismatches the detector shape"
                    f" {self.detector_shape}. Will not add frame."
                ),
            )
            return

        if self.raw_flat_frames is None:
            # This is the first frame.
            self.raw_flat_frames = [
                input_flat_frame,
            ]
        else:
            # We just add it to the rest.
            self.raw_flat_frames.append(input_flat_frame)

        # All done.
        return

    def add_dark_current_frame(
        self: hint.Self,
        dark_current_frame: hint.NDArray,
    ) -> None:
        """Add a detector dark current frame to the simulated detector.

        Parameters
        ----------
        dark_current_frame : NDArray
            The dark current frame which we are going to add.

        Returns
        -------
        None

        """
        # We just need to check it is of proper shape.
        input_dark_current_frame = np.array(dark_current_frame, copy=True)
        if input_dark_current_frame.shape != self.detector_shape:
            # It is wrong, we do not do anything as we cannot add it.
            logging.error(
                error_type=logging.InputError,
                message=(
                    "Input dark current frame shape"
                    f" {input_dark_current_frame.shape} mismatches the detector"
                    f" shape {self.detector_shape}. Will not add frame."
                ),
            )
            return

        if self.raw_dark_current_frames is None:
            # This is the first frame.
            self.raw_dark_current_frames = [
                input_dark_current_frame,
            ]
        else:
            # We just add it to the rest.
            self.raw_dark_current_frames.append(input_dark_current_frame)

        # All done.
        return

    def add_linearity_function(
        self: hint.Self,
        linearity_function: hint.Callable[[float], float],
    ) -> None:
        """Add a detector linearity function to the simulated detector.

        Parameters
        ----------
        linearity_function : NDArray
            The linearity function which we are going to add.

        Returns
        -------
        None

        """
        # It is kind of hard to test a linearity function. We assume the
        # user knows what they are doing.
        self.raw_linearity_function = linearity_function

    def add_efficiency_function(
        self: hint.Self,
        efficiency_function: hint.Callable[[float], float],
    ) -> None:
        """Add an efficiency function to the simulated detector.

        The efficiency function is a function of the efficiency of the
        detector over wavelength. The efficiency of the detector is typically
        due to the quantum efficiency function of the detector but it
        nessecilarily need not be limited to it. Any efficiency detriment
        is provided here.

        Parameters
        ----------
        efficiency_function : NDArray
            The efficiency function which we are going to add.
            The input of

        Returns
        -------
        None

        """
        # It is kind of hard to test a efficiency function. We assume
        # the user knows what they are doing.
        self.raw_efficiency_function = efficiency_function

    def add_read_noise(
        self: hint.Self,
        read_noise: float,
        read_noise_stddev: float = 0,
    ) -> None:
        """Add the detector read noise to the simulated detector.

        Parameters
        ----------
        read_noise : float
            The read noise of a given pixel within the simulated detector.
        read_noise_stddev : float, default = 0
            The standard deviation of the read noise of a single pixel within
            the simulated detector.

        Returns
        -------
        None

        """
        # Adding the parameters.
        self.read_noise = read_noise
        self.read_noise_stddev = read_noise_stddev

    def add_hot_pixel_map(
        self: hint.Self,
        hot_pixel_map: hint.NDArray,
        hot_pixel_value: float | None,
    ) -> None:
        """Add the detector hot pixel map to the simulated detector.

        Parameters
        ----------
        hot_pixel_map : NDArray
            The hot pixel map as a boolean array. Where True, the pixel is
            considered hot.
        hot_pixel_value : float, default = None
            The hot pixel value, used when a pixel is hot. If not provided,
            we default to the current hot pixel value.

        Returns
        -------
        None

        """
        # Determining what the hot pixel value provided is.
        if hot_pixel_value is None:
            # None provided so we just stick with the default.
            input_hot_pixel_value = self.hot_pixel_value

        # We just need to check it is of proper shape.
        input_hot_pixel_map = np.array(hot_pixel_map, copy=True, dtype=bool)
        if input_hot_pixel_map.shape != self.detector_shape:
            # It is wrong, we do not do anything as we cannot add it.
            logging.error(
                error_type=logging.InputError,
                message=(
                    "Input hot pixel map shape"
                    f" {input_hot_pixel_map.shape} mismatches the detector"
                    f" shape {self.detector_shape}. Will not add the map."
                ),
            )
            return

        # Otherwise.
        if self.raw_hot_pixel_maps is None:
            # This is the first frame.
            self.raw_hot_pixel_maps = [
                input_hot_pixel_map,
            ]
        else:
            # We just add it to the rest.
            self.raw_hot_pixel_maps.append(input_hot_pixel_map)
        # And the value.
        self.hot_pixel_value = input_hot_pixel_value

    def add_dead_pixel_map(
        self: hint.Self,
        dead_pixel_map: hint.NDArray,
        dead_pixel_value: float | None,
    ) -> None:
        """Add the detector dead pixel map to the simulated detector.

        Parameters
        ----------
        dead_pixel_map : NDArray
            The dead pixel map as a boolean array. Where True, the pixel is
            considered dead.
        dead_pixel_value : float, default = None
            The dead pixel value, used when a pixel is dead. If not provided,
            we default to the current dead pixel value.

        Returns
        -------
        None

        """
        # Determining what the dead pixel value provided is.
        if dead_pixel_value is None:
            # None provided so we just stick with the default.
            input_dead_pixel_value = self.dead_pixel_value

        # We just need to check it is of proper shape.
        input_dead_pixel_map = np.array(dead_pixel_map, copy=True, dtype=bool)
        if input_dead_pixel_map.shape != self.detector_shape:
            # It is wrong, we do not do anything as we cannot add it.
            logging.error(
                error_type=logging.InputError,
                message=(
                    "Input dead pixel map shape"
                    f" {input_dead_pixel_map.shape} mismatches the detector"
                    f" shape {self.detector_shape}. Will not add the map."
                ),
            )
            return

        # Otherwise.
        if self.raw_dead_pixel_maps is None:
            # This is the first frame.
            self.raw_dead_pixel_maps = [
                input_dead_pixel_map,
            ]
        else:
            # We just add it to the rest.
            self.raw_dead_pixel_maps.append(input_dead_pixel_map)
        # And the value.
        self.dead_pixel_value = input_dead_pixel_value

    def add_cosmic_ray_rates(
        self: hint.Self,
        cosmic_ray_rate: float,
        cosmic_ray_value: float | None = None,
    ) -> None:
        """Add the rate of cosmic rays hitting the detector's pixels.

        Parameters
        ----------
        cosmic_ray_rate : float
            The rate that cosmic rays are hitting the detector. The units
            are cosmic ray per second per square meter.
        cosmic_ray_value : float, default = None
            If a cosmic ray does hit the detector, then we add this value;
            the expected detector value a cosmic ray imparts on the detector.
            If None, then we assume an arbitrary high value.

        Returns
        -------
        None

        """
        # Adding the rate.
        self.cosmic_ray_rate = cosmic_ray_rate
        # And, if provided, the value.
        self.cosmic_ray_value = (
            +np.nan if cosmic_ray_value is None else cosmic_ray_value
        )

    def recalculate_detector(self: hint.Self) -> None:
        """Recalculate the detector frames from the input raw frames.

        Typically, only the average frame data is needed. However, we do not
        want to always recalculate the average frame data from the raw
        data every time. It would take too much time and it would also not
        allow any overwrites. So, we do it here.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        # We need to recalculate all of the frames. We do it in order...

        # Averaging the bias frame...
        if self.raw_bias_frames is None or len(self.raw_bias_frames) == 0:
            # There is no raw bias frames to use to compute.
            self.bias_frame = np.zeros(self.detector_shape)
            self.bias_stddev_frame = np.zeros(self.detector_shape)
        else:
            # Computing the average bias and its deviation.
            self.bias_frame = np.nanmean(self.raw_bias_frames, axis=0)
            self.bias_stddev_frame = np.nanstd(self.raw_bias_frames, axis=0)

        # Averaging the flat frame...
        if self.raw_flat_frames is None or len(self.raw_flat_frames) == 0:
            # There is no raw flat frames to use to compute.
            self.flat_frame = np.ones(self.detector_shape)
            self.flat_stddev_frame = np.zeros(self.detector_shape)
        else:
            # Computing the average bias and its deviation.
            self.flat_frame = np.nanmean(self.raw_flat_frames, axis=0)
            self.flat_stddev_frame = np.nanstd(self.raw_flat_frames, axis=0)

        # Averaging the dark current frame...
        if (
            self.raw_dark_current_frames is None
            or len(self.raw_dark_current_frames) == 0
        ):
            # There is no raw flat frames to use to compute.
            self.dark_current_frame = np.zeros(self.detector_shape)
            self.dark_current_stddev_frame = np.zeros(self.detector_shape)
        else:
            # Computing the average bias and its deviation.
            self.dark_current_frame = np.nanmean(
                self.raw_dark_current_frames,
                axis=0,
            )
            self.dark_current_stddev_frame = np.nanstd(
                self.raw_dark_current_frames,
                axis=0,
            )

        # Summing the hot pixel maps...
        if self.raw_hot_pixel_maps is None or len(self.raw_hot_pixel_maps) == 0:
            # There is no raw hot pixel maps to use to compute.
            self.hot_pixel_map = np.full(self.detector_shape, False, dtype=bool)
        else:
            # Computing the average bias and its deviation.
            self.hot_pixel_map = np.any(self.raw_hot_pixel_maps, axis=0)

        # Summing the dead pixel maps...
        if (
            self.raw_dead_pixel_maps is None
            or len(self.raw_dead_pixel_maps) == 0
        ):
            # There is no raw hot pixel maps to use to compute.
            self.dead_pixel_map = np.full(
                self.detector_shape,
                False,
                dtype=bool,
            )
        else:
            # Computing the average bias and its deviation.
            self.dead_pixel_map = np.any(self.raw_dead_pixel_maps, axis=0)

        # All done.

    def simulate_bias_frame(self: hint.Self) -> hint.NDArray:
        """Generate a new simulated bias frame.

        Parameters
        ----------
        None

        Returns
        -------
        bias_frame : NDArray
            The simulated bias frame.

        """
        # We attempt to compute a bias frame.
        bias_frame = scipy.stats.norm.rvs(
            size=self.detector_shape,
            loc=self.bias_frame,
            scale=self.bias_stddev_frame,
            random_state=self.get_random_state(),
        )
        return bias_frame

    def simulate_flat_frame(self: hint.Self) -> hint.NDArray:
        """Generate a new simulated flat frame.

        Parameters
        ----------
        None

        Returns
        -------
        flat_frame : NDArray
            The simulated flat frame.

        """
        # We attempt to compute a flat frame.
        flat_frame = scipy.stats.norm.rvs(
            size=self.detector_shape,
            loc=self.flat_frame,
            scale=self.flat_stddev_frame,
            random_state=self.get_random_state(),
        )
        return flat_frame

    def simulate_dark_current_frame(self: hint.Self) -> hint.NDArray:
        """Generate a new simulated dark current frame.

        Parameters
        ----------
        None

        Returns
        -------
        dark_current_frame : NDArray
            The simulated dark current frame.

        """
        # We attempt to compute a dark current frame.
        # The detector detector dark current, either as a value or a map.
        # The dark current itself may vary from its actual value. We model
        # such variations using a Gaussian. These values may also be
        # calculated from actual darks.
        dark_current_frame = scipy.stats.norm.rvs(
            size=self.detector_shape,
            loc=self.dark_current_frame,
            scale=self.dark_current_stddev_frame,
            random_state=self.get_random_state(),
        )
        return dark_current_frame

    def simulate_read_noise_frame(self: hint.Self) -> hint.NDArray:
        """Generate a new simulated read noise frame.

        Parameters
        ----------
        None

        Returns
        -------
        read_noise_frame : NDArray
            The simulated read noise frame.

        """
        # We attempt to compute a the read noise frame.
        full_read_noise_frame = scipy.stats.norm.rvs(
            size=self.detector_shape,
            loc=self.read_noise,
            scale=self.read_noise_stddev,
            random_state=self.get_random_state(),
        )

        # Applying the NDR factor.
        read_noise_frame = full_read_noise_frame / np.sqrt(
            self.nondestructive_reads,
        )
        return read_noise_frame

    def simulate_cosmic_ray_map(
        self: hint.Self,
        exposure_time: float,
    ) -> hint.NDArray:
        """Generate a new simulated cosmic ray map, detailing where rays hit.

        Note, cosmic rays often hit multiple pixels at once. We currently
        do not simulate this. The cosmic ray map is created by
        this function, while the :py:meth:`simulate_cosmic_ray_frame`
        function applies the cosmic ray values of this map and returns the
        additive component of cosmic rays to the detector frame data.

        Parameters
        ----------
        exposure_time : float
            The exposure time, in seconds. This is pertinent as cosmic rays
            are a rate and if one hit or not depends on the integration time.

        Returns
        -------
        cosmic_ray_map : NDArray
            The simulated cosmic ray map.

        """
        # The area we are looking over, statistically, is a pixel. We handle
        # each one differently.
        pixel_area = self.pixel_size**2

        # Cosmic rays can be modeled by Poisson statistics, so we determine
        # which pixels got hit within the exposure time.
        expected_cray_count = self.cosmic_ray_rate * exposure_time * pixel_area
        cosmic_ray_hit_map = scipy.stats.poisson.rvs(
            mu=expected_cray_count,
            loc=0,
            size=self.detector_shape,
            random_state=self.get_random_state(),
        )
        cosmic_ray_map = np.array(cosmic_ray_hit_map, dtype=bool)
        return cosmic_ray_map

    def simulate_cosmic_ray_frame(
        self: hint.Self,
        exposure_time: float,
    ) -> hint.NDArray:
        """Generate a new simulated cosmic ray frame, detailing where rays hit.

        Note, cosmic rays often hit multiple pixels at once. We currently
        do not simulate this. The cosmic ray map is created by
        :py:meth:`simulate_cosmic_ray_map`, while this function applies the
        cosmic ray values to this map and returns the additive component of
        cosmic rays to the detector frame data.

        Parameters
        ----------
        exposure_time : float
            The exposure time, in seconds. This is pertinent as cosmic rays
            are a rate and if one hit or not depends on the integration time.

        Returns
        -------
        cosmic_ray_frame : NDArray
            The simulated cosmic ray frame based on the map.

        """
        # We first need to get the cosmic ray map.
        cosmic_ray_map = self.simulate_cosmic_ray_map(
            exposure_time=exposure_time,
        )

        # Applying the cosmic ray values...
        cosmic_ray_frame = np.zeros_like(cosmic_ray_map)
        cosmic_ray_frame[cosmic_ray_map] = self.cosmic_ray_value
        # All done.
        return cosmic_ray_frame

    def linearity_function(
        self: hint.Self,
        counts: hint.NDArray,
    ) -> hint.NDArray:
        """The linearity function of the detector.

        The linearity function of the detector is used to determine actual
        ADUs from an expected ADU input, given the lack of linearity at very
        low and very high values. If no raw linearity function has been
        provided, :py:method:`add_linearity_function`, then we default to
        a perfect linearity function.

        Parameters
        ----------
        counts : NDArray
            The input counts which we will use to determine the real output
            counts on the detector.

        Returns
        -------
        output : NDArray
            The output counts after the detector linearity has been taken
            into account.

        """
        # If there is no added raw linearity function, we are forced to assume
        # a completely perfect function.
        if self.raw_linearity_function is None:
            # Assuming a perfect function.
            def perfect_linearity(input_: hint.NDArray) -> hint.NDArray:
                """A perfect linearity function; same input as output."""
                return np.asarray(input_)

            using_linearity_function = perfect_linearity
        else:
            using_linearity_function = self.raw_linearity_function

        # Applying the linearity function.
        output = using_linearity_function(counts)
        return output

    def efficiency_function(
        self: hint.Self,
        wavelength: hint.NDArray,
    ) -> hint.NDArray:
        """The efficiency function of the detector.

        The efficiency function of the detector. This is the function which
        determines the efficiency of the detector at a provided wavelength.
        The quantum efficiency function should be contained within this
        function along with any other wavelength-dependent efficiency
        considerations. If no raw efficiency function has been
        provided, :py:method:`add_efficiency_function`, then we default to
        a perfect efficiency function.

        Parameters
        ----------
        wavelength : NDArray
            The wavelength to determine the efficiency of the detector at, as
            a function of wavelength. The units of the wavelength should be
            in meters.

        Returns
        -------
        efficiency : NDArray
            The output efficiency per the detector.

        """
        # If there is no added raw efficiency function, we are forced to assume
        # a completely perfect function.
        if self.raw_efficiency_function is None:
            # Assuming a perfect function.
            def perfect_efficiency(wave: hint.NDArray) -> hint.NDArray:
                """A perfect efficiency function; 100%."""
                return np.ones_like(wave)

            using_efficiency_function = perfect_efficiency
        else:
            using_efficiency_function = self.raw_efficiency_function

        # Applying the linearity function.
        efficiency = using_efficiency_function(wavelength)
        return efficiency

    def efficiency_spectrum(
        self: hint.Self,
        wavelength: hint.NDArray,
        **kwargs: hint.Any,
    ) -> hint.LezargusSpectrum:
        """The efficiency function of the detector, except as a spectrum.

        As a convenience function, we package the efficiency data as a
        full LezargusSpectrum.

        Parameters
        ----------
        wavelength : NDArray
            The wavelength to determine the efficiency of the detector at, as
            a function of wavelength. The units of the wavelength should be
            in meters.
        **kwargs : Any
            Any additional keywords are passed to LezargusSpectrum. Note,
            the uncertainty, units, and masks are already handled and
            duplicate entries will raise an error.

        Returns
        -------
        efficiency_spectrum : LezargusSpectrum
            The efficiency data, except as a spectrum.

        """
        # We take the data, assuming next to no uncertainty.
        efficiency_wavelength = wavelength
        efficiency_data = self.efficiency_function(wavelength=wavelength)
        efficiency_uncertainty = np.zeros_like(efficiency_wavelength)

        # As assume no mask either.
        efficiency_mask = np.zeros_like(efficiency_wavelength, dtype=bool)

        # Packaging it as a spectrum.
        efficiency_spectrum = lezargus.library.container.LezargusSpectrum(
            wavelength=efficiency_wavelength,
            data=efficiency_data,
            uncertainty=efficiency_uncertainty,
            wavelength_unit="m",
            data_unit="",
            mask=efficiency_mask,
            **kwargs,
        )

        # All done.
        return efficiency_spectrum
