"""Interpolation routines, across both multi-dimensional and multi-mode.

We have many interpolation functions for a wide variety of use cases. We store
all of them here. We usually derive the more specialty interpolation functions
from a set of base functions.
"""

# isort: split
# Import required to remove circular dependencies from type checking.
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lezargus.library import hint
# isort: split

import numpy as np
import scipy.interpolate

import lezargus
from lezargus.library import logging


def get_smallest_gap(wavelength: hint.NDArray) -> float:
    """Find the smallest possible gap value for a wavelength array.

    Gaps, which are important in gap-based interpolation, are where there is
    no data. Gaps are primarily a wavelength criterion: should data be missing
    for enough of a wavelength range, it is determined to be a gap. This
    function determines the smallest possible gap in the provided wavelength
    array for which a data-only gap may exist.

    Basically, we find the maximum spacing in the wavelength array and assume
    that is it perfect and determine a gap from it.

    Parameters
    ----------
    wavelength : ndarray
        The wavelength array which is used to find the small gap.

    Returns
    -------
    small_gap : float
        The wavelength spacing for the small gap, in the same units as the
        provided wavelength array.

    """
    # We need to make sure the wavelength is in order.
    sorted_wavelength = np.sort(np.asarray(wavelength))

    # We just find the largest separation.
    small_gap_guess = np.nanmax(sorted_wavelength[1:] - sorted_wavelength[:-1])
    # However, we pad it just by some epsilon to ensure that the derived
    # separation itself is not considered a gap.
    epsilon = np.nanmax(np.spacing(sorted_wavelength))
    small_gap = small_gap_guess + epsilon
    # All done.
    return small_gap


class Generic1DInterpolate:
    """Internal class for 1D interpolators, the exact method to be determined.

    This class is mostly a wrapper class around other implementations of
    interpolators to provide a unified interface for special handling of the
    different styles of interpolations and some edge cases.

    To build a more specific interpolation class, create a subclass of this
    method and override the :py:meth:`_interpolator_generator` function with
    the detail implementation to integrate the wrapped interpolator.

    Attributes
    ----------
    x : ndarray
        The input which we will be using to interpolate from. This is not
        always the same as the input parameters due to data sanitization.
    v : ndarray
        The output which we will be using to interpolate from. This is not
        always the same as the input parameters due to data sanitization.
    raw_interpolator : function
        The interpolator generated which this class wraps around.
    extrapolate : bool
        If True, we extrapolate outside the bounds of the domain. Else we fill
        with :py:attr:`extrapolate_fill`
    extrapolate_fill : float
        The value used to fill out of bounds interpolations if
        :py:attr:`extrapolate` is False.
    gap : float
        The minimum gap spacing in the domain values for the domain to be
        considered a gap. Interpolated values in a gap filled with
        :py:attr:`gap_fill`.
    gap_fill : float
        The value used to fill interpolations in gaps.
    gap_bounds : tuple[tuple, tuple]
        The boundaries of the gaps, an internal cached value used to
        determine if inputted values are within gaps or not. One minimum and
        one maximum parallel tuples are stored.

    """

    def __init__(
        self: Generic1DInterpolate,
        x: hint.NDArray,
        v: hint.NDArray,
        extrapolate: bool = False,
        extrapolate_fill: float = np.nan,
        gap: float = +np.inf,
        gap_fill: float = np.nan,
    ) -> None:
        """Create the interpolator.

        Parameters
        ----------
        x : ndarray
            The input for interpolation.
        v : ndarray
            The output for interpolation.
        extrapolate : bool, default = False
            If True, we extrapolate, else we use the fill value.
        extrapolate_fill : float, default = np.nan
            The fill value for interpolations outside of the domain without
            extrapolation.
        gap : float, default = 0
            The minimum spacing between input points for it to be a gap.
            We default to +inf, so no gaps.
        gap_fill : float, default = np.nan
            The fill value to fill in for interpolations inside a gap region.

        Returns
        -------
        None

        """
        # We first sanitize the raw data so that the used interpolations
        # do not really complain. We also sort the arrays, interpolation on
        # unsorted arrays don't make any sense.
        clean_x, clean_v = lezargus.library.sanitize.clean_finite_arrays(x, v)
        sort_index = np.argsort(clean_x)
        self.x = clean_x[sort_index]
        self.v = clean_v[sort_index]

        # We generate the interpolator itself.
        self.raw_interpolator = self._interpolator_generator(x=self.x, v=self.v)

        # We then store all of the boundary condition flags and values.
        self.extrapolate = extrapolate
        self.extrapolate_fill = extrapolate_fill
        self.gap = gap
        self.gap_fill = gap_fill
        self.gap_bounds = self._calculate_gap_bounds()
        # All done.

    @staticmethod
    def _interpolator_generator(
        x: hint.NDArray,
        v: hint.NDArray,
    ) -> hint.Callable[[hint.NDArray], hint.NDArray]:
        """Define the integration with the wrapped interpolator here.

        This function needs to be overwritten with the implementation of the
        wrapped interpolator before using a class derived from this class.

        Parameters
        ----------
        x : ndarray
            The input data which will be fed to the interpolator.
        v : ndarray
            The output data which will be fed to the interpolator.

        Returns
        -------
        interpolator : Callable
            The interpolator function. It should accept the input provided by
            py:meth:`interpolate` as a parameter.

        """
        # This function should not be called without being overwritten by the
        # interpolation implementation subclasses of this class.
        lezargus.library.wrapper.do_nothing(x=x, v=v)
        logging.critical(
            critical_type=logging.DevelopmentError,
            message=(
                "Generic1DInterpolate interpolation generator function needs to"
                " be overwritten with the interpolation implementation."
            ),
        )

        # We define a dummy function here so that PyLint is happy. Of course,
        # the critical error process above should really prevent this code from
        # running in the first place.
        def _dummy_interpolator(input_: hint.NDArray) -> hint.NDArray:
            return input_

        return _dummy_interpolator

    def _calculate_gap_bounds(
        self: hint.Self,
    ) -> tuple[hint.NDArray, hint.NDArray]:
        """Calculate the gap lower and upper bounds.

        Parameters
        ----------
        None

        Returns
        -------
        lower_bounds : tuple
            The lower bound values of the found gaps.
        upper_bounds : tuple
            The upper bound values of the found gaps.

        """
        # We next need to find where the bounds of the gap regions are,
        # measuring based on the gap delta criteria.
        x_delta = self.x[1:] - self.x[:-1]
        is_gap = x_delta > self.gap
        # And the bounds of each of the gaps.
        lower_bounds = self.x[:-1][is_gap]
        upper_bounds = self.x[1:][is_gap]
        # All done.
        return lower_bounds, upper_bounds

    def interpolate(self: hint.Self, x: hint.NDArray) -> hint.NDArray:
        """Interpolate the input value.

        Parameters
        ----------
        x : ndarray
            The input that we are going to interpolate given the values we
            have.

        Returns
        -------
        v : ndarray
            The values after interpolation, taking into account any criteria.

        """
        # We generally do not touch the input, including sanitization.
        interp_x = np.asarray(x).copy()
        # We first calculate the interpolated data.
        interp_v = self.raw_interpolator(interp_x)
        # Then we apply all of the criteria if specified.
        if not self.extrapolate:
            # No extrapolation was desired, we apply the criteria.
            interp_v = self._apply_extrapolation_criteria(
                interp_x=interp_x,
                interp_v=interp_v,
            )
        if self.gap:
            # Gaps are present in the data and should be taken care of.
            interp_v = self._apply_gap_criteria(
                interp_x=interp_x,
                interp_v=interp_v,
            )

        # All done.
        v = interp_v
        return v

    def _apply_extrapolation_criteria(
        self: hint.Self,
        interp_x: hint.NDArray,
        interp_v: hint.NDArray,
    ) -> hint.NDArray:
        """Apply the extrapolation criteria to interpolated data.

        Namely, if there was to be no extrapolation, we replace any data with
        the extrapolation fill value.

        Parameters
        ----------
        interp_x : ndarray
            The interpolated input values.
        interp_v : ndarray
            The interpolated output values which we will apply the criteria
            too.

        Returns
        -------
        output : ndarray
            A copy of the interpolated output after the criteria has been
            applied.

        """
        # For the values which are outside, we fill them in if we are not
        # supposed to extrapolate.
        if self.extrapolate:
            # We were supposed to extrapolate, no change.
            return interp_v

        # By definition, any values which would have been extrapolated falls
        # outside of the original domain.
        x_min = np.nanmin(self.x)
        x_max = np.nanmax(self.x)
        is_outside = ~((x_min <= interp_x) & (interp_x <= x_max))

        # We make a small copy of the output data that we will need to
        # modify. We adapt to the type of the extrapolation fill value.
        output_type = np.result_type(interp_v, self.extrapolate_fill)
        output = np.asarray(interp_v, dtype=output_type)

        # Filling them in.
        output[is_outside] = self.extrapolate_fill
        # All done.
        return output

    def _apply_gap_criteria(
        self: hint.Self,
        interp_x: hint.NDArray,
        interp_v: hint.NDArray,
    ) -> hint.NDArray:
        """Apply the gap criteria to interpolated data.

        Namely, if the interpolated value falls within a gap, we replace the
        value with the gap fill value.

        Parameters
        ----------
        interp_x : ndarray
            The interpolated input values.
        interp_v : ndarray
            The interpolated output values which we will apply the criteria
            too.

        Returns
        -------
        output : ndarray
            A copy of the interpolated output after the criteria has been
            applied.

        """
        # We first check if we were to even find and exclude gaps in the first
        # place.
        if not self.gap:
            # No, the gap flag is false, no gaps. So, no change.
            return interp_v

        # We make a small copy of the output data that we will need to
        # modify. We use floats as NaNs don't work with any other type.
        output_type = np.result_type(interp_v, np.nan)
        output = np.asarray(interp_v, dtype=output_type)

        # We already computed where the gaps are. All that is left is
        # checking if they are within them.
        lower_gap, upper_gap = self.gap_bounds
        for lowerdex, upperdex in zip(lower_gap, upper_gap, strict=True):
            # We NaN out points based on the input. We do not want to NaN
            # the actual bounds themselves however.
            output[(lowerdex < interp_x) & (interp_x < upperdex)] = np.nan

        # All done.
        return output

    @classmethod
    def template_class(
        cls: type[hint.Self],
        **kwargs: hint.Any,
    ) -> hint.Callable[[hint.NDArray, hint.NDArray], hint.Self]:
        """Provide a template with the same flags as this interpolator class.

        This function does the same thing as :py:meth:`template_instance`, but
        this function operates on the class itself as opposed to the instance.

        Parameters
        ----------
        **kwargs : Any
            Any keyword arguments provided will be passed to the constructor,
            overriding any local flags for the purposes of creating the
            template function.

        Returns
        -------
        interpolator_template : Callable
            The interpolator template with all of the flags the same as this
            current instance.

        """
        # We just make a dummy instance of the interpolator and use the
        # template instance functionality type instead.
        dummy_x = np.linspace(1, 3, 10)
        dummy_v = np.pi * dummy_x
        interpolator = cls(x=dummy_x, v=dummy_v)
        interpolator_template = interpolator.template_instance(**kwargs)
        return interpolator_template

    def template_instance(
        self: hint.Self,
        **kwargs: hint.Any,
    ) -> hint.Callable[[hint.NDArray, hint.NDArray], hint.Self]:
        """Provide a template with the same flags as this interpolator.

        Sometimes it is needed to have an interpolator which you can make
        on the fly. This function makes an interpolator template. Data
        still needs to be provided to defined to make the interpolator from
        the template; but the flags are kept the same.

        Parameters
        ----------
        **kwargs : Any
            Any keyword arguments provided will be passed to the constructor,
            overriding any local flags for the purposes of creating the
            template function.

        Returns
        -------
        interpolator_template : Callable
            The interpolator template with all of the flags the same as this
            current instance.

        """
        # The current interpolator. We need to do this for subclassing
        # purposes.
        interpolator_class = type(self)
        # We need to get the main parameters.
        self_params = {
            "extrapolate": self.extrapolate,
            "extrapolate_fill": self.extrapolate_fill,
            "gap": self.gap,
            "gap_fill": self.gap_fill,
        }
        # We overwrite any of the local parameters with those provided.
        template_params = kwargs | self_params

        # Defining the interpolator template.
        def interpolator_template(
            x: hint.NDArray,
            v: hint.NDArray,
        ) -> hint.Self:
            """Create the interpolator from this template.

            Parameters
            ----------
            x : ndarray
                The input for interpolation.
            v : ndarray
                The output for interpolation.

            Returns
            -------
            interpolator : Generic1DInterpolate
                The interpolator class generated from the provided data and
                the template flags.

            """
            # Building the interpolator and filling in the flags.
            interpolator = interpolator_class(x=x, v=v, **template_params)
            # All done.
            return interpolator

        # All done.
        return interpolator_template

    # A quick alias so that we can compute the interpolation as a simple call.
    __call__ = interpolate


class Nearest1DInterpolate(Generic1DInterpolate):
    """Nearest value based interpolation class.

    A simple linear interpolator.
    """

    @staticmethod
    def _interpolator_generator(
        x: hint.NDArray,
        v: hint.NDArray,
    ) -> hint.Callable[[hint.NDArray], hint.NDArray]:
        """Linear interpolator.

        Parameters
        ----------
        x : ndarray
            The input data fed to the linear interpolator.
        v : ndarray
            The output data fed to the linear interpolator.

        Returns
        -------
        interpolator : Callable
            The linear interpolator.

        """
        interpolator = scipy.interpolate.interp1d(
            x,
            v,
            kind="nearest",
            fill_value="extrapolate",
        )
        # And we send off the interpolation function.
        return interpolator


class Linear1DInterpolate(Generic1DInterpolate):
    """Linear based interpolation class.

    A simple linear interpolator.
    """

    @staticmethod
    def _interpolator_generator(
        x: hint.NDArray,
        v: hint.NDArray,
    ) -> hint.Callable[[hint.NDArray], hint.NDArray]:
        """Linear interpolator.

        Parameters
        ----------
        x : ndarray
            The input data fed to the linear interpolator.
        v : ndarray
            The output data fed to the linear interpolator.

        Returns
        -------
        interpolator : Callable
            The linear interpolator.

        """

        # The Numpy linear interpolator doesn't return an interpolation
        # class so we need to do it ourselves. We also need to implement
        # extrapolation.
        def interpolator(input_: hint.NDArray) -> hint.NDArray:
            """Linear interpolator."""
            # The original data.
            # The output.
            output = np.zeros_like(input_)
            # First, we split the input into the three main regimes: lower
            # inner, and upper of the limits of the original data.
            lo_i = input_ < x.min()
            in_i = (x.min() <= input_) & (input_ <= x.max())
            up_i = x.max() < input_
            # We first do the internal linear interpolation,
            output[in_i] = np.interp(input_[in_i], x, v)
            # Then the extrapolations, using the y=mx+b formulations.
            output[lo_i] = v[0] + (input_[lo_i] - x[0]) * (v[1] - v[0]) / (
                x[1] - x[0]
            )
            output[up_i] = v[-2] + (input_[up_i] - x[-2]) * (v[-1] - v[-2]) / (
                x[-1] - x[-2]
            )
            # All done.
            return output

        # And we send off the interpolation function.
        return interpolator


class Spline1DInterpolate(Generic1DInterpolate):
    """Spline based interpolation class.

    We use a polynomial piece-wise spline. This is better than a pure cubic
    interpolator as the modified Akima spline method used preserves the curve
    shapes better.
    """

    @staticmethod
    def _interpolator_generator(
        x: hint.NDArray,
        v: hint.NDArray,
    ) -> hint.Callable[[hint.NDArray], hint.NDArray]:
        """Generate modified Akima interpolator.

        Parameters
        ----------
        x : ndarray
            The input data fed to the modified Akima interpolator.
        v : ndarray
            The output data fed to the modified Akima interpolator.

        Returns
        -------
        interpolator : Callable
            The modified Akima interpolator.

        """
        # The Akima1D interpolator. The modified version is the better one to
        # use considering its advantages. Namely the higher dimensionality and
        # better handling of flat data.
        interpolator = scipy.interpolate.Akima1DInterpolator(
            x,
            v,
            method="makima",
            extrapolate=True,
        )
        return interpolator


class RepeatNDInterpolate:
    """An ND interpolator class for multi-dimensional interpolation.

    This interpolation requires a rectilinear grid like arrangement of data,
    but we do accept gaps and NaNs in the data. We perform interpolation
    by repeated 1D interpolations across the dimensions until we get the
    interpolated value. The order of the interpolations and the
    actual 1D interpolation algorithm is provided on instantiation.

    We do suggest using the repeat interpolators which actually define their
    axes; such are built for 2D :py:class:`Repeat2DInterpolate` and
    3D :py:class:`Repeat3DInterpolate` interpolators. For single dimensions,
    see :py:class:`Generic1DInterpolate` and its subclasses.
    For higher dimensions,  we suggest using this class.

    Attributes
    ----------
    domain : list
        A list of the domain axes values which define the multidimensional
        data.
    v : ndarray
        The multi-dimensional data who's axes are defined. This data is the
        data interpolated.
    interpolator_template : Callable
        The template function for the 1D interpolations.

    """

    def __init__(
        self: RepeatNDInterpolate,
        domain: list[hint.NDArray],
        v: hint.NDArray,
        template: hint.Callable,
    ) -> None:
        """Create the 2D interpolator, constructed from many 1D interpolations.

        Parameters
        ----------
        domain : list
            The list of domain axis values of the multi-dimensional data.
            Note, the repeated interpolation procedure follows the axis order
            provided.
        v : ndarray
            The data itself, the dimensions must match the provided axes.
        template : Callable
            The 1D interpolator template function which will be used to build
            the needed 1D interpolators.

        Returns
        -------
        None

        """
        # We check that the shape provided by the domain matches the data
        # shape. The domain order provided above is actually the reverse of
        # the Numpy convention.
        domain_shape = tuple(domaindex.size for domaindex in domain)
        if reversed(domain_shape) != v.shape:
            logging.error(
                error_type=logging.InputError,
                message=(
                    f"The shape of the data is {v.shape} which does not "
                    " match the expected shape from the provided domain"
                    f" {domain_shape}."
                ),
            )

        # We test the template function here with dummy data, just to make
        # sure it is a template function.
        temp_linear_data = np.linspace(0, 10, 10)
        template_return = template(x=temp_linear_data, v=temp_linear_data)
        if not isinstance(template_return, Generic1DInterpolate):
            logging.error(
                error_type=logging.InputError,
                message=(
                    f"The template return has a class {type(template_return)}"
                    " which is not expected from an interpolator template"
                    " function."
                ),
            )

        # Finally, assigning everything.
        self.domain = domain
        self.v = v
        self.interpolator_template = template

    @staticmethod
    def _interpolate_reduce_dimension(
        data: hint.NDArray,
        single_domain: hint.NDArray,
        point: float,
        template: hint.Callable,
        axis: int = -1,
    ) -> hint.NDArray:
        """Interpolate and reduce the multi-dimensional data by one dimension.

        This function interpolates a multi-dimensional data set by creating 1D
        interpolators along the dimension that we will reduce. We evaluate
        the interpolators at a single point, replacing that entire dimension
        with single values. The new interpolated data is one dimension
        reduced from the original. This is basically an "iteration" in the
        repeated 1D interpolations.

        This function generally should not be used by an end-user and should
        only be used internally.

        Parameters
        ----------
        data : ndarray
            The multi-dimensional data we will be interpolating and reducing
            in one dimension.
        single_domain : ndarray
            The input domain axis of the data for the axis that we are
            reducing.
        point : ndarray
            The single point value we are evaluating the interpolation
            instances at to reduce the dimension down.
        template : Callable
            The 1D interpolator template function which will be used to build
            the needed 1D interpolators.
        axis : int, default = -1
            The axis we are reducing down. By default, we reduce along the
            first axis in the order provided by this class.


        Returns
        -------
        reduced : ndarray
            The new interpolated multi-dimensional data after the reduction
            of the dimension from the interpolation.

        """
        # We first check that dimension that we will be reducing, to make
        # sure it is valid and proper.
        if data.shape[axis] != single_domain.size:
            logging.error(
                error_type=logging.InputError,
                message=(
                    f"The specified axis {axis} has a provided"
                    f" {single_domain} length, the actual data shape"
                    f" {data.shape} has length {data.shape[axis]}."
                ),
            )

        # We build the interpolation generating function which will be
        # mapped across the array. To save resources, we also evaluate it
        # at the same time as the mapping.
        def interpolate_evaluate(y: hint.NDArray) -> float:
            """Create an interpolator across the axis and evaluate it.

            Parameters
            ----------
            y : ndarray
                The data along the slice we are mapped to; given the input
                x defined.

            Returns
            -------
            val : ndarray
                The evaluated value at the provided point we are reducing to.

            """
            # A quick check on the mapped data and what we have for
            # interpolation.
            if single_domain.size != y.size:
                logging.critical(
                    logging.DevelopmentError,
                    message=(
                        "Interpolation reducing map function data length"
                        f" {y.size}, input axis length {single_domain.size}."
                    ),
                )
            # Otherwise, we can compute the value.
            # We skip the interpolator check, due to this function being an
            # internal function. The higher level checks should have caught it.
            val = template(single_domain, y)(point)
            return val

        # We apply and evaluate the function across the provided axis, reducing
        # it down.
        reduced = np.apply_along_axis(interpolate_evaluate, axis, data)
        return reduced

    def _interpolate(self: hint.Self, *domain: hint.NDArray) -> hint.NDArray:
        """Interpolate the data points; internal function.

        The shape and arrangement of the input points provided is preserved.
        We just assume the shape of the input axis points, interpolate point
        by point, then reshape the result based on the shape of the input.

        This function is hidden so this class can be subclassed easily. Please
        call the public interface class to interpolate:
        :py:meth:`interpolate`.

        Parameters
        ----------
        *domain : ndarray
            The domain value axes which we are interpolating at, given in
            order as the axes domain of this class.

        Returns
        -------
        v : ndarray
            The interpolated values.

        """
        # We only want to work with arrays.
        domain_array = [np.asarray(domaindex) for domaindex in domain]

        # The shape of all the the input must be the same shape. Defaulting
        # to have the first element be the primary.
        input_shape = domain_array[0].shape
        for domaindex in domain_array:
            if domaindex.shape != input_shape:
                logging.error(
                    logging.InputError,
                    message="Not all input domain axes have the same shape.",
                )

        # We work with flat arrays, evaluating the interpolation points then
        # repackage them back into the proper shape later.
        flat_domain = [np.ravel(domaindex) for domaindex in domain_array]
        flat_result = np.empty(input_shape)
        for index, pointdex in enumerate(zip(*flat_domain, strict=True)):
            flat_result[index] = self._interpolate_point(*pointdex)

        # Repackaging.
        v = np.reshape(flat_result, input_shape)
        # All done.
        return v

    def _interpolate_point(self: hint.Self, *point: float) -> float:
        """Interpolate a single point.

        This function determines the interpolated value for a given single
        point. We suggest that this method is not called directly unless
        really only a single point is needed.

        Parameters
        ----------
        *point : float
            The point that we are interpolating to. The order of the float
            values in this point should match the interpolation order of the
            axes; similar to a Cartesian grid point.

        Returns
        -------
        v : float
            The interpolated output value.

        """
        # Need to make sure there is enough data points.
        if len(point) != len(self.domain):
            logging.error(
                error_type=logging.InputError,
                message=(
                    f"Defined point has {len(point)} values, incompatible with"
                    f" {len(self.domain)} dimensions."
                ),
            )

        # We continuously reduce the dimensions, evaluating based on the
        # input point.
        reduced_data = self.v
        for domaindex, valuedex in zip(self.domain, point, strict=True):
            reduced_data = self._interpolate_reduce_dimension(
                data=reduced_data,
                single_domain=domaindex,
                point=valuedex,
                template=self.interpolator_template,
                axis=-1,
            )

        # The reduced value should be able to be cast to a single value to
        # be returned.
        v = float(reduced_data)
        return v

    def _interpolate_slice(
        self: hint.Self,
        *slice_: float | None,
    ) -> hint.NDArray:
        """Interpolate a single slice of the data.

        A "slice" is provided by specifying the values of specific points
        to interpolate the given axis at. Specifying None keeps that dimension
        part of the slice.

        Parameters
        ----------
        slice_ : float | None
            The slice specification to interpolate at. The order of the
            parameters corresponds to the axis order. Specifying None
            means the axis is part of the slice and is not interpolated.

        Returns
        -------
        v : float
            The interpolated output value.

        """
        # Need to make sure there is enough data points.
        if len(slice_) != len(self.domain):
            logging.error(
                error_type=logging.InputError,
                message=(
                    f"Point has {len(slice_)} values, incompatible with"
                    f" {len(self.domain)} dimensions."
                ),
            )

        # The procedure is very similar to interpolating a single point, but
        # we skip the dimensions that are specified as None. Note, the
        # order of the domain axes are inverse of the Numpy convention.
        axes_order = np.flip(np.arange(len(slice_)))
        reduced_data = self.v
        for axisdex, domaindex, valuedex in zip(
            axes_order,
            self.domain,
            slice_,
            strict=True,
        ):
            # If the slice value is None, we do not interpolate this slice.
            if valuedex is None:
                continue
            # Otherwise, we reduce along the slice value.
            reduced_data = self._interpolate_reduce_dimension(
                data=reduced_data,
                single_domain=domaindex,
                point=valuedex,
                template=self.interpolator_template,
                axis=axisdex,
            )

        # All done.
        v = reduced_data
        return v

    # The actual exposed interpolation functions. The heavy lifting is
    # done by the hidden functions. This makes it easier to subclass this
    # function.
    interpolate = _interpolate
    interpolate_point = _interpolate_point
    interpolate_slice = _interpolate_slice

    # A quick alias so that we can compute the interpolation as a simple call.
    __call__ = interpolate


class Repeat2DInterpolate(RepeatNDInterpolate):
    """A 2D interpolator class for multi-dimensional interpolation.

    This interpolation requires a rectilinear grid, with the structure defined.
    Like the parent class, we do interpolation by successive 1D interpolation.
    However, we wrap the parent class to make it more understandable.


    Attributes
    ----------
    x : ndarray
        The first axis of the multi-dimensional data.
    y : ndarray
        The second axis of the multi-dimensional data.
    v : ndarray
        The data itself.
    interpolator_template : Callable
        The template function for the 1D interpolations.
    _parent : RepeatNDInterpolate
        The parent instance that does all of the heavy lifting.

    """

    def __init__(
        self: Repeat2DInterpolate,
        x: hint.NDArray,
        y: hint.NDArray,
        v: hint.NDArray,
        template: hint.Callable,
    ) -> None:
        """Create the 2D interpolator, constructed from many 1D interpolations.

        Parameters
        ----------
        x : ndarray
            The first axis of the multi-dimensional data.
        y : ndarray
            The second axis of the multi-dimensional data.
        v : ndarray
            The data itself, the dimensions must match x and y.
        template : Callable
            The 1D interpolator template function which will be used to build
            the needed 1D interpolators. A template function can be constructed
            from the helper function :py:func:`generate_template`.

        """
        # We make sure that the axes provided properly match the array.
        # The domain order provided above is actually the reverse of
        # the Numpy convention.
        domain_shape = (x.size, y.size)
        if v.shape != reversed(domain_shape):
            logging.error(
                error_type=logging.InputError,
                message=(
                    f"The shape of the data is {v.shape} which does not"
                    f" match the provided axes (x, y): {domain_shape}."
                ),
            )

        # Finally, assigning everything.
        self.x = x
        self.y = y
        self.v = v
        self.template_interpolator = template

        # And creating the super class.
        _domain = [self.x, self.y]
        super().__init__(
            domain=_domain,
            v=self.v,
            template=self.template_interpolator,
        )

    def interpolate(
        self: hint.Self,
        x: hint.NDArray,
        y: hint.NDArray,
    ) -> hint.NDArray:
        """Interpolate the data points.

        We interpolate the data at the given x and y values. The shapes
        of all of the inputs must be the same, and the output shape is
        preserved as based as possible.

        Parameters
        ----------
        x : ndarray
            The x values for interpolation.
        y : ndarray
            The y values for interpolation.

        Returns
        -------
        v : ndarray
            The interpolated values.

        """
        # The interpolation process works on flattened arrays, we record the
        # shape here so we can reshape the output later.
        x = np.asarray(x)
        y = np.asarray(y)
        if x.shape != y.shape:
            logging.error(
                error_type=logging.InputError,
                message=(
                    f"Input points have different dimensions: x {x.shape} vs y"
                    f" {y.shape}."
                ),
            )
        v = self._interpolate(x, y)
        return v

    def interpolate_point(self: hint.Self, x: float, y: float) -> float:
        """Interpolate a single point.

        This function determines the interpolated value for a given single
        point. We suggest that this method is not called directly unless
        really only a single point is needed.

        Parameters
        ----------
        x : float
            The x axis value we are interpolating at.
        y : float
            The y axis value we are interpolating at.

        Returns
        -------
        v : float
            The interpolated output value.

        """
        v = self._interpolate_point(x, y)
        # All done.
        return v

    def interpolate_slice(
        self: hint.Self,
        x: float | None,
        y: float | None,
    ) -> np.ndarray:
        """Interpolate a single slice of the data.

        A "slice" is provided by specifying the values of specific points
        to interpolate the given axis at. Specifying None keeps that dimension
        part of the slice.

        Parameters
        ----------
        x : float | None
            The x axis value we are interpolating the slice at. If None, then
            the slice runs down this axis.
        y : float | None
            The y axis value we are interpolating at. If None, then
            the slice runs down this axis.

        Returns
        -------
        v : float
            The interpolated output slice.

        """
        return self._interpolate_slice(x, y)

    # A quick alias so that we can compute the interpolation as a simple call.
    __call__ = interpolate


class Repeat3DInterpolate(RepeatNDInterpolate):
    """A 3D interpolator class for multi-dimensional interpolation.

    This interpolation requires a rectilinear grid, with the structure defined.
    Like the parent class, we do interpolation by successive 1D interpolation.
    However, we wrap the parent class to make it more understandable.


    Attributes
    ----------
    x : ndarray
        The first axis of the multi-dimensional data.
    y : ndarray
        The second axis of the multi-dimensional data.
    z : ndarray
        The third axis of the multi-dimensional data.
    v : ndarray
        The data itself.
    interpolator_template : Callable
        The template function for the 1D interpolations.
    _parent : RepeatNDInterpolate
        The parent instance that does all of the heavy lifting.

    """

    def __init__(
        self: Repeat3DInterpolate,
        x: hint.NDArray,
        y: hint.NDArray,
        z: hint.NDArray,
        v: hint.NDArray,
        template: hint.Callable,
    ) -> None:
        """Create the 2D interpolator, constructed from many 1D interpolations.

        Parameters
        ----------
        x : ndarray
            The first axis of the multi-dimensional data.
        y : ndarray
            The second axis of the multi-dimensional data.
        z : ndarray
            The third axis of the multi-dimensional data.
        v : ndarray
            The data itself, the dimensions must match x and y.
        template : Callable
            The 1D interpolator template function which will be used to build
            the needed 1D interpolators. A template function can be constructed
            from the helper function :py:func:`generate_template`.

        """
        # We make sure that the axes provided properly match the array.
        # The domain order provided above is actually the reverse of
        # the Numpy convention.
        domain_shape = (x.size, y.size, z.size)
        if v.shape != reversed(domain_shape):
            logging.error(
                error_type=logging.InputError,
                message=(
                    f"The shape of the data is {v.shape} which does not "
                    " match the provided axes (x, y, z):"
                    f" {domain_shape}."
                ),
            )

        # Finally, assigning everything.
        self.x = x
        self.y = y
        self.z = z
        self.v = v
        self.template_interpolator = template

        # And creating the super class.
        _domain = [self.x, self.y, self.z]
        super().__init__(
            domain=_domain,
            v=self.v,
            template=self.template_interpolator,
        )

    def interpolate(
        self: hint.Self,
        x: hint.NDArray,
        y: hint.NDArray,
        z: hint.NDArray,
    ) -> hint.NDArray:
        """Interpolate the data points.

        We interpolate the data at the given x, y, and z values. The shapes
        of all of the inputs must be the same, and the output shape is
        preserved as based as possible.

        Parameters
        ----------
        x : ndarray
            The x values for interpolation.
        y : ndarray
            The y values for interpolation.
        z : ndarray
            The z values for interpolation.

        Returns
        -------
        v : ndarray
            The interpolated values.

        """
        # The interpolation process works on flattened arrays, we record the
        # shape here so we can reshape the output later.
        x = np.asarray(x)
        y = np.asarray(y)
        z = np.asarray(z)
        if x.shape != y.shape != z.shape:
            logging.error(
                error_type=logging.InputError,
                message=(
                    f"Input points have different dimensions: x {x.shape} vs y"
                    f" {y.shape} vs z {z.shape}."
                ),
            )
        v = self._interpolate(x, y, z)
        return v

    def interpolate_point(
        self: hint.Self,
        x: float,
        y: float,
        z: float,
    ) -> float:
        """Interpolate a single point.

        This function determines the interpolated value for a given single
        point. We suggest that this method is not called directly unless
        really only a single point is needed.

        Parameters
        ----------
        x : float
            The x axis value we are interpolating at.
        y : float
            The y axis value we are interpolating at.
        z : float
            The z axis value we are interpolating at.

        Returns
        -------
        v : float
            The interpolated output value.

        """
        v = self._interpolate_point(x, y, z)
        # All done.
        return v

    def interpolate_slice(
        self: hint.Self,
        x: float | None,
        y: float | None,
        z: float | None,
    ) -> np.ndarray:
        """Interpolate a single slice of the data.

        A "slice" is provided by specifying the values of specific points
        to interpolate the given axis at. Specifying None keeps that dimension
        part of the slice.

        Parameters
        ----------
        x : float | None
            The x axis value we are interpolating the slice at. If None, then
            the slice runs down this axis.
        y : float | None
            The y axis value we are interpolating at. If None, then
            the slice runs down this axis.
        z : float | None
            The z axis value we are interpolating at. If None, then
            the slice runs down this axis.

        Returns
        -------
        v : float
            The interpolated output slice.

        """
        return self._interpolate_slice(x, y, z)

    # A quick alias so that we can compute the interpolation as a simple call.
    __call__ = interpolate


class RegularNDInterpolate(scipy.interpolate.RegularGridInterpolator):
    """Wrapper for Scipy's regular grid interpolator.

    This interpolator is more reliable than the :py:class:`RepeatNDInterpolate`
    for cases where the regular grid is strictly preserved and no extrapolation
    is needed. This interpolation is strictly cubic interpolation.

    Note, we do not document attributes for the parent class. See
    :py:class:`scipy.interpolate.RegularGridInterpolator` for more information.

    Attributes
    ----------
    domain : list
        A list of the domain axes values which define the multidimensional
        data.
    v : ndarray
        The multi-dimensional data who's axes are defined. This data is the
        data interpolated.

    """

    def __init__(
        self: RegularNDInterpolate,
        domain: list[hint.NDArray],
        v: hint.NDArray,
    ) -> None:
        """Create the interpolator, using the Scipy interpolator as a base.

        Parameters
        ----------
        domain : list
            The list of domain axis values of the multi-dimensional data.
        v : ndarray
            The data itself, the dimensions must match the provided axes.

        Returns
        -------
        None

        """
        # We check that the shape provided by the domain matches the data
        # shape.
        domain_shape = tuple(domaindex.size for domaindex in domain)
        if domain_shape != v.shape:
            logging.error(
                error_type=logging.InputError,
                message=(
                    f"The shape of the data is {v.shape} which does not "
                    " match the expected shape from the provided domain"
                    f" {domain_shape}."
                ),
            )

        # Assigning the class attributes; we do not document any of the
        # parent class attributes.
        self.domain = domain
        self.v = v

        # We determine the method by the shape of the data. By default we use
        # cubic, but we reduce to lower order methods if there are not
        # enough points.
        slinear_minimum = 2
        cubic_minimum = 4
        if np.all(np.asarray(self.v.shape) >= cubic_minimum):
            method = "cubic"
        elif np.all(np.asarray(self.v.shape) >= slinear_minimum):
            method = "slinear"
        else:
            method = "nearest"

        # Calling the parent class for the implementation.
        super().__init__(
            points=self.domain,
            values=self.v,
            method=method,
            bounds_error=False,
            fill_value=None,
        )

    def interpolate(self: hint.Self, *domain: hint.NDArray) -> hint.NDArray:
        """Interpolate the data points provided their axis values.

        Parameters
        ----------
        *domain : ndarray
            The domain value axes which we are interpolating at, given in
            order as the axes domain of this class.

        Returns
        -------
        v : ndarray
            The interpolated values.

        """
        # We only want to work with arrays.
        domain_array = [np.asarray(domaindex) for domaindex in domain]

        # The shape of all the the input must be the same shape. Defaulting
        # to have the first element be the primary.
        input_shape = domain_array[0].shape
        for domaindex in domain_array:
            if domaindex.shape != input_shape:
                logging.error(
                    logging.InputError,
                    message="Not all input domain axes have the same shape.",
                )

        # We work with flat arrays, evaluating the interpolation points then
        # repackage them back into the proper shape later.
        flat_domain = [np.ravel(domaindex) for domaindex in domain_array]
        # The points to interpolate at. Scipy can handle input of multiple
        # points.
        points = list(zip(*flat_domain, strict=True))
        flat_result = self(points)

        # Repackaging.
        v = np.reshape(flat_result, input_shape)
        # All done.
        return v

    def interpolate_point(self: hint.Self, *point: float) -> float:
        """Interpolate a single point.

        Parameters
        ----------
        *point : float
            The point that we are interpolating to. The order of the float
            values in this point should match the interpolation order of the
            axes; similar to a Cartesian grid point.

        Returns
        -------
        v : float
            The interpolated output value.

        """
        # Need to make sure there is enough data points.
        if len(point) != len(self.domain):
            logging.error(
                error_type=logging.InputError,
                message=(
                    f"Defined point has {len(point)} values, incompatible with"
                    f" {len(self.domain)} dimensions."
                ),
            )

        # We continuously reduce the dimensions, evaluating based on the
        # input point.
        v = float(self(point))
        return v

    def interpolate_slice(
        self: hint.Self,
        *slice_: float | None,
    ) -> hint.NDArray:
        """Interpolate a single slice of the data.

        A "slice" is provided by specifying the values of specific points
        to interpolate the given axis at. Specifying None keeps that dimension
        part of the slice.

        Parameters
        ----------
        slice_ : float | None
            The slice specification to interpolate at. The order of the
            parameters corresponds to the axis order. Specifying None
            means the axis is part of the slice and is not interpolated.

        Returns
        -------
        v : float
            The interpolated output value.

        """
        # Need to make sure there is enough data points.
        if len(slice_) != len(self.domain):
            logging.error(
                error_type=logging.InputError,
                message=(
                    f"Point has {len(slice_)} values, incompatible with"
                    f" {len(self.domain)} dimensions."
                ),
            )

        logging.critical(
            critical_type=logging.ToDoError,
            message=(
                "Slice not figured out for Scipy regular grid interpolation."
            ),
        )

        # All done.
        v = np.array([1, 2])
        return v
