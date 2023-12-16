"""Stitch spectra, images, and cubes together.

Stitching spectra, images, and cubes consistently, while keeping all of the
pitfalls in check, is not trivial. We group these three stitching functions,
and the required spin-off functions, here.
"""

import numpy as np

from lezargus import library
from lezargus.library import hint
from lezargus.library import logging


def stitch_spectra_functional(
    wavelength_functions: list[hint.Callable[[hint.ndarray], hint.ndarray]],
    data_functions: list[hint.Callable[[hint.ndarray], hint.ndarray]],
    uncertainty_functions: list[hint.Callable[[hint.ndarray], hint.ndarray]],
    weight_functions: list[hint.Callable[[hint.ndarray], hint.ndarray]],
    average_routine: hint.Callable[
        [hint.ndarray, hint.ndarray, hint.ndarray],
        tuple[float, float],
    ] = None,
    reference_points : hint.ndarray = None,
) -> tuple[
    hint.Callable[[hint.ndarray], hint.ndarray],
    hint.Callable[[hint.ndarray], hint.ndarray],
    hint.Callable[[hint.ndarray], hint.ndarray],
]:
    R"""Stitch spectra functions together.
    
    We take functional forms of the wavelength, data, uncertainty, and weight
    (in the form of f(wave) = result), and determine the average spectra. 
    We assume that the all of the functional forms properly handle any bounds,
    gaps, and interpolative limits. The input lists of functions should be 
    parallel and all of them should be of the same (unit) scale.

    Parameters
    ----------
    wavelength_functions : list[Callable]
        The list of the wavelength function. The inputs to these functions 
        should be the wavelength.
    data_functions : list[Callable]
        The list of the data function. The inputs to these functions should 
        be the wavelength.
    uncertainty_functions : list[Callable]
        The list of the uncertainty function. The inputs to these functions 
        should be the wavelength.
    weight_functions : list[Callable]
        The list of the weight function. The weights are passed to the 
        averaging routine to properly weight the average.
    average_routine : Callable
        The averaging function. It must be able to support the propagation of 
        uncertainties and weights. As such, it should have the form of
        :math:`f(x, \sigma, w) = \bar{x} \pm \sigma`.
    reference_points : ndarray, default = None
        The reference points which we are going to evaluate the above functions 
        at. The values should be of the same (unit) scale as the input of the 
        above functions. If None, we default to...
        
        ..math :: \left{ x \in \mathbb{R}, n=1000000 \middle 0.30 \leq x \leq 5.50\right}
    """
