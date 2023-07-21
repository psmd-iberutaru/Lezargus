"""This file keeps track of all of the functions and computations which deal
with the atmosphere."""

import numpy as np

from lezargus import library
from lezargus.library import logging
from lezargus.library import hint

def index_of_refraction_ideal_air(wavelength:hint.Array) -> hint.Array:
    """Calculate the ideal refraction of air over wavelength.
    
    The index of refraction of air depends slightly on wavelength, we use 
    the updated Edlen equations found in DOI: 10.1088/0026-1394/30/3/004.

    Parameters
    ----------
    wavelength : Array
        The wavelength that we are calculating the index of refraction over.
        This must in microns.

    Returns
    -------
    ior_ideal_air : Array
        The ideal air index of refraction.
    """
    # The wave number is actually used more in these equations.
    wavenumber = 1 / wavelength
    # Calculating the index of refraction, left hand then right hand side of 
    # the equation.
    ior_ideal_air = (8342.54 + 2406147 / (130 - wavenumber) + 15998 / (38.9 - wavenumber))
    ior_ideal_air = ior_ideal_air/1e8 + 1
    return ior_ideal_air

def index_of_refraction_dry_air(wavelength:hint.Array, pressure:float, temperature:float) -> hint.Array:
    """Calculate the refraction of air of pressured warm dry air.
    
    The index of refraction depends on wavelength, pressure and temperature, we 
    use the updated Edlen equations found in DOI: 10.1088/0026-1394/30/3/004.
    
    Parameters
    ----------
    wavelength : Array
        The wavelength that we are calculating the index of refraction over.
        This must in microns.
    pressure : float
        The pressure of the atmosphere, in Pascals.
    temperature : float
        The temperature of the atmosphere, in Kelvin.

    Returns
    -------
    ior_dry_air : Array
        The dry air index of refraction.
    """
    # The wave number is actually used more in these equations.
    wavenumber = 1 / wavelength
    # We need the ideal air case first.
    ior_ideal_air = index_of_refraction_ideal_air(wavelength=wavelength)

    # The Edlen equations use Celsius as the temperature unit, we need to 
    # convert from the standard Kelvin.
    temperature = temperature - 273.15

    # Calculating the pressure and temperature term.
    pt_factor = (pressure / 96095.43) * ((1 + pressure*(0.601-0.009723*temperature))/(1 + 0.003661 * temperature)) * 1e-8

    # Calculating the index of refraction of dry air.
    ior_dry_air = (ior_ideal_air - 1) * pt_factor
    ior_dry_air = ior_dry_air + 1
    return ior_dry_air

def index_of_refraction_moist_air(wavelength:hint.Array, pressure:float, temperature:float, water_pressure:float) -> hint.Array:
    """Calculate the refraction of air of pressured warm moist air.
    
    The index of refraction depends on wavelength, pressure, temperature, and 
    humidity, we use the updated Edlen equations found in 
    DOI: 10.1088/0026-1394/30/3/004. We use the partial pressure of water in 
    the atmosphere as opposed to actual humidity.
    
    Parameters
    ----------
    wavelength : Array
        The wavelength that we are calculating the index of refraction over.
        This must in microns.
    pressure : float
        The pressure of the atmosphere, in Pascals.
    temperature : float
        The temperature of the atmosphere, in Kelvin.
    water_pressure : float
        The partial pressure of water in the atmosphere, Pascals.

    Returns
    -------
    ior_moist_air : Array
        The moist air index of refraction.
    """
    # The wave number is actually used more in these equations.
    wavenumber = 1 / wavelength
    # We need the dry air case first.
    ior_dry_air = index_of_refraction_dry_air(wavelength=wavelength, pressure=pressure, temperature=temperature)
    
    # Calculating the water vapor factor.
    wv_factor = -water_pressure * (3.7345 - 0.0401 * wavenumber) * 1e-10
    
    # Computing the moist air index of refraction.
    ior_moist_air = ior_dry_air + wv_factor
    return ior_moist_air

def absolute_atmospheric_refraction_function(wavelength:hint.Array, zenith_angle:float, pressure:float, temperature:float, water_pressure:float) -> hint.Callable[[hint.Array], hint.Array]:
    """Compute the absolute atmospheric refraction function. 
    
    The absolute atmospheric refraction is not as useful as the relative 
    atmospheric refraction function. To calculate how the atmosphere refracts
    one's object, use that function instead.
    
    Parameters
    ----------
    wavelength : Array
        The wavelength over which the absolute atmospheric refraction is 
        being computed over, in microns.
    zenith_angle : float
        The zenith angle of the sight line, in radians.
    pressure : float
        The pressure of the atmosphere, in Pascals.
    temperature : float
        The temperature of the atmosphere, in Kelvin.
    water_pressure : float
        The partial pressure of water in the atmosphere, Pascals.

    Returns
    -------
    abs_atm_refr_func : Callable
        The absolute atmospheric refraction function, as an actual callable 
        function.
    """
    # We need to determine the index of refraction for moist air.
    ior_moist_air = index_of_refraction_moist_air(wavelength=wavelength, pressure=pressure, temperature=temperature, water_pressure=water_pressure)

    # The constant of refraction.
    const_of_refr = (ior_moist_air**2 - 1) / (2*ior_moist_air**2)
    # Incorporating the zenith angle.
    abs_atm_refr = const_of_refr * np.tan(zenith_angle)

    # Creating the function itself.
    abs_atm_refr_func = library.wrapper.cubic_interpolate_1d_function(x=wavelength, y=abs_atm_refr)
    return abs_atm_refr_func

def relative_atmospheric_refraction_function(wavelength:hint.Array, reference_wavelength:float, zenith_angle:float, pressure:float, temperature:float, water_pressure:float) -> hint.Callable[[hint.Array], hint.Array]:
    """Compute the relative atmospheric refraction function. 
    
    The relative refraction function is the same as the absolute refraction
    function, however, it is all relative to some specific wavelength.
    
    Parameters
    ----------
    wavelength : Array
        The wavelength over which the absolute atmospheric refraction is 
        being computed over, in microns.
    reference_wavelength : float
        The reference wavelength which the relative refraction is computed 
        against, in microns.
    zenith_angle : float
        The zenith angle of the sight line, in radians.
    pressure : float
        The pressure of the atmosphere, in Pascals.
    temperature : float
        The temperature of the atmosphere, in Kelvin.
    water_pressure : float
        The partial pressure of water in the atmosphere, Pascals.

    Returns
    -------
    rel_atm_refr_func : Callable
        The absolute atmospheric refraction function, as an actual callable 
        function.
    """
    # We need the absolute refraction function first.
    abs_atm_refr_func = absolute_atmospheric_refraction_function(wavelength=wavelength, zenith_angle=zenith_angle, pressure=pressure, temperature=temperature, water_pressure=water_pressure)

    # The refraction at the reference wavelength.
    ref_abs_refr = abs_atm_refr_func(reference_wavelength)
    def rel_atm_refr_func(wave:hint.Array) -> hint.Array:
        """The relative refraction function.
        
        Parameters
        ----------
        wave : Array
            The input wavelength for computation.
            
        Returns
        -------
        rel_atm_refr : Array
            The relative atmospheric refraction.
        """
        rel_atm_refr = abs_atm_refr_func(wave) - ref_abs_refr
        return rel_atm_refr
    # All done.
    return rel_atm_refr_func
