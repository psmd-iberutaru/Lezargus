.. _technical-conventions:

===========
Conventions
===========


Array Indexing
==============

[[TODO INTRO]]

We use the simple following conventions with our array indexes. 

- We default to using C-order array indexing. This is the Numpy default, but 
this default is opposite of Fortran and IDL indexing. 
- For multidimensional images and cubes, we use the following indexing order:
images, (x, y); cubes, (x, y, λ). Here, "x" is width and "y" is height. Special conversions may be needed as some image plotting conventions may be different.



Units
=====

[[TODO INTRO]]

We therefore have the following conventions established to deal with the many
differing unit conventions. 

For internal computations and exchanges, we only use pure SI base units. All 
quantities are in SI units, unless explicitly described otherwise. Some common 
quantities are described below. Please note that often Astropy
will condense these units automatically so combinations of these units may 
look weird. 

- Time: ``s``
- Length: ``m``
- Wavelength: ``m``
- Pressure: ``Pa``
- Flux density per unit wavelength: ``W m^-2 m^-1``
- Photon: ``ph``
- Pixel: ``pix``
- [[spaxel]]: ``voxel``
- Angles: ``rad``
- Solid angle: ``sr``
- Information: ``bit``

We however have the following exceptions. This is a complete list, any 
deviation from SI units not in the exception list is considered a bug.

- Atmospheric precipitable water vapor: ``mm``.


Using SI-only is beneficial in that we do not need to deal with unit clashes. 
We also allow Astropy to handle quite a few unit conversions behind the scene.
However, as developing Lezargus, care must still be taken into account. The 
ultimate idea being that units within Lezargus are all self-consistent, with 
conversions happening at input and output. Unit conversions are easily done 
using Astropy using the list above as common equivalencies.

Of course, we will provide all of the needed conversions without the need for 
heavy user interactions. Usually these conversions are input via a GUI or 
configurations. FITS header information also is considered to be 
post-conversion and so is more easily accessible to other users.

