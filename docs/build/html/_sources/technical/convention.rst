.. _technical-conventions:

===========
Conventions
===========


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

This is beneficial in that we do not need to deal with unit clashes. We 
also allow Astropy to handle quite a few unit conversions behind the scene.
However, as developing Lezargus, care must still be taken into account. The 
ultimate idea being that units within Lezargus are all self-consistent, with 
conversions happening at input and output. Unit conversions are easily done 
using Astropy using the list above as common equivalencies.

Of course, we will provide all of the needed conversions without the need for 
heavy user interactions. Usually these conversions are input via a GUI or 
configurations. FITS header information also is considered to be 
post-conversion and so is more easily accessible to other users.

