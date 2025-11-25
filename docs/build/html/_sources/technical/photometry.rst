.. _technical-photometry:

==========
Photometry
==========

Measuring photometry in Lezargus is done using first principles and the 
following assumptions in this section. Most, if not all, of the filters we 
use are Vega-based filters, and so the treatment here assumes Vega-based 
filters. AB-magnitudes, where relevant, have different treatments which we 
also add briefly here.

See [[NOTE]]


Energy Counting and Photon Counting Convention
==============================================

Photometric filter response curves found in the literature is often given 
as either energy integrating or photon counting conventions. More specifically,
for photon counting detectors, like CCDs, there is an extra wavelength term
to convert to the energy of a photon. As expected, they can be converted 
between each other:

.. math :: 

    \int F(\lambda) T_\gamma \mathrm{d}\lambda = \int F(\lambda) T_E \left(\frac{hc}{\lambda}\right)^{-1} \mathrm{d}\lambda
    T_\gamma \simeq T_E \lambda

More details can be found in 
`Casagrande et. al. 2014 <https://doi.org/10.1093/mnras/stu1476>`_. Overall, 
we follow their conventions here. We assume and keep the photometric filter 
response functions in their energy integrating form :math:`T_E` and leave the 
wavelength component :math:`\lambda` explicit. In the literature, some of the 
photometric response functions use the photon-counting formalism 
:math:`T_\gamma`; Lezargus converts them to the energy-integrating formalism 
to align with the procedures documented here. The photometry operations here
are all done to wavelength-based spectra :math:`F(\lambda)`, with assumed 
units.


Photometric Standard Star
=========================

Photometry is a relative system, measurements of brightness are done relative 
to a photometric standard star. There are a number to choose from
`Landolt 2009 <https://doi.org/10.1088/0004-6256/137/5/4186>`_, but, we pick 
Vega as our defining photometric standard star. We use the CALSPEC Vega 
spectrum 
(`alpha_lyr_stis_011.fits <https://archive.stsci.edu/hlsps/reference-atlases/cdbs/current_calspec/alpha_lyr_stis_011.fits>`_);
though should updated versions of said spectrum be available, we suggest 
updating to said versions. We cut the CALSPEC Vega spectrum to the wavelength 
limits that matter for Lezargus.

The relevant photometry magnitudes for Vega is as follows:

Johnson U   0.03   +/-   ???    2002yCat.2237....0D
Johnson B   0.03   +/-   ???    2002yCat.2237....0D
Johnson V   0.03   +/-   ???    2002yCat.2237....0D

Tycho2 BT   
Tycho2 VT   0.087  +/-   ???    2000A&A...355L..27H

GAIA G      0.029  +/-   ???    2018MNRAS.479L.102C
GAIA BP     0.039  +/-   ???    2018MNRAS.479L.102C
GAIA RP     0.023  +/-   ???    2018MNRAS.479L.102C

2MASS J    -0.177  +/-  0.206    2003yCat.2246....0C
2MASS H    -0.029  +/-  0.146    2003yCat.2246....0C
2MASS Ks    0.129  +/-  0.186    2003yCat.2246....0C

WISE 1      1.452  +/-           2012wise.rept....1C
WISE 2      1.143  +/-  0.019    2012wise.rept....1C
WISE 3     -0.067  +/-  0.008    2012wise.rept....1C
WISE 4     -0.127  +/-  0.006    2012wise.rept....1C


Synthetic Magnitudes
====================

The definition of magnitudes was set by convention from 
`Pogson 1856 <https://doi.org/10.1093%2Fmnras%2F17.1.12>`. Namely for a given 
filter profile :math:`T_E`, the magnitude :math:`m` of a star 
:math:`F(\lambda)` (relative to a photometric standard star 
:math:`F_0(\lambda)` of magnitude :math:`m_0`):

.. math ::

    m - m_0 = -2.5 \log \left(\frac{\int F(\lambda) T_E\mathrm{d}\lambda}{\int F_0(\lambda) T_E\mathrm{d}\lambda}\right)


Fundamentally, as synthetic magnitudes are a weighted average of flux 
`Casagrande et. al. 2014 <https://doi.org/10.1093/mnras/stu1476>`_, it is 
appropriate to normalize over :math:`T_E`. Moreover, in the optical and 
infrared most detectors are photon-counting in nature (i.e. optical CCDs 
and infrared detectors `Beletic et. al. 2008 <https://doi.org/10.1117/12.790382>`_), 
we will adapt our form accordingly. (Conveniently, normalizing removes the 
:math:`hc` constant.) The apparent magnitude relation can be rewritten as 
follows.

.. math ::

    m - m_0 = -2.5 \log \frac{\int F(\lambda) T_E \lambda \mathrm{d}\lambda}{\int T_E \lambda \mathrm{d}\lambda} + 2.5 \log \frac{\int F_0(\lambda) T_E \lambda \mathrm{d}\lambda}{\int T_E \lambda \mathrm{d}\lambda} 

Note, the factor of :math:`2.5` here is technically :math:`10^\frac{2}{5}`, 
per `Pogson 1856 <https://doi.org/10.1093%2Fmnras%2F17.1.12>`, but most 
literature rounds said factor. The units must all be self-consistent else 
taking the logarithm with a unit is impossible 
`Matta et. al. 2010 <https://doi.org/10.1021/ed1000476>`. Otherwise, the 
full equation below must be used, combining the logarithms, so that the 
result is unitless.

.. math ::

    m - m_0 = -2.5 \log \left[ \frac{\int F(\lambda) T_E \lambda \mathrm{d}\lambda}{\int T_E \lambda \mathrm{d}\lambda} \left(\frac{\int F_0(\lambda) T_E \lambda \mathrm{d}\lambda}{\int T_E \lambda \mathrm{d}\lambda} \right)^{-1} \right]


Zero Point Calculation
======================

An alternative, but more useful and common, formulation of synthetic magnitudes
start by defining the zero point :math:`Z`. (Other literature may have it 
noted as :math:`ZP`, but we use single letters here for clarity.) The zero 
point is defined solely by the filter profile :math:`T_E` and the 
photometric standard star :math:`F_0(\lambda)` and :math:`m_0`.

.. math ::

    Z \triangleq m_0 + 2.5 \log \frac{\int F_0(\lambda) T_E \lambda \mathrm{d}\lambda}{\int T_E \lambda \mathrm{d}\lambda} = m_0 + 2.5 \log \bar{F_0}

Note that :math:`\bar{F_0}` is called the absolute calibration of the 
photometric system and is sometimes also called the zero point in some 
tables. Again, keeping in mind equivalent units. The zero point values 
published have units, and the unit system provided must be used for future 
calculations using that zero point. Most pertinent, is the determination of
synthetic magnitudes.


Synthetic magnitudes using the zero point is calculated as follows, assuming
standard units.

.. math ::

    m = -2.5 \log \left( \frac{\int F(\lambda) T_E \lambda \mathrm{d}\lambda}{\int T_E \lambda \mathrm{d}\lambda} \right) + Z + \epsilon

Where :math:`\epsilon` is a "fudge" factor sometimes added as a corrective 
term to replicate the definition of the photometric system more accurately. 
This may be done in cases where the photometric standard star given is not of 
sufficient quality.
