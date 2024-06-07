=======================
Uncertainty Propagation
=======================

We follow the typical fundamental equations for the propagation of uncertainty 
for simple operations. We describe the methods for each operation as follows.
The calculations below propagate uncertainty via variances, though we adapt 
the methods below for uncertainty propagation via standard deviations.

The variables with some provided uncertainty are :math:`A` and :math:`B` with
their corresponding standard deviation uncertainty :math:`\sigma_A` and 
:math:`\sigma`. We assume the covariance term for all of our operations are 
zero, so, for our purposes, :math:`\sigma_{AB} \triangleq 0`; however, for the 
sake of clarity, we leave this term in our equations below.


.. _technical-uncertainty-addition-and-subtraction:

Addition and Subtraction
========================

The typical equations for the propagation of variance uncertainty for 
addition and subtraction are:

.. math :: 

    f = A + B \qquad \sigma_f^2 \approx \sigma_A^2 + \sigma_B^2 + 2 \sigma_{AB}

    f = A - B \qquad \sigma_f^2 \approx \sigma_A^2 + \sigma_B^2 - 2 \sigma_{AB}


.. _technical-uncertainty-multiplication_and_division:

Multiplication and Division
===========================

The typical equations for the propagation of variance uncertainty for 
multiplication and division are:

.. math :: 

    f = AB \qquad \sigma_f^2 \approx f^2 \left(\left(\frac{\sigma_A}{A}\right)^2 + \left(\frac{\sigma_B}{B}\right)^2 + 2\frac{\sigma_{AB}}{AB} \right)

    f = \frac{A}{B} \qquad \sigma_f^2 \approx f^2 \left(\left(\frac{\sigma_A}{A}\right)^2 + \left(\frac{\sigma_B}{B}\right)^2 - 2\frac{\sigma_{AB}}{AB} \right)

However, these formula are not very handy so we adapt it to remove possible 
division by zeros. This results in the following equations of the variance. 
The methodology is similar to Astropy.

.. math :: 

    f = AB \qquad \sigma_f^2 \approx \sigma_A^2 B^2 + \sigma_B^2 A^2  + 2AB\sigma_{AB}

    f = \frac{A}{B} \qquad \sigma_f^2 \approx \sigma_A^2 B^2 + \sigma_B^2 A^2 - 2AB\sigma_{AB}

For the case of multiplication without any variance (i.e. :math:`a` and 
:math:`\sigma_a = 0`), it simplifies to:

.. math :: 

    f = aB \qquad \sigma_f^2 \approx a^2 \sigma_B^2



.. _technical-uncertainty-exponentiation:

Exponentiation
==============

The typical equation for the propagation of variance uncertainty for 
exponentiation is:

.. math :: 

    f = A^B \qquad \sigma_f^2 \approx f^2 \left( \left(\frac{B}{A} \sigma_A\right)^2 + (\sigma_B \ln A)^2 + 2 \frac{B \ln A}{A} \sigma_{AB} \right)


.. _technical-uncertainty-logarithms:

Logarithms
==========

The typical equation for the propagation of variance uncertainty for 
exponentiation is:

.. math :: 

    f = \log_b A \qquad \sigma_f^2 = \left( \frac{\sigma_A}{A \ln b}\right)^2

Note, we assume that the logarithm base :math:`b` is exact in this case, and 
does not have any associated uncertainty to its value.



.. _technical-uncertainty-discrete-integration:

Discrete Integration
====================

[[TODO]].


.. _technical-uncertainty-weighted-mean:

Weighted Mean
=============

A weighted mean :math:`f` is defined by weights :math:`w_i` and values 
:math:`A_i`. The weights need to be normalized, thus :math:`\sum w_i = 1`. 
The typical equation for the propagation of variance uncertainty for a 
weighted mean is:

.. math :: 

    f = \frac{1}{\sum w_i} \sum w_i A_i \qquad \sigma_f^2 = \sum \left(w_i {\sigma_{A}}_i \right)^2


.. _technical-uncertainty-median:


Median
=============

[[TODO]].
