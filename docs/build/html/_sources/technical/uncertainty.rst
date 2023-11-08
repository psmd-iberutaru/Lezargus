=======================
Uncertainty Propagation
=======================

We follow the typical fundamental equations for the propagation of uncertainty 
for simple operations. We describe the methods for each operation as follows.

Addition and Subtraction
========================
[[TODO]]


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
The methodology is similar to Astropy. We calculate the standard deviation 
uncertainty from the variance.

.. math :: 

    f = AB \qquad \sigma_f^2 \approx \sigma_A^2 B^2 + \sigma_B^2 A^2  + 2AB\sigma_{AB}

    f = \frac{A}{B} \qquad \sigma_f^2 \approx \sigma_A^2 B^2 + \sigma_B^2 A^2 - 2AB\sigma_{AB}