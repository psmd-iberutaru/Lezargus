"""Tests wrapper functions to ensure that they are behaving normally."""

import numpy as np

import lezargus


def test_wavelength_overlap_fraction() -> None:
    """Test the wavelength_overlap_fraction function.

    Parameters
    ----------
    None

    Returns
    -------
    None

    """
    # We first check the obvious cases, where two different arrays overlap
    # or do not.
    main_array = np.linspace(5, 15, 50)
    overlap_array = np.linspace(7, 13, 30)
    super_overlap_array = np.linspace(4, 20, 60)
    lower_outside_array = np.linspace(1, 3, 15)
    upper_outside_array = np.linspace(17, 21, 35)
    # These arrays partially overlap.
    lower_partial_array = np.linspace(3, 8, 20)
    upper_partial_array = np.linspace(11, 23, 22)

    # Testing them...
    # An overlapping array overlaps with itself perfectly.
    self_fraction = lezargus.library.wrapper.wavelength_overlap_fraction(
        base=main_array,
        contain=main_array,
    )
    assert_message = (
        f"Overlap fraction for itself is expected to be 1, not {self_fraction}"
    )
    assert self_fraction == 1, assert_message
    # Then the expected overlap one.
    overlap_fraction = lezargus.library.wrapper.wavelength_overlap_fraction(
        base=main_array,
        contain=overlap_array,
    )
    assert_message = (
        "Overlap fraction for the overlapping array is expected to be 1,"
        f" not {overlap_fraction}"
    )
    # And one where the contain array is much bigger. We expect this to
    # throw a warning.
    try:
        __ = lezargus.library.wrapper.wavelength_overlap_fraction(
            base=main_array,
            contain=super_overlap_array,
        )
    except lezargus.library.logging.ElevatedError:
        # This error is expected.
        pass
    else:
        assert_message = (
            "The super overlap fraction call should have thrown some sort of"
            " elevated warning, which then should have been caught."
        )
        assert False, assert_message
    # Now we test the two arrays which are not overlapping at all.
    lower_fraction = lezargus.library.wrapper.wavelength_overlap_fraction(
        base=main_array,
        contain=lower_outside_array,
    )
    assert_message = (
        "Overlap fraction for the lower non-overlapping array is expected to be"
        f" 0, not {lower_fraction}"
    )
    assert lower_fraction == 0, assert_message
    upper_fraction = lezargus.library.wrapper.wavelength_overlap_fraction(
        base=main_array,
        contain=upper_outside_array,
    )
    assert_message = (
        "Overlap fraction for the upper non-overlapping array is expected to be"
        f" 0, not {upper_fraction}"
    )
    assert upper_fraction == 0, assert_message
    # And then the partial overlaps.
    lower_part_fraction = lezargus.library.wrapper.wavelength_overlap_fraction(
        base=main_array,
        contain=lower_partial_array,
    )
    assert_message = (
        "Overlap fraction for the lower partial overlapping array is expected"
        f" to be between 0 and 1, not {lower_part_fraction}"
    )
    assert 0 < lower_part_fraction < 1, assert_message
    upper_part_fraction = lezargus.library.wrapper.wavelength_overlap_fraction(
        base=main_array,
        contain=upper_partial_array,
    )
    assert_message = (
        "Overlap fraction for the upper partial overlapping array is expected"
        f" to be between 0 and 1, not {upper_part_fraction}"
    )
    assert 0 < upper_part_fraction < 1, assert_message
