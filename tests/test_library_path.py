"""Tests pathname manipulations."""

import sys

import conftest

import lezargus

# The operating system that is running, if it is Windows, the tests for
# path-names are different.
IS_OPERATING_SYSTEM_WINDOWS = sys.platform.startswith("win")


def test_get_directory() -> None:
    """Test the ability to get the directory from a pathname.

    Parameters
    ----------
    None

    Returns
    -------
    None

    """
    if IS_OPERATING_SYSTEM_WINDOWS:
        # Example of a Windows OS pathname with spaces and other interesting
        # characters.
        example_windows_pathname = (
            R"A:\Kalos\Music\Pokémon\Official Tracks\Mystery Dungeon\PMD"
            R" Explorers\Dialga's Fight to the Finish!.flac"
        )
        # Getting the directory.
        directory = lezargus.library.path.get_directory(
            pathname=example_windows_pathname,
        )
        expected_dir = (
            R"A:\Kalos\Music\Pokémon\Official Tracks\Mystery Dungeon\PMD"
            R" Explorers"
        )
        # Asserting.
        assert_message = "Windows based pathname fail."
        assert directory == expected_dir, assert_message
    else:
        # Example of a Linux OS pathname with spaces and other interesting
        # characters.
        example_linux_pathname = (
            R"/home/sparrow/Kirby/星のカービィ (Hoshi no Kaabii) (2001) -"
            R" Episode 1 - Kirby"
            R" - Right Back at Ya! Japanese [a9vrQ3Ns0gg].mkv"
        )
        # Getting the directory.
        directory = lezargus.library.path.get_directory(
            pathname=example_linux_pathname,
        )
        expected_dir = R"/home/sparrow/Kirby"
        # Asserting.
        assert_message = "Linux based pathnames fail."
        assert directory == expected_dir, assert_message


def test_get_most_recent_filename_in_directory() -> None:
    """Test the get_most_recent_filename_in_directory function.

    Parameters
    ----------
    None

    Returns
    -------
    None

    """
    # First we test if we provide an invalid directory.
    try:
        no_directory = "/this/is/a/directory/which/really/should/not/exist/"
        __ = lezargus.library.path.get_most_recent_filename_in_directory(
            directory=no_directory,
        )
    except lezargus.library.logging.InputError:
        # The error is expected.
        pass

    # We just use the test directory itself, because why not. We also test
    # different extensions.
    recent_directory = conftest.fetch_test_filename(basename="")
    __ = lezargus.library.path.get_most_recent_filename_in_directory(
        directory=recent_directory,
        extension=None,
        recursive=True,
    )
    __ = lezargus.library.path.get_most_recent_filename_in_directory(
        directory=recent_directory,
        extension=["txt"],
        recursive=True,
    )


def test_get_filename_without_extension() -> None:
    """Test the ability to get the filename without extension from a pathname.

    Parameters
    ----------
    None

    Returns
    -------
    None

    """
    if IS_OPERATING_SYSTEM_WINDOWS:
        # Example of a Windows OS pathname with spaces and other interesting
        # characters.
        example_windows_pathname = (
            R"A:\Kalos\Music\Pokémon\Official Tracks\Mystery Dungeon\PMD"
            R" Explorers\Dialga's Fight to the Finish!.flac"
        )
        # Getting the directory.
        filename = lezargus.library.path.get_filename_without_extension(
            pathname=example_windows_pathname,
        )
        expected_filename = R"Dialga's Fight to the Finish!"
        # Asserting.
        assert_message = "Windows based pathnames fail."
        assert filename == expected_filename, assert_message
    else:
        # Example of a Linux OS pathname with spaces and other interesting
        # characters.
        example_linux_pathname = (
            R"/home/sparrow/Kirby/星のカービィ (Hoshi no Kaabii) (2001) -"
            R" Episode 1 - Kirby"
            R" - Right Back at Ya! Japanese [a9vrQ3Ns0gg].mkv"
        )
        # Getting the directory.
        filename = lezargus.library.path.get_filename_without_extension(
            pathname=example_linux_pathname,
        )
        expected_filename = (
            R"星のカービィ (Hoshi no Kaabii) (2001) - Episode 1 - Kirby - Right"
            R" Back at Ya!"
            R" Japanese [a9vrQ3Ns0gg]"
        )
        # Asserting.
        assert_message = "Linux based pathnames fail."
        assert filename == expected_filename, assert_message


def test_get_filename_with_extension() -> None:
    """Test the ability to get the filename with extension from a pathname.

    Parameters
    ----------
    None

    Returns
    -------
    None

    """
    if IS_OPERATING_SYSTEM_WINDOWS:
        # Example of a Windows OS pathname with spaces and other interesting
        # characters.
        example_windows_pathname = (
            R"A:\Kalos\Music\Pokémon\Official Tracks\Mystery Dungeon\PMD"
            R" Explorers\Dialga's Fight to the Finish!.flac"
        )
        # Getting the directory.
        filename = lezargus.library.path.get_filename_with_extension(
            pathname=example_windows_pathname,
        )
        expected_filename = R"Dialga's Fight to the Finish!.flac"
        # Asserting.
        assert_message = "Windows based pathnames fail."
        assert filename == expected_filename, assert_message
    else:
        # Example of a Linux OS pathname with spaces and other interesting
        # characters.
        example_linux_pathname = (
            R"/home/sparrow/Kirby/星のカービィ (Hoshi no Kaabii) (2001) -"
            R" Episode 1 - Kirby"
            R" - Right Back at Ya! Japanese [a9vrQ3Ns0gg].mkv"
        )
        # Getting the directory.
        filename = lezargus.library.path.get_filename_with_extension(
            pathname=example_linux_pathname,
        )
        expected_filename = (
            R"星のカービィ (Hoshi no Kaabii) (2001) - Episode 1 - Kirby - Right"
            R" Back at Ya!"
            R" Japanese [a9vrQ3Ns0gg].mkv"
        )
        # Asserting.
        assert_message = "Linux based pathnames fail."
        assert filename == expected_filename, assert_message


def test_get_file_extension() -> None:
    """Test the ability to get the file extension from a pathname.

    Parameters
    ----------
    None

    Returns
    -------
    None

    """
    if IS_OPERATING_SYSTEM_WINDOWS:
        # Example of a Windows OS pathname with spaces and other interesting
        # characters.
        example_windows_pathname = (
            R"A:\Kalos\Music\Pokémon\Official Tracks\Mystery Dungeon\PMD"
            R" Explorers\Dialga's Fight to the Finish!.flac"
        )
        # Getting the directory.
        extension = lezargus.library.path.get_file_extension(
            pathname=example_windows_pathname,
        )
        expected_extension = R"flac"
        # Asserting.
        assert_message = "Windows based pathnames fail."
        assert extension == expected_extension, assert_message
    else:
        # Example of a Linux OS pathname with spaces and other interesting
        # characters.
        example_linux_pathname = (
            R"/home/sparrow/Kirby/星のカービィ (Hoshi no Kaabii) (2001) -"
            R" Episode 1 - Kirby"
            R" - Right Back at Ya! Japanese [a9vrQ3Ns0gg].mkv"
        )
        # Getting the directory.
        extension = lezargus.library.path.get_file_extension(
            pathname=example_linux_pathname,
        )
        expected_extension = R"mkv"
        # Asserting.
        assert_message = "Linux based pathnames fail."
        assert extension == expected_extension, assert_message


def test_merge_pathname() -> None:
    """Test the ability to merge a pathname.

    Parameters
    ----------
    None

    Returns
    -------
    None

    """
    if IS_OPERATING_SYSTEM_WINDOWS:
        # For Windows based pathnames.
        windows_directory = R"A:\Kalos\Pictures\Space Battleship Yamato"
        windows_filename = R"Space Battle Ship USS Arizona"
        windows_extension = R"jpg"
        # Merging with an extension.
        windows_pathname = lezargus.library.path.merge_pathname(
            directory=windows_directory,
            filename=windows_filename,
            extension=windows_extension,
        )
        windows_expected_pathname = (
            R"A:\Kalos\Pictures\Space Battleship Yamato\Space Battle Ship USS"
            R" Arizona.jpg"
        )
        assert_message = (
            "Windows pathname merging with an extension did not work."
        )
        assert windows_pathname == windows_expected_pathname, assert_message
        # Merging without an extension.
        windows_pathname = lezargus.library.path.merge_pathname(
            directory=windows_directory,
            filename=windows_filename,
            extension="",
        )
        windows_expected_pathname = (
            R"A:\Kalos\Pictures\Space Battleship Yamato\Space Battle Ship USS"
            R" Arizona"
        )
        assert_message = (
            "Windows pathname merging without an extension did not work."
        )
        assert windows_pathname == windows_expected_pathname, assert_message
    else:
        # For Linux based pathnames.
        linux_directory = R"/home/sparrow/test/wiki"
        linux_filename = R"docker-compose"
        linux_extension = R"yml"
        # Merging with an extension.
        linux_pathname = lezargus.library.path.merge_pathname(
            directory=linux_directory,
            filename=linux_filename,
            extension=linux_extension,
        )
        linux_expected_pathname = R"/home/sparrow/test/wiki/docker-compose.yml"
        assert_message = (
            "Linux pathname merging with an extension did not work."
        )
        assert linux_pathname == linux_expected_pathname, assert_message
        # Merging without an extension.
        linux_pathname = lezargus.library.path.merge_pathname(
            directory=linux_directory,
            filename=linux_filename,
            extension="",
        )
        linux_expected_pathname = R"/home/sparrow/test/wiki/docker-compose"
        assert_message = (
            "Linux pathname merging with an extension did not work."
        )
        assert linux_pathname == linux_expected_pathname, assert_message


def test_split_pathname() -> None:
    """Test the ability to split a pathname.

    Parameters
    ----------
    None

    Returns
    -------
    None

    """
    if IS_OPERATING_SYSTEM_WINDOWS:
        # For Windows based pathnames.
        windows_pathname = (
            R"A:\Kalos\Pictures\Space Battleship Yamato\Space Battle Ship USS"
            R" Arizona.jpg"
        )
        # Splitting.
        (
            windows_directory,
            windows_filename,
            windows_extension,
        ) = lezargus.library.path.split_pathname(pathname=windows_pathname)
        expected_windows_directory = (
            R"A:\Kalos\Pictures\Space Battleship Yamato"
        )
        expected_windows_filename = R"Space Battle Ship USS Arizona"
        expected_windows_extension = R"jpg"

        assert_message = "Windows pathname splitting did not work."
        assert (
            windows_directory == expected_windows_directory
            and windows_filename == expected_windows_filename
            and windows_extension == expected_windows_extension
        ), assert_message
    else:
        # For Linux based pathnames.
        linux_pathname = R"/home/sparrow/test/wiki/docker-compose.yml"
        # Splitting.
        (
            linux_directory,
            linux_filename,
            linux_extension,
        ) = lezargus.library.path.split_pathname(pathname=linux_pathname)
        expected_linux_directory = R"/home/sparrow/test/wiki"
        expected_linux_filename = R"docker-compose"
        expected_linux_extension = R"yml"

        assert_message = "Linux pathname splitting did not work."
        assert (
            linux_directory == expected_linux_directory
            and linux_filename == expected_linux_filename
            and linux_extension == expected_linux_extension
        ), assert_message
