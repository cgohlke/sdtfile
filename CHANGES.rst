Revisions
---------

2026.1.14

- Improve code quality.

2025.12.12

- Add new SPC modules and MEASURE_INFO_EXT fields.
- Drop support for Python 3.10.

2025.5.10

- Support Python 3.14.

2025.3.25

- Fix shape of data with routing channels.
- Drop support for Python 3.9, support Python 3.13.

2024.12.6

- Fix read MeasureInfo fields as scalars (breaking).
- Update some structure field names with BH reference (breaking).
- Parse some SetupBlock binary structures (#7).
- Include more information in str(SdtFile).
- Add subtype to FileRevision.

2024.11.24

- Update MEASURE_INFO_EXT struct to SPCM v.9.9 (breaking).

2024.5.24

- Fix docstring examples not correctly rendered on GitHub.

2024.4.24

- Support NumPy 2.

2023.9.28

- Update structs to SPCM v.9.66 (breaking).
- Shorten MEASURE_INFO struct to meas_desc_block_length.

2023.8.30

- Fix linting issues.
- Add py.typed marker.
- Drop support for Python 3.8 and numpy < 1.22 (NEP29).

2022.9.28

- Convert docstrings to Google style with Sphinx directives.

2022.2.2

- Add type hints.
- Drop support for Python 3.7 and numpy < 1.19 (NEP29).

2021.11.18

- Fix reading FLIM files created by Prairie View software (#5).

2021.3.21

- Add sdt2dat script.

2020.12.10

- Fix shape of non-square frames.

2020.8.3

- Fix integer overflow (#3).
- Support os.PathLike file names.

2020.1.1

- Fix reading MCS_BLOCK data.
- Remove support for Python 2.7 and 3.5.
- Update copyright.

2019.7.28

- Fix reading compressed, multi-channel data.

2018.9.22

- Use str, not bytes for ASCII data.

2018.8.29

- Move module into sdtfile package.

2018.2.7

- Bug fixes.

2016.3.30

- Support revision 15 files and compression.

2015.1.29

- Read SPC DLL data files.

2014.9.5

- Fix reading multiple MEASURE_INFO records.
- …
