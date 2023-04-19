import os
import stat
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, TextIO, Union

import pytest

from ert._c_wrappers.enkf import ErtConfig
from ert.parsing import ConfigValidationError

test_config_file_base = "test"
test_config_filename = f"{test_config_file_base}.ert"


@dataclass
class FileDetail:
    contents: str
    is_executable: bool = False


def assert_error_from_config_with(
    contents: str,
    expected_line: Optional[int],
    expected_column: Optional[int],
    expected_end_column: Optional[int],
    expected_filename: str = "test.ert",
    filename: str = "test.ert",
    other_files: Union[dict, FileDetail] = None,
    match: str = None,
    write_after: Callable[[TextIO], None] = None,
):
    with open(filename, "w", encoding="utf-8") as fh:
        fh.write(contents)

        if write_after:
            write_after(fh)

    if other_files is not None:
        for other_filename, content in other_files.items():
            if isinstance(content, str):
                with open(other_filename, mode="w", encoding="utf-8") as fh:
                    fh.writelines(content)
            else:
                content: FileDetail
                with open(other_filename, mode="w", encoding="utf-8") as fh:
                    fh.writelines(content.contents)

                if not content.is_executable:
                    os.chmod(other_filename, stat.S_IREAD)

    try:
        ErtConfig.from_file(filename, use_new_parser=True)
    except ConfigValidationError as err:
        collected_errors = err.errors

        # Find errors in matching file
        errors_matching_filename = (
            [err for err in collected_errors if expected_filename in err.filename]
            if expected_filename is not None
            else collected_errors
        )

        if expected_filename is not None:
            assert (
                len(errors_matching_filename) > 0
            ), f"Expected minimum 1 error matching filename {expected_filename}, got 0"
        else:
            assert (
                len(errors_matching_filename) > 0
            ), "Expected minimum 1 collected error, got 0"

    def equals_or_expected_any(actual, expected):
        return True if expected is None else actual == expected

    # Find errors matching location in file
    errors_matching_location = [
        x
        for x in errors_matching_filename
        if equals_or_expected_any(x.line, expected_line)
        and equals_or_expected_any(x.column, expected_column)
        and equals_or_expected_any(x.end_column, expected_end_column)
    ]

    assert len(errors_matching_location) > 0, (
        f"Expected to find at least 1 error matching location"
        f"(line={'*' if expected_line is None else expected_line},"
        f"column={'*' if expected_column is None else expected_column},"
        f"end_column{'*' if expected_end_column is None else expected_end_column})"
    )

    if match is not None:
        with pytest.raises(ConfigValidationError, match=match):
            raise ConfigValidationError(errors_matching_location)


@dataclass
class ExpectedErrorInfo:
    expected_line: Optional[int] = None
    expected_column: Optional[int] = None
    expected_end_column: Optional[int] = None
    expected_filename: str = "test.ert"
    filename: str = "test.ert"
    other_files: Dict[str, Union[str, FileDetail]] = None
    match: Optional[str] = None


def assert_multiple_errors_from_config_with(
    contents: str,
    expected_errors: List[ExpectedErrorInfo],
    write_after: Callable[[TextIO], None] = None,
):
    for info in expected_errors:
        assert_error_from_config_with(
            contents=contents,
            expected_line=info.expected_line,
            expected_column=info.expected_column,
            expected_end_column=info.expected_end_column,
            expected_filename=info.expected_filename,
            filename=info.filename,
            other_files=info.other_files,
            match=info.match,
            write_after=write_after,
        )


# "-4 -5" is parsed as the final arg, intended behavior?...
@pytest.mark.usefixtures("use_tmpdir")
def test_info_disallowed_kw():
    assert_error_from_config_with(
        contents="""
QUEUE_OPTION DOCAL MAX_RUNNING 4
        """,
        expected_line=2,
        expected_column=14,
        expected_end_column=19,
        match="argument .* must be one of",
    )


@pytest.mark.usefixtures("use_tmpdir")
def test_info_from_allow_constraints():
    assert_multiple_errors_from_config_with(
        contents="""
QUEUE_OPTION DOCAL MAX_RUNNING 4
STOP_LONG_RUNNING flase
NUM_REALIZATIONS not_int
ENKF_ALPHA not_float
RUN_TEMPLATE dsajldkald/sdjkahsjka/wqehwqhdsa
JOB_SCRIPT dnsjklajdlksaljd/dhs7sh/qhwhe
JOB_SCRIPT non_executable_file
NUM_REALIZATIONS 1 2 3 4 5
NUM_REALIZATIONS

        """,
        expected_errors=[
            ExpectedErrorInfo(
                expected_line=2,
                expected_column=14,
                expected_end_column=19,
                match="argument .* must be one of",
            ),
            ExpectedErrorInfo(
                expected_line=3,
                expected_column=19,
                expected_end_column=24,
                match="must have a boolean value",
            ),
            ExpectedErrorInfo(
                expected_line=4,
                expected_column=18,
                expected_end_column=25,
                match="must have an integer value",
            ),
            ExpectedErrorInfo(
                expected_line=5,
                expected_column=12,
                expected_end_column=21,
                match="must have a number",
            ),
            ExpectedErrorInfo(
                expected_line=6,
                expected_column=14,
                expected_end_column=46,
                match="Cannot find file or directory",
            ),
            ExpectedErrorInfo(
                expected_line=7,
                expected_column=12,
                expected_end_column=41,
                match="Could not find executable",
            ),
            ExpectedErrorInfo(
                other_files={
                    "non_executable_file": FileDetail(contents="", is_executable=False)
                },
                expected_line=8,
                expected_column=12,
                expected_end_column=31,
                match="File not executable",
            ),
            ExpectedErrorInfo(
                expected_line=9,
                expected_column=1,
                expected_end_column=17,
                match="must have maximum",
            ),
            ExpectedErrorInfo(
                expected_line=10,
                expected_column=1,
                expected_end_column=17,
                match="must have at least",
            ),
        ],
    )