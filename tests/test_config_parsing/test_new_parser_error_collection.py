import os
from typing import Callable, Optional, TextIO

import pytest

from ert._c_wrappers.enkf import ErtConfig
from ert.parsing import ConfigValidationError

test_config_file_base = "test"
test_config_filename = f"{test_config_file_base}.ert"


def assert_error_from_config_with(
    contents: str,
    expected_line: Optional[int],
    expected_column: Optional[int],
    expected_end_column: Optional[int],
    expected_filename: str = "test.ert",
    filename: str = "test.ert",
    other_files: dict = None,
    match: str = None,
    write_after: Callable[[TextIO], None] = None,
):
    with open(filename, "w", encoding="utf-8") as fh:
        fh.write(contents)

        if write_after:
            write_after(fh)

    if other_files is not None:
        for other_filename, content in other_files.items():
            with open(other_filename, mode="w", encoding="utf-8") as fh:
                fh.writelines(content)

    collected_errors = []
    ErtConfig.from_file(
        filename, use_new_parser=True, collected_errors=collected_errors
    )

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
            ConfigValidationError.raise_from_collected(errors_matching_location)


@pytest.mark.usefixtures("use_tmpdir")
def test_info_queue_content_negative_value_invalid():
    assert_error_from_config_with(
        contents="""
NUM_REALIZATIONS  1
DEFINE <STORAGE> storage/<CONFIG_FILE_BASE>-<DATE>
RUNPATH <STORAGE>/runpath/realization-<IENS>/iter-<ITER>
ENSPATH <STORAGE>/ensemble
QUEUE_SYSTEM LOCAL
QUEUE_OPTION LOCAL MAX_RUNNING -4
        """,
        expected_line=7,
        expected_column=32,
        expected_end_column=34,
    )


@pytest.mark.usefixtures("use_tmpdir")
def test_info_summary_given_without_eclbase_gives_error(tmp_path):
    assert_error_from_config_with(
        contents="""
NUM_REALIZATIONS 1
SUMMARY summary""",
        expected_line=3,
        expected_column=1,
        expected_end_column=8,
    )


@pytest.mark.usefixtures("use_tmpdir")
def test_info_gen_kw_with_incorrect_format(tmp_path):
    assert_error_from_config_with(
        contents="""
JOBNAME my_name%d
NUM_REALIZATIONS 1
GEN_KW KW_NAME template.txt kw.txt priors.txt INIT_FILES:custom_param0
""",
        expected_line=4,
        expected_column=47,
        expected_end_column=71,
        other_files={
            "template.txt": "MY_KEYWORD <MY_KEYWORD>",
            "priors.txt": "MY_KEYWORD NORMAL 0 1",
        },
    )


@pytest.mark.usefixtures("use_tmpdir")
def test_info_gen_kw_forward_init(tmp_path):
    assert_error_from_config_with(
        contents="""
JOBNAME my_name%d
NUM_REALIZATIONS 1
GEN_KW KW_NAME template.txt kw.txt priors.txt FORWARD_INIT:True INIT_FILES:custom_param
    """,
        expected_line=4,
        expected_column=47,
        expected_end_column=64,
        other_files={
            "template.txt": "MY_KEYWORD <MY_KEYWORD>",
            "priors.txt": "MY_KEYWORD NORMAL 0 1",
        },
    )


@pytest.mark.usefixtures("use_tmpdir")
def test_info_missing_forward_model_job(tmp_path):
    assert_error_from_config_with(
        contents="""
JOBNAME my_name%d
NUM_REALIZATIONS 1
FORWARD_MODEL ECLIPSE9001
""",
        expected_line=4,
        expected_column=15,
        expected_end_column=26,
        other_files={
            "template.txt": "MY_KEYWORD <MY_KEYWORD>",
            "priors.txt": "MY_KEYWORD NORMAL 0 1",
        },
    )


@pytest.mark.usefixtures("use_tmpdir")
def test_info_positional_forward_model_args_gives_error():
    assert_error_from_config_with(
        contents="""
NUM_REALIZATIONS  1
FORWARD_MODEL RMS <IENS>
""",
        expected_line=3,
        expected_column=15,
        expected_end_column=25,
    )


@pytest.mark.usefixtures("use_tmpdir")
def test_info_unknown_run_mode_gives_error():
    assert_error_from_config_with(
        contents="""
NUM_REALIZATIONS  1
FORWARD_MODEL RMS <IENS>
HOOK_WORKFLOW MAKE_DIRECTORY PRE_SIMULATIONnn
""",
        expected_line=4,
        expected_column=30,
        expected_end_column=46,
        match="Run mode .* not supported.*",
    )


@pytest.mark.usefixtures("use_tmpdir")
def test_info_missing_simulation_job_gives_error():
    assert_error_from_config_with(
        contents="""
NUM_REALIZATIONS  1
SIMULATION_JOB this-is-not-the-job-you-are-looking-for hello
        """,
        expected_line=3,
        expected_column=16,
        expected_end_column=55,
    )


@pytest.mark.usefixtures("use_tmpdir")
def test_info_unknown_hooked_job_gives_error():
    assert_error_from_config_with(
        contents="""
NUM_REALIZATIONS  1
HOOK_WORKFLOW NO_SUCH_JOB PRE_SIMULATION
""",
        expected_line=3,
        expected_column=15,
        expected_end_column=26,
    )


@pytest.mark.usefixtures("use_tmpdir")
def test_that_non_exising_job_directory_gives_error():
    assert_error_from_config_with(
        contents="""
NUM_REALIZATIONS  1
DEFINE <STORAGE> storage/<CONFIG_FILE_BASE>-<DATE>
RUNPATH <STORAGE>/runpath/realization-<IENS>/iter-<ITER>
ENSPATH <STORAGE>/ensemble
INSTALL_JOB_DIRECTORY does_not_exist
""",
        expected_line=6,
        expected_column=23,
        expected_end_column=37,
    )


@pytest.mark.usefixtures("use_tmpdir")
def test_info_ext_job_executable_without_table(tmp_path):
    assert_error_from_config_with(
        contents="""
JOBNAME my_name%d
NUM_REALIZATIONS 1
INSTALL_JOB test THE_JOB_FILE
""",
        expected_filename=os.path.join(os.getcwd(), "THE_JOB_FILE"),
        expected_line=4,
        expected_column=18,
        expected_end_column=30,
        other_files={"THE_JOB_FILE": "EXECU missing_script.sh\n"},
        match="Item:EXECUTABLE must be set - parsing",
    )


@pytest.mark.usefixtures("use_tmpdir")
def test_info_ext_job_no_portable_exe(tmp_path):
    assert_error_from_config_with(
        contents="""
JOBNAME my_name%d
NUM_REALIZATIONS 1
INSTALL_JOB test THE_JOB_FILE
""",
        expected_filename=os.path.join(os.getcwd(), "THE_JOB_FILE"),
        expected_line=4,
        expected_column=18,
        expected_end_column=30,
        other_files={"THE_JOB_FILE": "PORTABLE_EXE never executed\n"},
        match='"PORTABLE_EXE" key is deprecated, please replace with "EXECUTABLE"',
    )


@pytest.mark.usefixtures("use_tmpdir")
def test_info_ext_job_missing_executable_file(tmp_path):
    assert_error_from_config_with(
        contents="""
JOBNAME my_name%d
NUM_REALIZATIONS 1
INSTALL_JOB test THE_JOB_FILE
""",
        expected_filename=os.path.join(os.getcwd(), "THE_JOB_FILE"),
        expected_line=4,
        expected_column=18,
        expected_end_column=30,
        other_files={"THE_JOB_FILE": "EXECUTABLE not_found.sh\n"},
        match="Could not find executable",
    )


@pytest.mark.usefixtures("use_tmpdir")
def test_info_ext_job_is_directory(tmp_path):
    assert_error_from_config_with(
        contents="""
JOBNAME my_name%d
NUM_REALIZATIONS 1
INSTALL_JOB test THE_JOB_FILE
""",
        expected_filename=os.path.join(os.getcwd(), "THE_JOB_FILE"),
        expected_line=4,
        expected_column=18,
        expected_end_column=30,
        other_files={"THE_JOB_FILE": "EXECUTABLE /tmp\n"},
        match="executable set to directory",
    )


@pytest.mark.usefixtures("use_tmpdir")
def test_info_workflow_job_not_found(tmp_path):
    assert_error_from_config_with(
        contents="""
JOBNAME my_name%d
NUM_REALIZATIONS 1
LOAD_WORKFLOW_JOB non_existing workflow_job_name
""",
        expected_line=4,
        expected_column=19,
        expected_end_column=31,
        match="Could not open",
    )


@pytest.mark.usefixtures("use_tmpdir")
def test_info_unexpected_characters(tmp_path):
    assert_error_from_config_with(
        contents="""
$$#{{}$#JOBNAME my_name%d
NUM_REALIZATIONS 1
LOAD_WORKFLOW_JOB non_existing workflow_job_name
""",
        expected_line=2,
        expected_column=1,
        expected_end_column=None,
    )


def write_dirty_bytes(fh: TextIO):
    with open(fh.name, mode="wb") as bfile:
        bfile.seek(0)
        bfile.write(b'\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff')


@pytest.mark.usefixtures("use_tmpdir")
def test_info_unicode_decode_error(tmp_path):
    assert_error_from_config_with(
        contents="""
""",
        expected_line=None,
        expected_column=None,
        expected_end_column=None,
        write_after=write_dirty_bytes,
        match="Unsupported non UTF",
    )
