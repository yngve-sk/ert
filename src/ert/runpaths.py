from collections.abc import Iterable
from pathlib import Path
from typing import Any

from pydantic import BaseModel, field_validator

from ert.config.model_config import DEFAULT_JOBNAME_FORMAT, DEFAULT_RUNPATH, \
    _replace_runpath_format
from ert.substitutions import Substitutions


class Runpaths(BaseModel):
    """The Runpaths are the ensemble workspace directories.

    Generally there is one runpath for each realization and iteration, although
    depending on the given format string for the paths they may coincide. There
    is one job name for each of the runpaths.

    :param jobname_format: The format of the job name, e.g., "job_<IENS>"
    :param runpath_format: The format of the runpath, e.g.
        "/path/<case>/ensemble-<IENS>/iteration-<ITER>"
    :param filename: The filename of the runpath list file. Defaults to
        ".ert_runpath_list".
    :param substitute: Function called to perform arbitrary substitution on
        jobname and runpath, e.g., transforms
        "/path/<case>/ensemble-1/iteration-2/" to
        "/path/my_case/ensemble-1/iteration-2". The function is given
        three arguments, the defaults to no substitutions, ie.:

            def default_substitute(to_replace:str, realization:int, iteration:int):
                return to_replace


    """

    jobname_format: str = DEFAULT_JOBNAME_FORMAT
    runpath_format: str = DEFAULT_RUNPATH
    runpath_file: str = ".ert_runpath_list"
    substitutions: Substitutions | None = None
    eclbase: str | None = None

    @field_validator("eclbase", mode="before")
    @classmethod
    def transform(cls, eclbase: str) -> str:
        return _replace_runpath_format(eclbase)

    @field_validator("jobname_format", mode="before")
    @classmethod
    def transform(cls, jobname_format: str) -> str:
        return _replace_runpath_format(jobname_format)

    def model_post_init(self, context: Any) -> None:
        self.runpath_file = Path(self.runpath_file)
        self.runpath_format = str(Path(self.runpath_format).resolve())
        self.substitutions = self.substitutions or Substitutions()

    def set_ert_ensemble(self, ensemble_name: str) -> None:
        self.substitutions["<ERT-CASE>"] = ensemble_name
        self.substitutions["<ERTCASE>"] = ensemble_name

    def get_paths(self, realizations: Iterable[int], iteration: int) -> list[str]:
        return [
            self.substitutions.substitute_real_iter(
                self.runpath_format, realization, iteration
            )
            for realization in realizations
        ]

    def get_jobnames(self, realizations: Iterable[int], iteration: int) -> list[str]:
        return [
            self.substitutions.substitute_real_iter(
                self.jobname_format, realization, iteration
            )
            for realization in realizations
        ]

    def write_runpath_list(
        self,
        iteration_numbers: list[int],
        realization_numbers: list[int],
    ) -> None:
        """Writes the runpath_list_file, which lists jobs and runpaths.

        The runpath list file is parsed by some workflows in order to find
        which path was used by each iteration and ensemble.

        Calling write_runpath_list([0,1], [3,4]) with "/cwd/" as the
        current working directory will result in a runpath list file containing:

            003  /cwd/realization-3/iteration-0  job3  000
            004  /cwd/realization-4/iteration-0  job4  000
            003  /cwd/realization-3/iteration-1  job3  001
            004  /cwd/realization-4/iteration-1  job4  001

        The example assumes that jobname_format is "job<IENS>", that there is
        no eclbase and runpath_format is "realization<ITER>/iteration-<IENS>"

        :param iteration_numbers: The list of iterations to write entries for
        :param realization_numbers: The list of realizations to write entries for
        """
        Path(self.runpath_file).parent.mkdir(parents=True, exist_ok=True)
        with open(self.runpath_file, "w", encoding="utf-8") as filehandle:
            for iteration in iteration_numbers:
                for realization in realization_numbers:
                    job_name_or_eclbase = self.substitutions.substitute_real_iter(
                        self.eclbase or self.jobname_format,
                        realization,
                        iteration,
                    )
                    runpath = self.substitutions.substitute_real_iter(
                        self.runpath_format, realization, iteration
                    )

                    filehandle.write(
                        f"{realization:03d}  {runpath}  "
                        f"{job_name_or_eclbase}  {iteration:03d}\n"
                    )
