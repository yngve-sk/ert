from __future__ import annotations

import importlib.util
import inspect
import logging
import sys
import traceback
from abc import abstractmethod
from collections.abc import Callable
from types import ModuleType
from typing import TYPE_CHECKING, Any, TypeAlias

from ert.config.workflow_fixtures import WorkflowFixtures

if TYPE_CHECKING:
    from ert.config import ErtConfig
    from ert.storage import Ensemble, Storage

    Fixtures: TypeAlias = ErtConfig | Ensemble | Storage
logger = logging.getLogger(__name__)


class ErtScript:
    """
    ErtScript is the abstract baseclass for workflow jobs and
    plugins. It provides access to the ert internals and lets
    jobs implement the "run" function which is called when
    a workflow is executed.
    """

    stop_on_fail = False

    def __init__(
        self,
    ) -> None:
        self.__is_cancelled = False
        self.__failed = False
        self._stdoutdata = ""
        self._stderrdata = ""

    @abstractmethod
    def run(self, *arg: Any, **kwarg: Any) -> Any:
        """
        This method is implemented by the workflow runners
        and executed when the workflow job is called.

        The parameters are gotten from the workflow file, e.g. a
        workflow file containing

        EXPORT_MISFIT_DATA path/to/output.hdf

        will put `path/to/output.hdf` in the first argument
        to run.
        """

    @property
    def stdoutdata(self) -> str:
        if isinstance(self._stdoutdata, bytes):
            self._stdoutdata = self._stdoutdata.decode()
        return self._stdoutdata

    @property
    def stderrdata(self) -> str:
        if isinstance(self._stderrdata, bytes):
            self._stderrdata = self._stderrdata.decode()
        return self._stderrdata

    def isCancelled(self) -> bool:
        return self.__is_cancelled

    def hasFailed(self) -> bool:
        return self.__failed

    def cancel(self) -> None:
        self.__is_cancelled = True

    def cleanup(self) -> None:
        """
        Override to perform cleanup after a run.
        """

    def initializeAndRun(
        self,
        argument_types: list[type[Any]],
        argument_values: list[str],
        fixtures: WorkflowFixtures | None = None,
        **kwargs: dict[str, Any],
    ) -> Any:
        fixtures = {} if fixtures is None else fixtures
        workflow_args = []
        for index, arg_value in enumerate(argument_values):
            arg_type = argument_types[index] if index < len(argument_types) else str

            if arg_value is not None:
                workflow_args.append(arg_type(arg_value))
            else:
                workflow_args.append(None)

        fixtures["workflow_args"] = workflow_args

        fixture_args = []
        all_func_args = inspect.signature(self.run).parameters
        is_using_wf_args_fixture = "workflow_args" in all_func_args

        try:
            if not is_using_wf_args_fixture:
                fixture_or_kw_arguments = list(all_func_args)[len(workflow_args) :]
            else:
                fixture_or_kw_arguments = list(all_func_args)

            func_args = {k: all_func_args[k] for k in fixture_or_kw_arguments}

            kwargs_defaults = {
                k: v.default
                for k, v in func_args.items()
                if k not in fixtures
                and v.kind != v.VAR_POSITIONAL
                and not str(v).startswith("*")
                and v.default != v.empty
            }
            use_kwargs = {
                k: (kwargs or {}).get(k, default_value)
                for k, default_value in ({**kwargs_defaults, **kwargs}).items()
            }
            # If the user has specified *args, we skip injecting fixtures, and just
            # pass the user configured arguments
            if not any(p.kind == p.VAR_POSITIONAL for p in func_args.values()):
                try:
                    fixture_args = self.insert_fixtures(func_args, fixtures, use_kwargs)
                except ValueError as e:
                    # This is here for backwards compatibility, the user does not have *argv
                    # but positional arguments. Can not be mixed with using fixtures.
                    logger.warning(
                        f"Mixture of fixtures and positional arguments, err: {e}"
                    )

            positional_args = (
                fixture_args
                if is_using_wf_args_fixture
                else [*workflow_args, *fixture_args]
            )
            if not positional_args and not use_kwargs:
                return self.run()
            elif positional_args and not use_kwargs:
                return self.run(*positional_args)
            elif not positional_args and use_kwargs:
                return self.run(**use_kwargs)
            else:
                return self.run(*positional_args, **use_kwargs)
        except AttributeError as e:
            error_msg = str(e)
            if not hasattr(self, "run"):
                error_msg = "No 'run' function implemented"
            self.output_stack_trace(error=error_msg)
            return None
        except KeyboardInterrupt:
            error_msg = "Script cancelled (CTRL+C)"
            self.output_stack_trace(error=error_msg)
            return None
        except UserWarning as uw:
            self.__failed = True
            return uw.args[0]
        except Exception as e:
            full_trace = "".join(traceback.format_exception(*sys.exc_info()))
            self.output_stack_trace(f"{e!s}\n{full_trace}")
            return None
        finally:
            self.cleanup()

    # Need to have unique modules in case of identical object naming in scripts
    __module_count = 0

    def insert_fixtures(
        self,
        func_args: dict[str, inspect.Parameter],
        fixtures: WorkflowFixtures,
        kwargs: dict[str, Any],
    ) -> list[Any]:
        arguments = []
        errors = []
        for val in func_args:
            if val in fixtures:
                arguments.append(fixtures.get(val))
            elif val not in kwargs:
                errors.append(val)
        if errors:
            kwargs_str = ",".join(f"{k}='{v}'" for k, v in kwargs.items())
            raise ValueError(
                f"Plugin: {self.__class__.__name__} misconfigured, arguments: {errors} "
                f"not found in fixtures: {list(fixtures)} or kwargs {kwargs_str}"
            )
        return arguments

    def output_stack_trace(self, error: str = "") -> None:
        stack_trace = error or "".join(traceback.format_exception(*sys.exc_info()))
        sys.stderr.write(
            f"The script '{self.__class__.__name__}' caused an "
            f"error while running:\n{str(stack_trace).strip()}\n"
        )

        self._stderrdata = error
        self.__failed = True

    @staticmethod
    def loadScriptFromFile(
        path: str,
    ) -> Callable[[], ErtScript]:
        module_name = f"ErtScriptModule_{ErtScript.__module_count}"
        ErtScript.__module_count += 1

        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None:
            raise ValueError(f"Could not find spec for {module_name}")
        module = importlib.util.module_from_spec(spec)
        if module is None:
            raise ValueError(f"Could not find {module_name} with spec {spec}")
        if spec.loader is None:
            raise ValueError(f"No loader for module {module} with spec {spec}")
        try:
            spec.loader.exec_module(module)
        except (SyntaxError, ImportError) as err:
            raise ValueError(f"ErtScript {path} contains syntax error {err}") from err
        return ErtScript.__findErtScriptImplementations(module)

    @staticmethod
    def __findErtScriptImplementations(
        module: ModuleType,
    ) -> Callable[[], ErtScript]:
        result = []
        for _, member in inspect.getmembers(
            module,
            lambda member: inspect.isclass(member)
            and member.__module__ == module.__name__,
        ):
            if ErtScript in inspect.getmro(member):
                result.append(member)

        if len(result) == 0:
            raise ValueError(f"Module {module.__name__} does not contain an ErtScript!")
        if len(result) > 1:
            raise ValueError(
                f"Module {module.__name__} contains more than one ErtScript"
            )
        return result[0]

    @staticmethod
    def validate(args: list[Any]) -> None:
        """
        If the workflow has problems it can validate against
        the arguments on startup. If it raises ConfigValidationError
        this will be caught and presented to the user.
        """
