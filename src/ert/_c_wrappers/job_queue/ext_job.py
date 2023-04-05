import logging
import os
import os.path
import re
import shutil
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ert._c_wrappers.config import ConfigParser, ContentTypeEnum
from ert.parsing import ConfigValidationError
from ert._c_wrappers.util import SubstitutionList
from ert._clib.job_kw import type_from_kw

_SUBSTITUTED_AT_EXECUTION_TIME: List[str] = ["<ITER>", "<IENS>"]

logger = logging.getLogger(__name__)


class ExtJobInvalidArgsException(BaseException):
    pass


@dataclass
class ExtJob:
    name: str
    executable: str
    stdin_file: Optional[str] = None
    stdout_file: Optional[str] = None
    stderr_file: Optional[str] = None
    start_file: Optional[str] = None
    target_file: Optional[str] = None
    error_file: Optional[str] = None
    max_running: Optional[int] = None
    max_running_minutes: Optional[int] = None
    min_arg: Optional[int] = None
    max_arg: Optional[int] = None
    arglist: List[str] = field(default_factory=list)
    arg_types: List[ContentTypeEnum] = field(default_factory=list)
    environment: Dict[str, str] = field(default_factory=dict)
    exec_env: Dict[str, str] = field(default_factory=dict)
    default_mapping: Dict[str, str] = field(default_factory=dict)
    private_args: SubstitutionList = field(default_factory=SubstitutionList)
    help_text: str = ""

    @staticmethod
    def _resolve_executable(executable, name, config_file_location):
        """
        :returns: The resolved path to the executable

        :raises: ConfigValidationError if the executable cannot be found
            or we don't have permissions to execute.
        """
        # PS: This operation has surprising behavior but is kept this way for
        # backwards compatability
        if not os.path.isabs(executable):
            path_to_executable = os.path.abspath(
                os.path.join(config_file_location, executable)
            )
        else:
            path_to_executable = executable

        resolved = None
        if os.path.exists(path_to_executable):
            resolved = path_to_executable
        elif not os.path.isabs(executable):
            # look for system installed executable
            resolved = shutil.which(executable)

        if resolved is None:
            raise ConfigValidationError(
                config_file=config_file_location,
                errors=[f"Could not find executable {executable!r} for job {name!r}"],
            )

        if not os.access(resolved, os.X_OK):  # is not executable
            raise ConfigValidationError(
                config_file=config_file_location,
                errors=[
                    f"ExtJob {name!r} with executable"
                    f" {resolved!r} does not have execute permissions"
                ],
            )

        if os.path.isdir(resolved):
            raise ConfigValidationError(
                config_file=config_file_location,
                errors=[
                    f"ExtJob {name!r} has executable set to directory {resolved!r}"
                ],
            )

        return resolved

    _int_keywords = ["MAX_RUNNING", "MAX_RUNNING_MINUTES", "MIN_ARG", "MAX_ARG"]
    _str_keywords = [
        "STDIN",
        "STDOUT",
        "STDERR",
        "START_FILE",
        "TARGET_FILE",
        "ERROR_FILE",
        "START_FILE",
    ]

    @classmethod
    def _parse_config_file(cls, config_file: str):
        parser = ConfigParser()
        for int_key in cls._int_keywords:
            parser.add(int_key, value_type=ContentTypeEnum.CONFIG_INT).set_argc_minmax(
                1, 1
            )
        for path_key in cls._str_keywords:
            parser.add(path_key).set_argc_minmax(1, 1)

        parser.add("EXECUTABLE", required=True).set_argc_minmax(1, 1)
        parser.add("ENV").set_argc_minmax(1, 2)
        parser.add("EXEC_ENV").set_argc_minmax(1, 2)
        parser.add("DEFAULT").set_argc_minmax(2, 2)
        parser.add("ARGLIST").set_argc_minmax(1, -1)
        arg_type_schema = parser.add("ARG_TYPE")
        arg_type_schema.set_argc_minmax(2, 2)
        arg_type_schema.iset_type(0, ContentTypeEnum.CONFIG_INT)

        return parser.parse(
            config_file,
        )

    @classmethod
    def _read_str_keywords(cls, config_content):
        result = {}

        def might_set_value_none(keyword, key):
            value = config_content.getValue(keyword)
            if value == "null":
                value = None
            result[key] = value

        for key in cls._str_keywords:
            if config_content.hasKey(key):
                if key in ("STDIN", "STDOUT", "STDERR"):
                    might_set_value_none(key, key.lower() + "_file")
                else:
                    might_set_value_none(key, key.lower())
        return result

    @classmethod
    def _read_int_keywords(cls, config_content):
        result = {}
        for key in cls._int_keywords:
            if config_content.hasKey(key):
                value = config_content.getValue(key)
                if value > 0:
                    # less than or equal to 0 in the config is equivalent to
                    # setting None (backwards compatability)
                    result[key.lower()] = value
        return result

    @staticmethod
    def _make_arg_types_list(
        config_content, max_arg: Optional[int]
    ) -> List[ContentTypeEnum]:
        arg_types_dict = defaultdict(lambda: ContentTypeEnum.CONFIG_STRING)
        if max_arg is not None:
            arg_types_dict[max_arg - 1] = ContentTypeEnum.CONFIG_STRING
        for arg in config_content["ARG_TYPE"]:
            arg_types_dict[arg[0]] = ContentTypeEnum(type_from_kw(arg[1]))
        if arg_types_dict:
            return [
                arg_types_dict[j]
                for j in range(max(i for i in arg_types_dict.keys()) + 1)
            ]
        else:
            return []

    @classmethod
    def from_config_file(cls, config_file: str, name: Optional[str] = None):
        if name is None:
            name = os.path.basename(config_file)
        try:
            config_content = cls._parse_config_file(config_file)
        except ConfigValidationError as conf_err:
            err_msg = "Item:EXECUTABLE must be set - parsing"
            err_index = next(
                (i_err for i_err, err in enumerate(conf_err.errors) if err_msg in err),
                None,
            )
            if err_index is None:
                raise conf_err from None
            with open(config_file, encoding="utf-8") as f:
                if "PORTABLE_EXE " in f.read():
                    conf_err.errors[err_index] = conf_err.errors[err_index].replace(
                        err_msg,
                        '"PORTABLE_EXE" key is deprecated, '
                        'please replace with "EXECUTABLE" in',
                    )
            raise ConfigValidationError(conf_err.errors, conf_err.config_file) from None
        except IOError as err:
            raise ConfigValidationError(
                config_file=config_file,
                errors=f"Could not open job config file {config_file!r}",
            ) from err

        logger.info(
            "Content of job config %s: %s",
            name,
            Path(config_file).read_text(encoding="utf-8"),
        )
        content_dict = {}

        content_dict.update(**cls._read_str_keywords(config_content))
        content_dict.update(**cls._read_int_keywords(config_content))

        content_dict["executable"] = config_content.getValue("EXECUTABLE")
        if config_content.hasKey("ARGLIST"):
            # We unescape backslash here to keep backwards compatability ie. If
            # the arglist contains a '\n' we interpret it as a newline.
            content_dict["arglist"] = [
                s.encode("utf-8", "backslashreplace").decode("unicode_escape")
                for s in config_content["ARGLIST"][-1]
            ]

        content_dict["arg_types"] = cls._make_arg_types_list(
            config_content,
            content_dict["max_arg"]
            if "max_arg" in content_dict and content_dict["max_arg"] > 0
            else None,
        )

        def set_env(key, keyword):
            content_dict[key] = {}
            if config_content.hasKey(keyword):
                for env in config_content[keyword]:
                    if len(env) > 1:
                        content_dict[key][env[0]] = env[1]
                    else:
                        content_dict[key][env[0]] = None

        set_env("environment", "ENV")
        set_env("exec_env", "EXEC_ENV")

        content_dict["default_mapping"] = {}
        if config_content.hasKey("DEFAULT"):
            for key, value in config_content["DEFAULT"]:
                content_dict["default_mapping"][key] = value

        content_dict["executable"] = ExtJob._resolve_executable(
            content_dict["executable"], name, os.path.dirname(config_file)
        )

        # The default for stdout_file and stdin_file is
        # {name}.std{out/err}
        for handle in ("stdout", "stderr"):
            if handle + "_file" not in content_dict:
                content_dict[handle + "_file"] = name + "." + handle

        return cls(
            name,
            **content_dict,
        )

    def validate_args(self, context: SubstitutionList) -> None:
        self._validate_all_passed_private_args_have_an_effect()
        self._validate_all_magic_string_in_arglist_get_resolved(context)

    def _validate_all_passed_private_args_have_an_effect(self) -> None:
        """raises InvalidArgsException if validation fails"""
        # private args are always applied first, so we can grab the keys of the private
        # args list, and for every key, check if it matches on a substring of any
        # argument from the argument list
        relevant_private_args_keys = [
            key
            for key in self.private_args.keys()
            if key not in _SUBSTITUTED_AT_EXECUTION_TIME and key not in self.exec_env
        ]
        unused_private_args_keys = list(
            filter(
                lambda private_arg_key: all(
                    private_arg_key not in arg for arg in self.arglist
                ),
                relevant_private_args_keys,
            )
        )
        if unused_private_args_keys:
            unused_private_args_representation = [
                f"{key}={self.private_args[key]}" for key in unused_private_args_keys
            ]
            raise ExtJobInvalidArgsException(
                f"following arguments to job {self.name!r} were not found in the"
                f" argument list: {','.join(unused_private_args_representation)}"
            )

    def _validate_all_magic_string_in_arglist_get_resolved(
        self, context: SubstitutionList
    ) -> None:
        """raises InvalidArgsException if validation fails"""
        args_substituted_with_private_list = [
            (arg, self.private_args.substitute(arg, "", 1)) for arg in self.arglist
        ]
        args_substituted_with_context_and_private_list = [
            (orig_arg, context.substitute(modified_arg))
            for orig_arg, modified_arg in args_substituted_with_private_list
        ]
        defaulted_and_substituted_args = [
            (orig_arg, self.default_mapping.get(modified_arg, modified_arg))
            for orig_arg, modified_arg in args_substituted_with_context_and_private_list
        ]

        def arg_has_unresolved_substring(arg_tuple: Tuple[str, str]) -> bool:
            _, arg = arg_tuple
            unresolved_substrings = re.findall(r"<.*?>", arg)
            relevant_unresolved_substrings = list(
                filter(
                    lambda substr: substr not in _SUBSTITUTED_AT_EXECUTION_TIME,
                    unresolved_substrings,
                )
            )
            return bool(relevant_unresolved_substrings)

        args_with_unresolved_substrings = list(
            filter(
                arg_has_unresolved_substring,
                defaulted_and_substituted_args,
            )
        )
        if args_with_unresolved_substrings:
            unresolved_args_representation = [
                repr(orig_arg) for orig_arg, _ in args_with_unresolved_substrings
            ]
            raise ExtJobInvalidArgsException(
                f"Job {self.name!r} has unresolved arguments after "
                "applying argument substitutions: "
                f"{', '.join(unresolved_args_representation)}"
            )
