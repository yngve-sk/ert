from collections import defaultdict
from textwrap import indent
from typing import List, Optional, Tuple, Union

from ert.parsing.lark_parser_error_info import ErrorInfo


class ConfigWarning(UserWarning):
    pass


class ConfigValidationError(ValueError):
    def __init__(
        self,
        errors: Union[str, List[Union[ErrorInfo, str, Tuple[Optional[str], str]]]],
        config_file: Optional[str] = None,
    ) -> None:
        self.errors: List[ErrorInfo] = []
        if isinstance(errors, list):
            for err in errors:
                if isinstance(err, ErrorInfo):
                    self.errors.append(err)
                elif isinstance(err, str):
                    self.errors.append(ErrorInfo(message=err, filename=config_file))
                else:
                    filename, message = err
                    self.errors.append(ErrorInfo(message=message, filename=filename))
        else:
            self.errors.append(ErrorInfo(message=errors, filename=config_file))
        super().__init__(
            ";".join([self._get_error_message(error) for error in self.errors])
        )

    @classmethod
    def from_info(cls, info: ErrorInfo) -> "ConfigValidationError":
        return cls([info])

    @classmethod
    def _get_error_message(cls, info: ErrorInfo) -> str:
        return (
            (
                f"Parsing config file `{info.filename}` "
                f"resulted in the errors: {info.message}"
            )
            if info.filename is not None
            else info.message
        )

    def get_error_messages(self) -> List[str]:
        return [self._get_error_message(error) for error in self.errors]

    def get_cli_message(self) -> str:
        by_filename = defaultdict(list)
        for error in self.errors:
            by_filename[error.filename].append(error)

        return ";".join(
            [
                f"Parsing config file `{filename}` resulted in the errors: \n"
                + indent(
                    "\n".join([err.message_with_location for err in info_list]), "  * "
                )
                for filename, info_list in by_filename.items()
            ]
        )

    @classmethod
    def from_collected(
        cls, errors: List[Union[ErrorInfo, "ConfigValidationError"]]
    ) -> "ConfigValidationError":
        # Turn into list of only ConfigValidationErrors
        as_errors_only: List[ConfigValidationError] = [
            (ConfigValidationError([e]) if isinstance(e, ErrorInfo) else e)
            for e in errors
        ]

        return cls(
            [error_info for error in as_errors_only for error_info in error.errors]
        )
