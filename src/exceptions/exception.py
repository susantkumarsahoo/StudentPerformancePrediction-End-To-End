import sys
import traceback

class CustomException(Exception):
    """
    Custom Exception Class for ML Pipeline
    """

    def __init__(self, message: str, error_detail: sys):
        super().__init__(message)
        self.error_message = CustomException.get_detailed_error_message(message, error_detail)

    @staticmethod
    def get_detailed_error_message(message: str, error_detail: sys) -> str:
        _, _, exc_tb = error_detail.exc_info()
        if exc_tb is not None:
            file_name = exc_tb.tb_frame.f_code.co_filename
            line_number = exc_tb.tb_lineno
        else:
            file_name = "Unknown File"
            line_number = "Unknown Line"

        detailed_message = (
            f"\nError Occurred:\n"
            f"Message: {message}\n"
            f"File: {file_name}\n"
            f"Line: {line_number}\n"
            f"Traceback: {''.join(traceback.format_exc())}"
        )
        return detailed_message

    def __str__(self):
        return self.error_message



