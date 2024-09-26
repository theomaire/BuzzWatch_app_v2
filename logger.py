import logging

class MultiLogger:
    def __init__(self, log_func, log_file_path=None):
        self.log_func = log_func
        self.logger = logging.getLogger("multi_logger")
        self.logger.setLevel(logging.DEBUG)

        # Clear existing handlers to avoid duplicate logs
        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        # If a valid log_file_path is provided, add a FileHandler
        if log_file_path:
            try:
                self.file_handler = logging.FileHandler(log_file_path)
                formatter = logging.Formatter('%(asctime)s - %(message)s')
                self.file_handler.setFormatter(formatter)
                self.logger.addHandler(self.file_handler)
            except Exception as e:
                self.log_func(f"Error setting up file logging: {e}")

    def write(self, message):
        if message.strip():
            # Log to the provided log function
            self.log_func(message)
            # Log to file only if a file handler was successfully added
            if self.logger.hasHandlers():
                self.logger.debug(message)

    def flush(self):
        if self.logger.hasHandlers():
            for handler in self.logger.handlers:
                handler.flush()

    def progress(self, current, total, bar_length=20):
        fraction = current / total
        arrow = int(fraction * bar_length - 1) * '-' + '>'
        padding = int(bar_length - len(arrow)) * ' '
        ending = '\n' if current == total else '\r'
        message = f'Progress: [{arrow}{padding}] {int(fraction*100)}% {ending}'
        self.log_func(message.strip(), end='')  # Avoid adding a newline
        print(message, end='')  # Print to stdout to also handle in console

class LogFunctionStream:
    def __init__(self, log_func):
        self.log_func = log_func

    def write(self, message):
        if message.strip():
            self.log_func(message)  # Write to the UI logging function

    def flush(self):
        pass  # No need to flush the UI log function
