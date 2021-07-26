import contextlib

import loguru

from litreading.config import SKLEARN_LOGLEVEL


class RedirectStdoutToLogger:
    def __init__(self, logger: "loguru.Logger") -> None:
        stream = StreamToLogger(logger, level=SKLEARN_LOGLEVEL)
        self.context = contextlib.redirect_stdout(stream)

    def __enter__(self):
        self.context.__enter__()

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.context.__exit__(exc_type, exc_value, exc_tb)


class StreamToLogger:
    def __init__(self, logger: "loguru.Logger", level: "loguru.Level" = "INFO"):
        self._level = level
        self.logger = logger

    def write(self, buffer):
        stripped_buffer = buffer.rstrip()
        if len(stripped_buffer) > 0:
            self.logger.opt(depth=1).log(self._level, stripped_buffer)

    def flush(self):
        pass
