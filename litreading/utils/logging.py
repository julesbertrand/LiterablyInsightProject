from loguru import logger


class StreamToLogger:
    def __init__(self, level="INFO"):
        self._level = level

    def write(self, buffer):
        stripped_buffer = buffer.rstrip()
        if len(stripped_buffer) > 0:
            logger.opt(depth=1).log(self._level, stripped_buffer)

    def flush(self):
        pass
