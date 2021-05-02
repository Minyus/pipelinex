import logging
from time import sleep

from pipelinex import log_time

logging.basicConfig(level=logging.DEBUG)


@log_time
def sleeping_identity(inp):
    sleep(0.1)
    return inp


sleeping_identity(1)
