from pipelinex import log_time
from time import sleep
import logging

logging.basicConfig(level=logging.DEBUG)


@log_time
def sleeping_identity(inp):
    sleep(0.1)
    return inp


sleeping_identity(1)
