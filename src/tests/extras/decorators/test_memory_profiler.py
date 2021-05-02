import logging
from time import sleep

from pipelinex import mem_profile

logging.basicConfig(level=logging.DEBUG)


@mem_profile
def sleeping_identity(inp):
    sleep(0.1)
    return inp


sleeping_identity(1)
