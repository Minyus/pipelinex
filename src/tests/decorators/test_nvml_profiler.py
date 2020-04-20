from pipelinex import nvml_profile
from time import sleep
import logging

logging.basicConfig(level=logging.DEBUG)


@nvml_profile
def sleeping_identity(inp):
    sleep(0.1)
    return inp


sleeping_identity(1)
