import logging
import os

name = 'dbg'

logger = logging.getLogger()
filename = os.path.join(name, 'log.%s' % name)
logging.basicConfig(handlers=[logging.FileHandler(filename=filename, mode='w'),
                              logging.StreamHandler()],
                    level=logging.INFO, format='%(message)s', )
print(logger.handlers[0].baseFilename)
# print(filename)
# logger.info('aaaaaaaaa')
