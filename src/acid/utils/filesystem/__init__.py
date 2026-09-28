import logging


# ---- Setting built-in logging
logging.getLogger(__name__).addHandler(logging.NullHandler())
