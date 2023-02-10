import SimpleITK as sitk
import logging


class Logger(sitk.LoggerBase):
    """
    Adapts SimpleITK messages to be handled by a Python Logger object.

    Allows using the logging module to control the handling of messages coming
    from ITK and SimpleTK. Messages such as debug and warnings are handled by
    objects derived from sitk.LoggerBase.

    The LoggerBase.SetAsGlobalITKLogger method must be called to enable
    SimpleITK messages to use the logger.

    The Python logger module adds a second layer of control for the logging
    level in addition to the controls already in SimpleITK.

    The "Debug" property of a SimpleITK object must be enabled (if
    available) and the support from the Python "logging flow" hierarchy
    to handle debug messages from a SimpleITK object.

    Warning messages from SimpleITK are globally disabled with
    ProcessObject:GlobalWarningDisplayOff.

    """

    def __init__(self, logger: logging.Logger = logging.getLogger("SimpleITK")):
        """
        Initializes with a Logger object to handle the messages emitted from
        SimpleITK/ITK.
        """
        super(Logger, self).__init__()
        self._logger = logger

    @property
    def logger(self):
        return self._logger

    @logger.setter
    def logger(self, logger):
        self._logger = logger

    def __enter__(self):
        self._old_logger = self.SetAsGlobalITKLogger()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._old_logger.SetAsGlobalITKLogger()
        del self._old_logger

    def DisplayText(self, s):
        # Remove newline endings from SimpleITK/ITK messages since the Python
        # logger adds during output.
        self._logger.info(s.rstrip())

    def DisplayErrorText(self, s):
        self._logger.error(s.rstrip())

    def DisplayWarningText(self, s):
        self._logger.warning(s.rstrip())

    def DisplayGenericOutputText(self, s):
        self._logger.info(s.rstrip())

    def DisplayDebugText(self, s):
        self._logger.debug(s.rstrip())
