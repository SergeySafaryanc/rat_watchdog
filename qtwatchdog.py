from watchdog.conrtollers.QtWatchdogController import QtWatchdogController
from watchdog.utils.readme import Singleton, write

if __name__ == "__main__":
    try:
        controller = QtWatchdogController()
    except KeyboardInterrupt:
        pass
    finally:
        write(Singleton.get('Predict'))
