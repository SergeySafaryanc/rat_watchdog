import datetime
import glob
import os
import shutil

from loguru import logger
from watchdog.conrtollers.QtWatchdogController import QtWatchdogController


def clear_logs():
    if not os.path.isfile(path="pretrained_model_LDA-ORS.mdl"):
        return
    main_dir = f"results/{str(datetime.datetime.today().strftime('R_%b_%d_%Y_%H_%M_%S'))}"
    mdl_dir = f"{main_dir}/mdl"
    logs_dir = f"{main_dir}/logs"
    output_dir = f"{main_dir}/output"
    os.mkdir(main_dir)
    os.mkdir(mdl_dir)
    os.mkdir(logs_dir)
    os.mkdir(output_dir)
    for mdl in glob.glob("*.mdl"):
        shutil.move(mdl, mdl_dir)
    for log in glob.glob("logs/*"):
        shutil.move(log, logs_dir)
    for file in glob.glob("data/out/*"):
        shutil.move(file, output_dir)
    logger.info("logs clear - COMPLETE")


if __name__ == "__main__":
    #clear_logs()
    controller = QtWatchdogController()
