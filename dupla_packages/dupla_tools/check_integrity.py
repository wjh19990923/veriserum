"""Runs the following integrity checks on the database whether
- each measurement in the database has a corresponding .tif file on the server
- whether `image_tools.load_img_by_meas_id` throws an assertion error (which are quality checks)

Run from parent directory as
python -m dupla_database.check_integrity integrity.log --anchor "/home/brknkybrd/eth_lmb"

Any errors are written to the output-logfile
"""
import argparse
import logging
from pathlib import Path, PureWindowsPath

from .database_tools import DuplaDatabaseConnector
from .image_tools import load_img_by_meas_id

if __name__ == "__main__":
    raise DeprecationWarning("not up to date")
    parser = argparse.ArgumentParser(
        description="Run integrity checks using the database configuration in config.py. Make sure that your normal user account has access to the relevant image directories on the server."
    )
    parser.add_argument("logfile", type=Path, help="set the path of the output logfile")
    parser.add_argument(
        "--anchor",
        type=Path,
        default=None,
        help="replace the anchor of each path (e.g. P://)",
    )
    args = parser.parse_args()

    logging.basicConfig(filename=args.logfile, level=logging.INFO)
    """new version:
        config = ConfigParser()
    config.read("config.ini")
    dbcon = DuplaDatabaseConnector(
        config["database"]["host"],
        config["database"]["database_name"],
        config["database"]["username"],
        config["database"]["password"],
        config["database"]["data_anchor"],
    )"""
    ddc = DuplaDatabaseConnector(**config)
    sql = """SELECT measurements.id, studies.data_path, trials.path FROM measurements 
             LEFT JOIN trials ON trials.id=measurements.trials_id 
             LEFT JOIN studies ON trials.study_id = studies.id
             """
    ddc.cursor.execute(sql)
    for measId, trialsPath, studiesPath in ddc.cursor:
        for plane in ["fs", "bs"]:
            p = PureWindowsPath(studiesPath) / PureWindowsPath(trialsPath)
            if args.anchor is not None:
                p = args.anchor / p.relative_to(p.anchor)
            try:
                load_img_by_meas_id(p, measId, plane=plane)
            except Exception:
                logging.exception(f"{measId},{plane}: ")
