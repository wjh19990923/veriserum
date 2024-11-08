"""
edit the config.ini and runs as python -m dupla_sical.run_siCal
make sure to set the PYTHONPATH to include the other modules
"""
from configparser import ConfigParser
import git
from pathlib import Path
import logging

from dupla_tools.database_tools import DuplaDatabaseConnector
from dupla_tools.image_tools import load_img_by_path
from dupla_discal import GridCalibration_HoughROI_PassThrough_PointExtractor

if __name__ == "__main__":
    logging.basicConfig(filename='run_disCals.log', encoding='utf-8', level=logging.INFO, filemode="w")
    print("Loading config and connecting to database...")
    config = ConfigParser()
    config.read(Path(__file__).parent / "config.ini")
    dbCredentials = {
        "user": config["database"]["username"],
        "password": config["database"]["password"],
        "host": config["database"]["host"],
        "database": config["database"]["database_name"],
        "data_anchor": config["database"]["data_anchor"],
    }
    ddc = DuplaDatabaseConnector(**dbCredentials)

    # gather git information
    repo = git.Repo(Path(__file__).parent)
    sha = repo.head.object.hexsha
    remoteRepo = list(repo.remote().urls)[0]
    c = config["run_disCal"]
    debug = True if c["debug"] == "True" else False
    calSetIdx = int(c["calibrationSet"])
    disCalOperator = int(c["disCalOperator"])
    plane_results = {}
    for plane in ["bs", "fs"]:
        print(f"Working on {plane}")
        query = """SELECT ct.id, cm.id, ct.path 
            FROM calibration_trials as ct 
            JOIN calibration_measurements as cm ON cm.calibration_trials_id = ct.id 
            WHERE ct.phantom=%s AND ct.target=%s AND magnification=%s
            """
        res = ddc.execute_fetchall(query, ("sipla_grid", plane, "m0"))
        #calibrationtrial_id
        assert len(res) >= 1, "Expected only one result"
        ctId, cmId, ctPath = res[0]
        imgPath = ddc.cal_measurement_path_from_cal_measurement_id(cmId)
        print(f"Loading calibration measurement {cmId}")
        img = load_img_by_path(imgPath[plane])
        gc = GridCalibration_HoughROI_PassThrough_PointExtractor(polynomialN=3, regularisation=False)
        gc.low_threshold = 400
        gc.high_threshold = 800
        gc.debug = debug
        print(f"Fitting {gc}...")
        gc.fit(img)
        ps = gc.pixel_sizes
        plane_results[plane] = {"result":gc.write_to_string(), "psh":ps["h"], "psv":ps["v"], "trialId":ctId}
    print(f"Inserting result into database: {plane_results}")
    ddc.insert_distortion_calibration(calibration_trial_id_fs=plane_results["fs"]["trialId"], 
                                        calibration_trial_id_bs=plane_results["bs"]["trialId"],
                                        result_fs=plane_results["fs"]["result"],
                                        result_bs=plane_results["bs"]["result"],
                                        method="whatever",
                                        gitHash=sha,
                                        gitRepo=list(repo.remote().urls)[0],
                                        pixelSizeH_fs=plane_results["fs"]["psh"],
                                        pixelSizeV_fs=plane_results["fs"]["psv"],
                                        pixelSizeH_bs=plane_results["bs"]["psh"],
                                        pixelSizeV_bs=plane_results["bs"]["psv"],
                                        )