"""A library of common operations for using the LMB MySQL database using mysql.connector
all paths are saved as PureWindowsPaths
all strings are converted to lower case
just put the birthday of a phantom to 1900.01-01 00:00:00

..note:: should rework the functions to take dictionaries and create the 
query strings from that. something like
already done a little bit for the insert functions

fields = {"col1":"val1"}
squeries = somethings(fields.keys)
execute(squeries, fields.vals)
"""

from configparser import ConfigParser
import datetime
import mysql.connector as conn
from mysql.connector import errorcode
from pathlib import Path, PureWindowsPath
import sqlite3
import sys

from .errors import DataExistsError, DataNotFoundError, DBConsistencyError
from .image_tools import image_name_from_idx
from .anatomy_tools import anatomy_name_from_idx


class DuplaDatabaseConnector:
    def __init__(self, host: str, database: str, user, password: str, data_anchor: Path):
        """Handle connection to database and provide data inserts or reads
        Args:
          host, user, password, database: DB connection information
          data_anchor: the anchor of the data server (e.g. P:/ or mount point)
        """
        self.host = host
        self.user = user
        self.password = password
        self.database = database
        self.cnx = self._connect()
        self.cursor = self.cnx.cursor(buffered=True)
        # replace the anchor of each path that we concatenate
        self.data_anchor = Path(data_anchor)
        # what character to use as placeholder in queries
        self.placeholder = "%s"

    def _connect(self):
        """connect to the database with the loaded credentials"""
        try:
            cnx = conn.connect(
                user=self.user,
                password=self.password,
                host=self.host,
                database=self.database,
            )
        except conn.Error as err:
            if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
                print(
                    "Something is wrong with your user name or password. "
                    "Please check your configuration file."
                )
            elif err.errno == errorcode.ER_BAD_DB_ERROR:
                print("Database {} does not exist".format(self.database))
            elif err.errno == errorcode.CR_CONN_HOST_ERROR:
                print("Connection to host failed. Please check that you are in the HEST2 VPN.")
            else:
                print(err)
            sys.exit()
        return cnx

    def _disconnect(self):
        """close cursor and connection"""
        try:
            self.cursor.close()
            self.cnx.close()
        except (AttributeError, ReferenceError):
            pass

    def __del__(self):
        """call disconnect() to close cursor and connection"""
        self._disconnect()

    def path_and_anchor(self, p: PureWindowsPath) -> Path:
        """add path's anchor to self.data_anchor
        Note that paths in the DB are always WindowsPaths
        Returns:
          a normal Path though, so it also works on Linux"""
        p = PureWindowsPath(p)
        trunk = p.relative_to(p.anchor)
        return self.data_anchor / trunk

    def get_next_id(self, table: str, id_col: str = "id"):
        """Return the max_id + 1
        note that this does not necessarily correspond to the id that would be
        returned by an auto-increment insert, e.g. when the highest values were deleted

        Args:
            table - tablename
            id_col - the column in which to find the max
        Returns:
            1 if no entry exists, otherwise max(id) + 1
        """
        query = f"""SELECT MAX({id_col}) FROM {table}"""
        self.cursor.execute(query)
        res = self.cursor.fetchone()[0]
        if res is None:
            return 1
        else:
            return int(res) + 1

    def execute_fetchall(self, query: str, args: tuple = None):
        """execute a query and return/fetchall the results"""
        if args is None:
            self.cursor.execute(query)
        else:
            self.cursor.execute(query, args)
        res = self.cursor.fetchall()
        return res

    def executemany_fetchall(self, query: str, args: tuple):
        self.cursor.executemany(query, args)
        res = self.cursor.fetchall()
        return res

    def execute_commit(self, query: str, args: tuple = None, commit=True):
        """
        execute one or multiple queries and commit the changes. used when we want to wirte something into database.
        if you want to run multiple queries, the queries must be seperated in string by `;` character.
        this function makes sure to apply the changes if and only if all the queries execute seccessfully.
        if even one query doesn't execute successfully, no changes will be applied into the database.
        Args:
            query: the sql query with self.placeholder as  placeholders
            args: tuple of python args to fill the placeholders in query
            commit: if False, only execute, dont commit
        """
        for query in query.split(";"):
            if query:
                if args is None:
                    self.cursor.execute(query)
                else:
                    self.cursor.execute(query, args)
        # commit after execution of all queries, not after one execution.
        if commit:
            self.cnx.commit()

    """//////////////////////////////////////////
    Reading functions
    //////////////////////////////////////////"""

    def subject_path_from_idx(self, idx: int) -> Path:
        """return `subjects.path` by querying `subjects.id`"""
        query = f"""SELECT path FROM subjects WHERE id={self.placeholder}"""
        res = self.execute_fetchall(query, (idx,))
        if len(res) == 0:
            raise DataNotFoundError(f"No study with id {idx} was found.")
        return Path(res[0][0])

    def measurement_path_from_measurement_id(self, idx: int) -> dict:
        """return concatenated fs and bs image paths for measurements Path
        given by data_anchor / trials.path / name derived from measurements.id
        Args:
             id - the measurement id
        Returns:
             dict - {'bs': img_bs_path, 'fs': img_fs_path}
        Exceptions:
             DataNotFoundError - if no entry is found
        """
        query = f"""SELECT trials.path, measurements.id
                   FROM measurements
                   INNER JOIN
                   trials ON measurements.trials_id = trials.id
                   WHERE measurements.id = {self.placeholder}
                 """
        res = self.execute_fetchall(query, (idx,))
        if len(res) == 0:
            raise DataNotFoundError(f"No measurement with measurement_id {idx} was found.")
        else:
            res = res[0]
            measurement_dict = {
                "bs": self.path_and_anchor(Path(res[0]) / image_name_from_idx(res[1], "bs", cal=False)),
                "fs": self.path_and_anchor(Path(res[0]) / image_name_from_idx(res[1], "fs", cal=False)),
            }
        return measurement_dict

    def cal_measurement_path_from_cal_measurement_id(self, idx: int) -> dict:
        """return concatenated fs and bs image paths for measurements Path
        given by data_anchor / calibration_trials.path / name derived from calibration_measurements.id
        Args:
             id - the calibration_measurement.id
        Returns:
             dict - {'bs': img_bs_path, 'fs': img_fs_path}
        Exceptions:
             DataNotFoundError - if no entry is found
        """
        query = f"""SELECT calibration_trials.path, cm.id
                   FROM calibration_measurements as cm
                   INNER JOIN
                   calibration_trials ON cm.calibration_trials_id = calibration_trials.id
                   WHERE cm.id = {self.placeholder}
                 """
        res = self.execute_fetchall(query, (idx,))
        if len(res) == 0:
            raise DataNotFoundError(f"No measurement with measurement_id {idx} was found.")
        else:
            res = res[0]
            measurement_dict = {
                "bs": self.path_and_anchor(Path(res[0]) / image_name_from_idx(res[1], "bs", cal=True)),
                "fs": self.path_and_anchor(Path(res[0]) / image_name_from_idx(res[1], "fs", cal=True)),
            }
        return measurement_dict

    def anatomy_path_from_anatomy_id(self, idx: int) -> str:
        """returns a concatenated Path of data_anchor / anatomies_path
        for the anatomy with id
        Args:
             idx - the anatomy id
        Returns:
             dict with keys anatomies.id -> {"path":subjects.path / formatted_anatomies_id,
                "category": femur_implant or tibia_implant}
        Exceptions:
             DataNotFoundError - if no Entry is found
        .. note :: for filename, see `dupla_tools.anatomy_tools.anatomy_name_from_idx()`
        """
        query = f""" SELECT anatomies.id, subjects.path, anatomies.filetype, anatomies.category
                    FROM anatomies
                    INNER JOIN
                    subjects ON subjects.id = anatomies.subject_id
                    WHERE anatomies.id = {self.placeholder}
                 """
        res = self.execute_fetchall(query, (idx,))
        anatomy_dict = dict()
        if len(res) == 0:
            raise DataNotFoundError(f"No anatomy with id {idx} was found.")
        for r in res:
            anaIdx, dataPath, filetype, cat = r
            anatomy_dict[anaIdx] = {
                "path": self.path_and_anchor(
                    PureWindowsPath(dataPath) / PureWindowsPath(anatomy_name_from_idx(anaIdx, filetype))
                ),
                "category": str(cat),
            }
        return anatomy_dict

    def anatomy_path_from_measurement_id(self, idx: int) -> dict:
        """get all anatomies for measurement_id (through trials and subjects)
        Args:
             idx - the measurement id
        Returns:
             dict with keys anatomies.id -> {"path":subjects.path / formatted_anatomies_id,
                "category": femur_implant or tibia_implant}
        Exceptions:
             DataNotFoundError - if no Entry is found
        .. note :: for filename, see `dupla_tools.anatomy_tools.anatomy_name_from_idx()`
        """
        query = f""" SELECT anatomies.id, subjects.path, anatomies.filetype, anatomies.category
                    FROM measurements
                    INNER JOIN
                    trials ON measurements.trials_id = trials.id
                    INNER JOIN
                    anatomies ON trials.subject_id = anatomies.subject_id
                    INNER JOIN
                    subjects ON subjects.id = anatomies.subject_id
                    WHERE measurements.id = {self.placeholder}
                 """
        res = self.execute_fetchall(query, (idx,))
        anatomy_dict = dict()
        if len(res) == 0:
            raise DataNotFoundError(f"No anatomy with id {idx} was found.")
        for r in res:
            anaIdx, dataPath, filetype, cat = r
            anatomy_dict[anaIdx] = {
                "path": self.path_and_anchor(
                    PureWindowsPath(dataPath) / PureWindowsPath(anatomy_name_from_idx(anaIdx, filetype))
                ),
                "category": str(cat),
            }
        return anatomy_dict

    def measurement_from_pose_id(self, protocol_id: str, trials_id: str):
        query = f"""SELECT * FROM measurements WHERE protocol_id={self.placeholder} AND trials_id={self.placeholder}"""
        return self.execute_fetchall(query, (protocol_id.lower(), trials_id))

    def measurement_path_from_trial_id(self, idx: int) -> dict:
        """assemble bs and fs image paths for all measurements of a trial
        Args:
             idx - the trial id
        Returns:
             dict - {measurement_id => {'bs': path, 'fs': path}}
        Exceptions:
             DataNotFoundError - if no Entry is found
        """
        query = f""" SELECT measurements.id, trials.path
                    FROM trials
                    INNER JOIN
                    measurements ON trials.id = measurements.trials_id
                    WHERE trials.id = {self.placeholder}
                 """
        res = self.execute_fetchall(query, (idx,))
        measurement_dict = dict()
        if len(res) == 0:
            raise DataNotFoundError(f"No measurements with trial_id {idx} was found.")
        for r in res:
            idx, p = r[0]
            measurement_dict[int(idx)] = {
                "bs": self.path_and_anchor(PureWindowsPath(p) / image_name_from_idx(idx, "bs", cal=False)),
                "fs": self.path_and_anchor(PureWindowsPath(p) / image_name_from_idx(idx, "fs", cal=False))
            }
        return measurement_dict

    def source_int_calibration_from_id(self, idx):
        """returns SI information from `source_int_calibration`
        Returns:
            dictionary with field-name:value
        .. note:: values are not converted to any specific python type
        """
        fields = (
            "id",
            "calibration_trial_id_fs",
            "calibration_trial_id_bs",
            "sid_fs",
            "sid_bs",
            "principalp_h_fs",
            "principalp_v_fs",
            "principalp_h_bs",
            "principalp_v_bs",
        )
        query = f"""SELECT
            {", ".join(fields)}
            FROM source_int_calibration
            WHERE id={self.placeholder}
            """
        res = self.execute_fetchall(query, (idx,))
        if len(res) == 0:
            raise DataNotFoundError(f"No SI calibration with measurement_id {idx} was found.")
        elif len(res) > 1:
            raise DBConsistencyError(f"Multiple SI calibrations with id {idx} found.")
        else:
            return {field: res[0][n] for n, field in enumerate(fields)}

    def distortion_calibration_from_id(self, idx: int) -> dict:
        """returns distortion calibration by `distortion_calibration.id`
        Args:
             idx - the id
        Returns:
             dict - with keys
                (
                "idx",    
                "calibration_trial_id_fs",
                "calibration_trial_id_bs",
                "distortion_result_fs",
                "distortion_result_bs",
                "method"
                  )
        Exceptions:
             DataNotFoundError - if no Entry is found
             DBConsistencyError if multiples are found
        ..note:: In most cases the pixel sizes of the distortion calibration are irrelevant, one wants the info on the transformed
            image, which depends on the size and resolution passed to the DistortionCalibration.transform()
        """
        fields = (
            "id",
            "calibration_trial_id_fs",
            "calibration_trial_id_bs",
            "distortion_result_fs",
            "distortion_result_bs",
            "method",
        )
        query = f"""SELECT
            {", ".join(fields)}
            FROM distortion_calibration
            WHERE id={self.placeholder}
            """
        res = self.execute_fetchall(query, (idx,))
        if len(res) == 0:
            raise DataNotFoundError(f"No distortion calibration with id {idx} was found.")
        elif len(res) > 1:
            raise DBConsistencyError(
                f"Multiple distortion calibrations with id {idx} found."
                "This should never happen, please contact the DB administrator"
            )
        else:
            return {field: res[0][n] for n, field in enumerate(fields)}

    def calibration_path_from_calibration_trial_id(self, idx: int) -> dict:
        """assemble bs and fs image paths for the calibration measurement of a calibration trial
        Args:
             idx - the calibration trial id
        Returns:
             dict - {"calibration_measurement_id": calMeasIdx, 'bs': path, 'fs': path}}
        Exceptions:
             DataNotFoundError - if no Entry is found
        Note:
             images are always saved according to cal_fs_{6 digit measurement id}.tif
        """
        query = f""" SELECT cm.id, ct.path
                    FROM calibration_trials as ct
                    INNER JOIN
                    calibration_measurements as cm ON ct.id = cm.calibration_trials_id
                    INNER JOIN trials ON trials.calibration_set_id = ct.calibration_set_id
                    WHERE ct.id = {self.placeholder}
                 """
        res = self.execute_fetchall(query, (idx,))
        if len(res) == 0:
            raise DataNotFoundError(f"No calibration measurements with calibration_trial_id {idx} was found.")
        # we will get many results, but make sure all are identical
        assert len(set(res)) == 1, f"Received multiple non-identical results, shouldnt happen: {set(res)}"
        calMeasIdx, calTrialPath = res[0]
        return {
            "calibration_measurement_id": calMeasIdx,
            "bs": self.path_and_anchor(
                PureWindowsPath(calTrialPath)
                / PureWindowsPath(f"cal_bs_{calMeasIdx:0>6}.tif")
            ),
            "fs": self.path_and_anchor(
                PureWindowsPath(calTrialPath)
                / PureWindowsPath(f"cal_fs_{calMeasIdx:0>6}.tif")
            ),
        }

    def pose_info_from_measurement_id(self, idx: int) -> dict:
        """return pose information of all anatomies for the measurement idx
        ordered by poses.id
        Args:
             idx - the measurement id
        Returns:
             {anatomy_id: list[
                {tx:float, ty:float, tz:float,
                r0:float, r1:float, r2:float, r3:float,
                sic_id:int, dist_id:int, update_timestamp: datetime.datetime}]
             }
        Exceptions:
             DataNotFoundError - if no Entry is found
        """
        query = f""" SELECT
                    anatomy_id,
                    tx,
                    ty,
                    tz,
                    r0,
                    r1,
                    r2,
                    r3,
                    source_int_calibration_id,
                    distortion_calibration_id,
                    update_timestamp,
                    id,
                    obsolete
                    FROM poses
                    WHERE measurement_id = {self.placeholder}
                    ORDER BY update_timestamp ASC
                 """
        res = self.execute_fetchall(query, (idx,))
        pose_dict = dict()
        if len(res) == 0:
            raise DataNotFoundError("No pose data with measurement_id {} was found.".format(idx))
        else:
            for r in res:
                pose_dict[int(r[0])] = []
            for r in res:
                if r[12] == 1:
                    print(
                        f"Warning: Pose with measurement id {idx} for anatomy_id {r[0]} is marked as obsolete({r[12]}).")
                pose_dict[int(r[0])].append(
                    {
                        "tx": float(r[1]),
                        "ty": float(r[2]),
                        "tz": float(r[3]),
                        "r0": float(r[4]),
                        "r1": float(r[5]),
                        "r2": float(r[6]),
                        "r3": float(r[7]),
                        "sic_id": float(r[8]),
                        "dist_id": float(r[9]),
                        "update_timestamp": r[10],
                        "pose_id": r[11],
                    }
                )
        return pose_dict

    def midpoint_info_from_measurement_id(self):
        """
        gets midpoint calibration information from database.
        this information basically tells us where is the physical
        midpoint of the intensifier located in the xray image.

        not implemented yet; because database is not ready for this yet.
        """
        raise NotImplementedError

    def fs_ii_info_from_measurement_id(self):
        """
        gets position and orientation of the fs intensifier in respect to the bs intensifier.
        we need is fs_ii's physical midpoint, normal vector and vertical vector;
        all defined in bs_ii coordinate system.

        not implemented yet; because database is not ready for this yet.
        """
        raise NotImplementedError

    """///////////////////////////////////////
    INSERT FUNCTIONS
    ///////////////////////////////////////"""

    def convert_input_dict(self, fieldsDic: dict):
        """convert input dictionary values to types that mysql.connector can work with"""
        d = self.convert_paths_to_str(fieldsDic)
        d = self.convert_to_lower(d)
        return d

    @staticmethod
    def convert_to_lower(fieldsDic: dict):
        """creates a new dictionary with all str values converted to lowercase"""
        return {k: v.strip().lower() if type(v) == str else v for k, v in fieldsDic.items()}

    @staticmethod
    def convert_paths_to_str(fieldsDic: dict):
        return {k: str(v) if isinstance(v, Path) else v for k, v in fieldsDic.items()}

    def insert_fields_values(self, table: str, fieldsVals: dict, commit=True):
        """insert into a table, with dictionary keys giving the fields and the values the values
        Args:
            table: the table name
            fieldsVals: keys are going to be the field names in the table, filled with the vals
            commit: not only execute the transaction but also commit
        Example
            `insert_field_values("subjects", {"name":"John", "age":33})`
        .. note:: vals are converted to lower() if they are `str`, and to str() if they are `Path()`
        """
        fv = self.convert_input_dict(fieldsVals)
        # needs to convert to str, as join cannot work with e.g. int values
        query = f"""INSERT INTO {table} ({",".join(fv.keys())}) VALUES ({",".join([f"%({k})s" for k in fv.keys()])})"""
        self.cursor.execute(query, fv)
        lr = self.cursor.lastrowid
        if commit:
            self.cnx.commit()
        return lr

    def insert_study(self, name: str, commit=True, **kwargs) -> int:
        """use name to check whether study exists, otherwise insert
        Args:
            name: `studies.name`, used to check whether it already exists
            commit: see `execute_commit()`
            kwargs: rest of fieldname => fieldval
        Returns:
            id of inserted entry
        Raises:
            DataExistsError if one entry exists
            DBConsistencyError if multiple entries exist (really bad)
        """
        addFields = self.convert_input_dict(kwargs)
        addFields["name"] = name
        query = f"""SELECT id FROM studies WHERE name={self.placeholder}"""
        self.cursor.execute(query, (name,))
        res = self.cursor.fetchall()
        if len(res) == 0:
            self.insert_fields_values("studies", addFields, commit=commit)
            return self.cursor.lastrowid
        elif len(res) == 1:
            raise DataExistsError(res[0], "studies")
        else:
            raise DBConsistencyError(
                f"{len(res)} entries found for studies with name {name}."
                " This should never happen! Please check with your code and database maintainer."
            )

    def insert_operator(self, fname: str, lname: str, commit=True, **kwargs):
        """use firstname and lastname to check, otherwise insert
        Args:
            commit: see `execute_commit()`
        Returns:
            id of inserted entry
        Raises:
            DataExistsError if one entry exists
            DBConsistencyError if multiple entries exist (really bad)
        """
        addFields = self.convert_input_dict(kwargs)
        addFields["fname"] = fname
        addFields["lname"] = lname
        query = f"SELECT id from operators WHERE firstname={self.placeholder} AND lastname={self.placeholder}"
        res = self.execute_fetchall(query, (fname, lname))
        if len(res) == 0:
            self.insert_fields_values("operators", addFields, commit=commit)
            return self.cursor.lastrowid
        elif len(res) == 1:
            raise DataExistsError(res[0], "operators")
        else:
            raise DBConsistencyError(
                f"{len(res)} entries found for operator with name {fname}  {lname}."
                " This should never happen! Please check with your code and database maintainer."
            )

    def insert_subject(self, protocol_id: str, commit=True, **kwargs):
        """use protocol_id to check whether subject exists, otherwise insert
        Args:
            protocol_id: `subjects.protocol_id`
            commit: see `execute_commit()`
            kwargs: rest of the fieldname => fieldvalue
            ..note:: if birthdate is included, use `birthdate.strftime("%Y-%m-%d %H:%M:%S")`
            ..note:: dictionary values are converted to lower() if they are strings
        Returns:
            id of inserted entry
        Raises:
            DataExistsError if one entry exists
            DBConsistencyError if multiple entries exist (really bad)

        """
        addFields = self.convert_input_dict(kwargs)
        addFields["protocol_id"] = protocol_id.lower()
        query = f"""SELECT id FROM subjects WHERE protocol_id={self.placeholder}"""
        res = self.execute_fetchall(query, (protocol_id.lower(),))
        if len(res) == 0:
            self.insert_fields_values("subjects", addFields, commit=commit)
            return self.cursor.lastrowid
        elif len(res) == 1:
            raise DataExistsError(res[0], "subjects")
        else:
            raise DBConsistencyError(
                f"{len(res)} entries found for subjects with protocol_id {protocol_id}."
                " This should never happen! Please check with your code and database maintainer."
            )

    def insert_anatomy(self, subject_id: int, protocol_id: str, commit=True, **kwargs):
        """use protocol_id and subject to check whether exists, otherwise insert
        Args:
            subject_id: bound to `subjects.id`
            protocol_id: used to check whether it already exists
            commit: see `execute_commit()`
            kwargs: rest of the fieldname => fieldvalue
            ..note:: dictionary values are converted to lower() if they are strings
        Returns:
            id of inserted entry
        Raises:
            DataExistsError if one entry exists
            DBConsistencyError if multiple entries exist (really bad)
        """
        addFields = self.convert_input_dict(kwargs)
        addFields["subject_id"] = subject_id
        addFields["protocol_id"] = protocol_id.lower()
        query = f"""SELECT id FROM anatomies WHERE protocol_id={self.placeholder} and subject_id={self.placeholder}"""
        res = self.execute_fetchall(query, (protocol_id, subject_id))
        if len(res) == 0:
            self.insert_fields_values("anatomies", addFields, commit=commit)
            return self.cursor.lastrowid
        elif len(res) == 1:
            raise DataExistsError(res[0], "anatomies")
        else:
            raise DBConsistencyError(
                f"{len(res)} entries found for anatomy with subject {subject_id} "
                " and name {name}."
                " This should never happen! Please check with your code and database maintainer."
            )

    def insert_calibration_trial(
            self,
            phantom: str,
            target: str,
            kV_fs: int,
            kV_bs: int,
            mA_fs: int,
            mA_bs: int,
            exposure_time_fs: int,
            exposure_time_bs: int,
            sampling_freq_bs: int,
            sampling_freq_fs: int,
            path: Path,
            protocol_id: str,
            magnification: str = None,
            **kwargs,
    ):
        """use protocol_id to check whether exists, otherwise insert
        Args:
             phantom: for valid values, check current database specification
             target: "fs" or "bs" or "both", where was the phantom mounted
        Returns:
             id of inserted entry
        Raises:
             DataExistsError if one entry exists
             DBConsistencyError if multiple entries exist (really bad)
        """
        query = f"""SELECT id FROM calibration_trials WHERE protocol_id={self.placeholder}"""
        res = self.execute_fetchall(query, (protocol_id.lower(),))
        if len(res) == 0:
            query = """INSERT INTO calibration_trials (
                phantom,
                target,
                kV_fs,
                kV_bs,
                mA_fs,
                mA_bs,
                exposure_time_fs,
                exposure_time_bs,
                sampling_freq_bs,
                sampling_freq_fs,
                path,
                magnification,
                protocol_id
                ) VALUES (
                %(phantom)s,
                %(target)s,
                %(kV_fs)s,
                %(kV_bs)s,
                %(mA_fs)s,
                %(mA_bs)s,
                %(exposure_time_fs)s,
                %(exposure_time_bs)s,
                %(sampling_freq_bs)s,
                %(sampling_freq_fs)s,
                %(path)s,
                %(magnification)s,
                %(protocol_id)s)"""
            query_dict = {
                "phantom": phantom.lower(),
                "target": target.lower(),
                "kV_fs": kV_fs,
                "kV_bs": kV_bs,
                "mA_fs": mA_fs,
                "mA_bs": mA_bs,
                "exposure_time_fs": exposure_time_fs,
                "exposure_time_bs": exposure_time_bs,
                "sampling_freq_fs": sampling_freq_fs,
                "sampling_freq_bs": sampling_freq_bs,
                "path": path,
                "magnification": magnification.lower() if magnification is not None else "m0",
                "protocol_id": protocol_id.lower(),
            }
            self.execute_commit(query, query_dict)
            return self.cursor.lastrowid
        elif len(res) == 1:
            raise DataExistsError(res[0], "calibration_trials")
        else:
            raise DBConsistencyError(
                f"{len(res)} entries found for calibration trial with protocol_id {protocol_id}."
                " This should never happen! Please check with your code and database maintainer."
            )

    def insert_trial(
            self,
            protocol_id: str,
            study_id: int,
            subject_id: int,
            calibration_set_id: int,
            task: str,
            radiation_fs: float,
            radiation_bs: float,
            kV_fs: int,
            kV_bs: int,
            mA_fs: int,
            mA_bs: int,
            exposure_time_fs: int,
            exposure_time_bs: int,
            sampling_freq_bs: int,
            sampling_freq_fs: int,
            path: Path,
            vicon_path: Path,
            exp_sid: int,
            magnification: str = None,
            **kwargs,
    ):
        """use protocol_id + study_id to check whether exists, otherwise insert
        Returns:
             id of inserted entry
        Raises:
             DataExistsError if one entry exists
             DBConsistencyError if multiple entries exist (really bad)
        """
        query = f"""SELECT id FROM trials WHERE protocol_id={self.placeholder} AND study_id={self.placeholder}"""
        res = self.execute_fetchall(query, (protocol_id.lower(), study_id))
        if len(res) == 0:
            fields = [
                ("protocol_id", protocol_id.lower()),
                ("study_id", study_id),
                ("subject_id", subject_id),
                ("calibration_set_id", calibration_set_id),
                ("task", task),
                ("radiation_fs", radiation_fs),
                ("radiation_bs", radiation_bs),
                ("kV_fs", kV_fs),
                ("kV_bs", kV_bs),
                ("mA_fs", mA_fs),
                ("mA_bs", mA_bs),
                ("exposure_time_fs", exposure_time_fs),
                ("exposure_time_bs", exposure_time_bs),
                ("sampling_freq_bs", sampling_freq_bs),
                ("sampling_freq_fs", sampling_freq_fs),
                ("path", path),
                ("vicon_path", vicon_path),
                ("magnification", magnification.lower() if magnification is not None else "m0"),
                ("exp_sid", exp_sid),
            ]
            columns, values = zip(*fields)
            query = f"""INSERT INTO trials
                    ({",".join(columns)})
                    VALUES ({",".join(["{self.placeholder}"] * len(columns))})"""
            self.execute_commit(query, values)
            return self.cursor.lastrowid
        elif len(res) == 1:
            raise DataExistsError(res[0], "trials")
        else:
            raise DBConsistencyError(
                f"{len(res)} entries found for trial with protocol_id {protocol_id}."
                " This should never happen! Please check with your code and database maintainer."
            )

    def insert_calibration_measurement(
            self,
            protocol_id: str,
            calibration_trials_id: int,
            dt: datetime.datetime,
            calType: str = None,
            idx: int = None,
            commit: bool = False,
    ):
        """use protocol_id+calibration_trials_id to check whether exists, otherwise insert
        Args:
            idx: pass None if you want to use autoincrement, or pass a number.
              This will reset the autoincrement counter
            calibration_trials_id: links to `calibration_trials.id`
            dt: the datetime of the measurement
            calType: additional info for legacy tube. None or (bottom, top, left, right, center)
            commit: if False, only execute
        Raises:
             DataExistsError if one entry exists
             DBConsistencyError if multiple entries exist (really bad)
        """
        dt = dt.strftime("%Y-%m-%d %H:%M:%S")
        query = (
            f"""SELECT id FROM calibration_measurements WHERE protocol_id={self.placeholder} AND calibration_trials_id={self.placeholder}"""
        )
        res = self.execute_fetchall(query, (protocol_id.lower(), calibration_trials_id))
        if len(res) == 0:
            query = f"""INSERT INTO calibration_measurements
                       (id, protocol_id, calibration_trials_id, datetime, type)
                       VALUES
                       ({self.placeholder}, {self.placeholder}, {self.placeholder}, {self.placeholder}, {self.placeholder})"""
            t = None if calType is None else calType.lower()
            self.execute_commit(
                query,
                (idx, protocol_id.lower(), calibration_trials_id, dt, t),
                commit=commit,
            )
            return self.cursor.lastrowid
        elif len(res) == 1:
            raise DataExistsError(res[0], "measurements")
        else:
            raise DBConsistencyError(
                f"{len(res)} entries found for measurement with protocol_id {protocol_id} "
                " and trial_id {trial_id}."
                " This should never happen! Please check with your code and database maintainer."
            )

    def insert_measurement(
            self,
            protocol_id: str,
            trials_id: int,
            dt: datetime.datetime,
            idx: int = None,
            commit=False,
    ):
        """use protocol_id+trials_id to check whether exists, otherwise insert
        Args:
            idx: pass None if you want to use autoincrement, or pass a number.
              This will reset the autoincrement counter
            trials_id: links to `trials.id`
            dt: the datetime of the measurement
            commit: if False, only execute
        Raises:
             DataExistsError if one entry exists
             DBConsistencyError if multiple entries exist (really bad)
        """
        dt = dt.strftime("%Y-%m-%d %H:%M:%S")
        query = f"""SELECT id FROM measurements WHERE protocol_id={self.placeholder} AND trials_id={self.placeholder}"""
        res = self.execute_fetchall(query, (protocol_id.lower(), trials_id))
        if len(res) == 0:
            query = f"""INSERT INTO measurements
                       (id, protocol_id, trials_id, datetime)
                       VALUES
                       ({self.placeholder}, {self.placeholder}, {self.placeholder}, {self.placeholder})"""
            self.execute_commit(query, (idx, protocol_id.lower(), trials_id, dt), commit=commit)
            return self.cursor.lastrowid
        elif len(res) == 1:
            raise DataExistsError(res[0], "measurements")
        else:
            raise DBConsistencyError(
                f"{len(res)} entries found for measurement with protocol_id {protocol_id} "
                " and trial_id {trial_id}."
                " This should never happen! Please check with your code and database maintainer."
            )

    def insert_pose(
            self,
            measId: int,
            anatomyId: int,
            pose: dict,
            method: str,
            source_int_calibration_id: int,
            distortion_calibration_id: int,
            git_hash: str = None,
            git_repo: str = None,
    ):
        """
        Args:
          pose: dict with {tx, ty, tz, r0, r1, r2, r3}
          method: "MANMATCH" or "ROBOT"
        """
        # check whether anatomyId is consistent with the measurement->subject->anatomy
        ana_dicts = self.anatomy_path_from_measurement_id(measId)
        if anatomyId not in list(ana_dicts):
            raise ValueError(
                f"Given anatomy id {anatomyId} was inconsistent with the "
                f"given measurement id {measId} belonging to ids {ana_dicts}"
            )
        query = f"""INSERT INTO poses
            (measurement_id, anatomy_id, tx, ty, tz,
            r0, r1, r2, r3,
            method, git_hash, git_repo,
            source_int_calibration_id, distortion_calibration_id)
            VALUES
            ({self.placeholder}, {self.placeholder}, {self.placeholder}, {self.placeholder}, {self.placeholder},
            {self.placeholder}, {self.placeholder}, {self.placeholder}, {self.placeholder},
            {self.placeholder}, {self.placeholder}, {self.placeholder},
            {self.placeholder}, {self.placeholder})"""
        self.execute_commit(
            query,
            (
                measId,
                anatomyId,
                pose["tx"],
                pose["ty"],
                pose["tz"],
                pose["r0"],
                pose["r1"],
                pose["r2"],
                pose["r3"],
                method,
                git_hash,
                git_repo,
                source_int_calibration_id,
                distortion_calibration_id,
            ),
        )

    def insert_distortion_calibration(
            self,
            calibration_trial_id_fs: int,
            calibration_trial_id_bs: int,
            result_fs: str,
            result_bs: str,
            method: str,
            gitHash: str,
            gitRepo: str,
            pixelSizeH_fs: float,
            pixelSizeV_fs: float,
            pixelSizeH_bs: float,
            pixelSizeV_bs: float,
    ):
        """insert with no checks whether already exists"""
        dt = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        query = f"""INSERT INTO distortion_calibration
                (calibration_trial_id_fs, calibration_trial_id_bs, update_timestamp,
                obsolete, distortion_result_fs, distortion_result_bs,  method,
                git_hash, git_repo, pixel_size_h_fs, pixel_size_v_fs, pixel_size_h_bs,
                pixel_size_v_bs)
                VALUES
                ({self.placeholder}, {self.placeholder}, {self.placeholder}, {self.placeholder}, {self.placeholder}, {self.placeholder}, {self.placeholder}, {self.placeholder}, {self.placeholder}, {self.placeholder}, {self.placeholder}, {self.placeholder}, {self.placeholder})"""
        self.execute_commit(
            query,
            (
                calibration_trial_id_fs,
                calibration_trial_id_bs,
                dt,
                False,
                result_fs,
                result_bs,
                method,
                gitHash,
                gitRepo,
                pixelSizeH_fs,
                pixelSizeV_fs,
                pixelSizeH_bs,
                pixelSizeV_bs,
            ),
        )
        return self.cursor.lastrowid

    def insert_source_int_calibration(
            self,
            calibration_trial_id_fs: int,
            calibration_trial_id_bs: int,
            principalp_h_fs: float,
            principalp_v_fs: float,
            principalp_h_bs: float,
            principalp_v_bs: float,
            sid_fs: float,
            sid_bs: float,
            method: str,
            gitHash: str,
            gitRepo: str,
    ):
        """insert with no checks whether already exists"""
        dt = datetime.datetime.now().strftime(f"%Y-%m-%d %H:%M:%S")
        fields = ["calibration_trial_id_fs",
                  "calibration_trial_id_bs",
                  "update_timestamp",
                  "obsolete",
                  "method",
                  "git_hash",
                  "git_repo",
                  "sid_fs",
                  "sid_bs",
                  "principalp_h_fs",
                  "principalp_v_fs",
                  "principalp_h_bs",
                  "principalp_v_bs"]
        query = f"""INSERT INTO source_int_calibration
                ({",".join(fields)})
                VALUES ({",".join(["{self.placeholder}"] * len(fields))}"""
        self.execute_commit(
            query,
            (
                calibration_trial_id_fs,
                calibration_trial_id_bs,
                dt,
                False,
                method,
                gitHash,
                gitRepo,
                sid_fs,
                sid_bs,
                principalp_h_fs,
                principalp_v_fs,
                principalp_h_bs,
                principalp_v_bs,
            ),
        )
        return self.cursor.lastrowid


class DuplaDatabaseSqlite3(DuplaDatabaseConnector):
    def __init__(self, sqliteFile: Path, data_anchor: Path):
        """Sqlite3 version
        Args:
          sqliteFile: where is the sqlite3 file
          data_anchor: the anchor of the data server (e.g. P:/ or mount point)
        """
        self.sqliteFile = sqliteFile
        self.cnx = self._connect()
        self.cursor = self.cnx.cursor()
        # replace the anchor of each path that we concatenate
        self.data_anchor = Path(data_anchor)
        self.placeholder = "?"

    def _connect(self):
        return sqlite3.connect(self.sqliteFile)


if __name__ == "__main__":
    config = ConfigParser()
    config.read("config.ini")
    dbcon = DuplaDatabaseConnector(
        config["database"]["host"],
        config["database"]["database_name"],
        config["user"]["username"],
        config["user"]["password"],
    )
    print(dbcon.measurement_path_from_measurement_id(1))
    print(dbcon.anatomy_path_from_anatomy_id(1))
    print(dbcon.anatomy_path_from_measurement_id(1))
    print(dbcon.measurement_path_from_trial_id(1))
    print(dbcon.sic_info_from_measurement_id(1))
    print(dbcon.dist_info_from_measurement_id(1))
    print(dbcon.pose_info_from_measurement_id(1))
