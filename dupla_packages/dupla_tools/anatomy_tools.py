import os
from pathlib import Path
import shutil

def anatomy_name_from_idx(idx: int, suffix: str) -> str:
    """formats the name of the anatomy from the anatomy.id and suffix
    Args:
        idx: the anatomies.id
        suffix: either ".stl" or ".nii"
    """
    assert idx < 10e6, "Only supporting 6 digits"
    assert suffix in (".stl", ".nii"), "Unsupported file format"
    return f"ana_{str(idx):0>6}{suffix}"


def transfer_anatomy_to_server(
    anaIdx: int, sourcePath: Path, targetDir: Path, check_only: bool = False
) -> Path:
    """copy anatomy file to the NAS
    Args:
        sourcePath: the absolute Path to the .nii or .stl file
        targetDir: absolute Path where to copy this file, name is given by anaIdx (6 digits) suffix
        check_only: only check whether the file could be copied
    Returns:
        complete Path to written file, with suffix. Example: ana_000015.stl
    """
    targetPath = targetDir / anatomy_name_from_idx(anaIdx, sourcePath.suffix)
    # creates directories if needed
    Path(targetDir).mkdir(parents=True, exist_ok=True)
    assert not targetPath.exists(), f"File under {targetPath} already exists"
    if check_only:
        # additional checks
        assert not targetDir.is_file(), f"The given targetPath {targetDir} points to a file instead of a directory"
        assert os.access(targetDir, os.W_OK), f"Parent directory {targetDir} is not writable"
    else:
        shutil.copyfile(sourcePath, targetPath)
    return targetPath
