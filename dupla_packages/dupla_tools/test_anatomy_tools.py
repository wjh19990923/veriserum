import pytest
from pathlib import Path
from .anatomy_tools import anatomy_name_from_idx, transfer_anatomy_to_server

@pytest.mark.parametrize(
    "idx, suffix, exp",
    [
        (2100, ".stl", "ana_002100.stl"),
        (15, ".nii", "ana_000015.nii"),
        pytest.param(10e7, ".stl", "foo", marks=pytest.mark.xfail(reason="too many digits")),
        pytest.param(500, ".STL", "foo", marks=pytest.mark.xfail(reason="uppercase suffix")),
        pytest.param(500, "txt", "foo", marks=pytest.mark.xfail(reason="wrong suffix")),
    ],
)
def test_anatomy_name_from_idx(idx, suffix, exp):
    res = anatomy_name_from_idx(idx, suffix)
    assert res == exp


@pytest.mark.parametrize(
    "anaIdx, sourcePath, check",
    [
        (15, Path("testfiles/testAna1.stl"), True),
        (7, Path("testfiles/testAna1.stl"), False),
        (21, Path("testfiles/testAna2.stl"), True),
        (15151, Path("testfiles/testAna2.stl"), False),
    ],
)
def test_transfer_anatomy_to_server(tmp_path, anaIdx, sourcePath, check):
    """transfer test stls and check whether the file exists in the target directory
    (or wheter it doesnt exist when check_only=True)"""
    transfer_anatomy_to_server(anaIdx=anaIdx, sourcePath=sourcePath, targetDir=tmp_path, check_only=check)
    assert (tmp_path / anatomy_name_from_idx(anaIdx, sourcePath.suffix)).exists() == (not check)

