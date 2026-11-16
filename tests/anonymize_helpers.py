"""Builders for the anonymizer's tests.

NIfTI files are assembled byte by byte (header, extension flag, extensions,
voxels, trailing bytes), so a test knows every byte of its input and can
plant text anywhere. gzip streams are assembled by hand too, because the
gzip module cannot write a comment or an extra field.
"""
from __future__ import annotations

import io
import struct
import zlib
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from nibabel.nifti1 import Nifti1Header
from nibabel.nifti2 import Nifti2Header

DATA_OFFSET = {1: 352, 2: 544}

# A planted patient name, and one distinct token per hiding place, so a leak
# names the place it came from.
NAME = "Hendrickx^Marieke"
PLANTS = {
    "descrip": "Hendrickx^Marieke DOB 19610423",
    "descrip_utf8": "Müller^Zoë",
    "aux_file": "HendrickxAUX",
    "db_name": "HendrickxDB",
    "intent_name": "HendrickxIN",
    "data_type": "HdxDTYPE",
    "unused_str": "HendrickxUS",
    "comment_ext": "Patient Hendrickx^Marieke comment extension",
    "dicom_ext": "Hendrickx^Marieke^DCMEXT",
    "utf16_ext": "Hendrickx^Marieke^UTF16",
    "gzip_name": "Hendrickx_Marieke_GZNAME.nii",
    "gzip_comment": "Hendrickx Marieke GZCOMMENT",
    "gzip_extra": "HendrickxGZEXTRA",
    "after_voxels": "TRAILER Hendrickx^Marieke",
    "file_name": "Hendrickx_Marieke_FILENAME",
}
ENCODINGS = ("ascii", "utf-8", "utf-16-le", "utf-16-be")


def leaks(data: bytes, tokens: Iterable[str]) -> List[Tuple[str, str]]:
    """(token, encoding) for every token found in ``data`` in any of
    ENCODINGS. A token that cannot be written in an encoding is skipped
    for that encoding only."""
    found = []
    for token in tokens:
        for encoding in ENCODINGS:
            try:
                needle = token.encode(encoding)
            except UnicodeEncodeError:
                continue
            if needle in data:
                found.append((token, encoding))
    return found


def all_tokens() -> List[str]:
    """Every planted token, plus the name's parts on their own."""
    return sorted(set(PLANTS.values()) | {"Hendrickx", "Marieke", "Müller", "Zoë"})


def extension_bytes(extensions: Sequence[Tuple[int, bytes]], endian: str) -> bytes:
    """NIfTI header extensions: size (a multiple of 16), code, padded data."""
    out = b""
    for code, data in extensions:
        size = 8 + len(data)
        size += (-size) % 16
        out += struct.pack(endian + "ii", size, code) + data + b"\x00" * (size - 8 - len(data))
    return out


def make_header(shape, dtype, *, version: int = 1, endian: str = "<",
                zooms: Optional[Sequence[float]] = None,
                qform: Optional[np.ndarray] = None, qcode: int = 1,
                sform: Optional[np.ndarray] = None, scode: int = 1,
                slope: Optional[float] = None, inter: Optional[float] = None,
                fields: Optional[Dict[str, object]] = None):
    """A nibabel header object with the given layout and geometry."""
    header = (Nifti1Header if version == 1 else Nifti2Header)(endianness=endian)
    header.set_data_dtype(np.dtype(dtype))
    header.set_data_shape(shape)
    if zooms is not None:
        header.set_zooms(zooms)
    if qform is not None:
        header.set_qform(qform, code=qcode)
    if sform is not None:
        header.set_sform(sform, code=scode)
    if slope is not None:
        header["scl_slope"] = slope
        header["scl_inter"] = 0.0 if inter is None else inter
    for name, value in (fields or {}).items():
        header[name] = value
    return header


def nifti_bytes(data: np.ndarray, *, version: int = 1, endian: str = "<",
                extensions: Sequence[Tuple[int, bytes]] = (), after: bytes = b"",
                header=None, **header_args) -> Tuple[bytes, bytes]:
    """(file bytes, voxel block bytes) of a single-file NIfTI holding
    ``data``, stored in ``endian`` byte order, Fortran order as NIfTI wants."""
    if header is None:
        header = make_header(data.shape, data.dtype, version=version, endian=endian,
                             **header_args)
    ext = extension_bytes(extensions, endian)
    header["vox_offset"] = DATA_OFFSET[version] + len(ext)
    flag = b"\x01\x00\x00\x00" if extensions else b"\x00\x00\x00\x00"
    stored = np.asarray(data, dtype=np.dtype(data.dtype).newbyteorder(endian))
    voxels = stored.tobytes(order="F")
    return header.binaryblock + flag + ext + voxels + after, voxels


def gzip_bytes(payload: bytes, *, name: Optional[bytes] = None,
               comment: Optional[bytes] = None, extra: Optional[bytes] = None,
               mtime: int = 0, level: int = 6) -> bytes:
    """One gzip member with the chosen optional header fields."""
    flags = 0
    tail = b""
    if extra is not None:
        flags |= 0x04
        sub = b"NV" + struct.pack("<H", len(extra)) + extra
        tail += struct.pack("<H", len(sub)) + sub
    if name is not None:
        flags |= 0x08
        tail += name + b"\x00"
    if comment is not None:
        flags |= 0x10
        tail += comment + b"\x00"
    head = b"\x1f\x8b\x08" + bytes([flags]) + struct.pack("<I", mtime) + b"\x00\xff" + tail
    compressor = zlib.compressobj(level, zlib.DEFLATED, -zlib.MAX_WBITS)
    body = compressor.compress(payload) + compressor.flush()
    return head + body + struct.pack("<II", zlib.crc32(payload) & 0xFFFFFFFF,
                                     len(payload) & 0xFFFFFFFF)


def dicom_extension(patient_name: str) -> bytes:
    """A small DICOM dataset (implicit VR little endian) holding a patient
    name, as a converter writes into a NIfTI DICOM extension (code 2)."""
    from pydicom.dataset import Dataset
    from pydicom.filebase import DicomBytesIO
    from pydicom.filewriter import write_dataset

    ds = Dataset()
    ds.PatientName = patient_name
    ds.PatientID = "PID-EXT-55501"
    buffer = DicomBytesIO()
    buffer.is_little_endian = True
    buffer.is_implicit_VR = True
    write_dataset(buffer, ds)
    return buffer.getvalue()


def write_dicom_series(folder: str, *, series_number: int, plants: Dict[str, str],
                       slices: int = 4, rows: int = 16, cols: int = 16, seed: int = 0,
                       study_uid: Optional[str] = None, prefix: str = "IM",
                       positions: Optional[Sequence[Sequence[float]]] = None,
                       orientation: Sequence[float] = (1, 0, 0, 0, 1, 0),
                       modality: str = "CT", image_type: Optional[Sequence[str]] = None,
                       extra: Optional[Dict[str, object]] = None,
                       pixels: Optional[np.ndarray] = None, first_instance: int = 1,
                       series_uid: Optional[str] = None,
                       frame_uid: Optional[str] = None) -> np.ndarray:
    """Write a small synthetic CT (or MR) series with ``plants`` as DICOM
    attributes (keyword -> value) and ``extra`` attributes on top (a gantry
    tilt, an echo number). ``positions`` gives each slice's position (default:
    2.5 mm apart along z). Returns the stored pixel array (slices, rows, cols)."""
    import os

    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, MRImageStorage, generate_uid

    if positions is None:
        positions = [(-6.0, -6.0, 2.5 * index) for index in range(slices)]
    slices = len(positions)
    if pixels is None:
        rng = np.random.default_rng(seed)
        pixels = rng.integers(0, 3000, size=(slices, rows, cols), dtype=np.int16)
    sop_class = MRImageStorage if modality == "MR" else CTImageStorage
    study_uid = study_uid or generate_uid()
    series_uid = series_uid or generate_uid()
    frame_uid = frame_uid or generate_uid()
    for index in range(slices):
        meta = FileMetaDataset()
        meta.MediaStorageSOPClassUID = sop_class
        meta.MediaStorageSOPInstanceUID = sop_uid = generate_uid()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds = Dataset()
        ds.file_meta = meta
        ds.SOPClassUID = sop_class
        ds.SOPInstanceUID = sop_uid
        for keyword, value in plants.items():
            setattr(ds, keyword, value)
        ds.Modality = modality
        ds.StudyInstanceUID = study_uid
        ds.SeriesInstanceUID = series_uid
        ds.FrameOfReferenceUID = frame_uid
        ds.SeriesNumber = series_number
        ds.InstanceNumber = first_instance + index
        if image_type is not None:
            ds.ImageType = list(image_type)
        ds.ImagePositionPatient = [float(v) for v in positions[index]]
        ds.ImageOrientationPatient = [float(v) for v in orientation]
        ds.PixelSpacing = [0.8, 0.8]
        ds.SliceThickness = 2.5
        ds.Rows, ds.Columns = rows, cols
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 1
        ds.RescaleIntercept = -1024
        ds.RescaleSlope = 1
        for keyword, value in (extra or {}).items():
            setattr(ds, keyword, value)
        ds.PixelData = np.ascontiguousarray(pixels[index], dtype=np.int16).tobytes()
        save_part10(ds, os.path.join(folder, f"{prefix}{series_number:02d}{index:04d}"))
    return pixels


def tilted_positions(slices: int = 6, step: float = 2.5) -> List[Tuple[float, float, float]]:
    """Slice positions of a CT with gantry tilt: the table moves along z while
    each image plane is tilted (pair with ``tilted_orientation``)."""
    return [(-6.0, -6.0, step * index) for index in range(slices)]


def tilted_orientation(degrees: float = 15.0) -> Tuple[float, ...]:
    """The image orientation of a CT slice tilted by ``degrees`` about x."""
    import math
    angle = math.radians(degrees)
    return (1.0, 0.0, 0.0, 0.0, math.cos(angle), math.sin(angle))


def write_secondary_capture(folder: str, *, plants: Dict[str, str], text: str,
                            series_number: int = 900, name: str = "SC0001") -> bytes:
    """Write a screenshot as scanners export one (a dose report, a saved
    view): a Secondary Capture image, DERIVED, 8-bit, with no geometry, the
    patient details in its tags and ``text`` written into its pixels (the
    ASCII bytes of each row). Returns the pixel bytes."""
    import os

    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid

    rows, cols = 32, 64
    pixels = np.zeros((rows, cols), dtype=np.uint8)
    line = np.frombuffer(text.encode("ascii")[:cols], dtype=np.uint8)
    for row in range(4, rows, 6):
        pixels[row, :len(line)] = line
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    meta.MediaStorageSOPInstanceUID = sop_uid = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = Dataset()
    ds.file_meta = meta
    ds.SOPClassUID = SecondaryCaptureImageStorage
    ds.SOPInstanceUID = sop_uid
    for keyword, value in plants.items():
        setattr(ds, keyword, value)
    ds.Modality = "OT"
    ds.ConversionType = "WSD"
    ds.ImageType = ["DERIVED", "SECONDARY"]
    ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = generate_uid()
    ds.SeriesNumber = series_number
    ds.InstanceNumber = 1
    ds.Rows, ds.Columns = rows, cols
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 8
    ds.BitsStored = 8
    ds.HighBit = 7
    ds.PixelRepresentation = 0
    ds.PixelData = pixels.tobytes()
    save_part10(ds, os.path.join(folder, name))
    return pixels.tobytes()


def write_localizer(folder: str, *, plants: Dict[str, str], series_number: int = 1,
                    study_uid: Optional[str] = None) -> None:
    """Write a three-plane scout: three single images of one series, each in
    another orientation, typed LOCALIZER."""
    from pydicom.uid import generate_uid

    series_uid, frame_uid = generate_uid(), generate_uid()
    study_uid = study_uid or generate_uid()
    for index, orientation in enumerate([(1, 0, 0, 0, 1, 0), (0, 1, 0, 0, 0, -1),
                                         (1, 0, 0, 0, 0, -1)]):
        write_dicom_series(folder, series_number=series_number, plants=plants,
                           positions=[(0.0, 0.0, 0.0)], orientation=orientation,
                           image_type=["ORIGINAL", "PRIMARY", "LOCALIZER"],
                           prefix=f"LOC{index}", seed=50 + index, first_instance=index + 1,
                           series_uid=series_uid, frame_uid=frame_uid, study_uid=study_uid)


def _pydicom_major() -> int:
    import pydicom
    return int(str(pydicom.__version__).split(".")[0])


def save_part10(ds, path: str) -> None:
    """Write ``ds`` as a DICOM Part 10 file (preamble, "DICM", file meta) with
    pydicom 2 or 3: the keyword changed between them."""
    if _pydicom_major() >= 3:
        ds.save_as(path, enforce_file_format=True)
    else:
        ds.save_as(path, write_like_original=False)


def save_bare(ds, path: str, *, implicit: bool) -> None:
    """Write ``ds`` with no preamble and no file meta (an older export),
    little-endian, implicit or explicit VR, with pydicom 2 or 3."""
    import pydicom

    if _pydicom_major() >= 3:
        pydicom.dcmwrite(path, ds, implicit_vr=implicit, little_endian=True,
                         enforce_file_format=False)
    else:
        ds.is_implicit_VR = implicit
        ds.is_little_endian = True
        pydicom.dcmwrite(path, ds, write_like_original=True)


DICOM_PLANTS = {
    "PatientName": "Hendrickx^Marieke",
    "PatientID": "PID-7731942",
    "PatientBirthDate": "19610423",
    "InstitutionName": "Sint-Hendrickx Ziekenhuis",
    "InstitutionAddress": "Kerkstraat 12 Brugge",
    "AccessionNumber": "ACC99887766",
    "StudyDate": "20240229",
    "SeriesDate": "20240229",
    "StudyTime": "134502",
    "AcquisitionTime": "134517",
    "ReferringPhysicianName": "Vandamme^Pieter",
    "PerformingPhysicianName": "Claes^Ingrid",
    "OperatorsName": "Peeters^Wout",
    "StudyDescription": "CT thorax Hendrickx",
    "SeriesDescription": "Thorax series Hendrickx",
    "ProtocolName": "Protocol Hendrickx",
    "ImageComments": "Comment Hendrickx Marieke",
    "StationName": "STATION-HDX7",
    "StudyID": "STUDY4471",
}
# What to search the output for: every planted value, and its parts.
DICOM_TOKENS = sorted(
    set(DICOM_PLANTS.values())
    | {"Hendrickx", "Marieke", "Vandamme", "Pieter", "Claes", "Ingrid", "Peeters", "Wout",
       "1961-04-23", "2024-02-29", "Kerkstraat", "Brugge"}
)


# A stand-in for dcm2niix (chosen with NALVERA_DCM2NIIX). It logs its command
# line, what it sees in its input folder, the folder modes and what its three
# standard streams are; prints patient details on stdout and stderr the way
# the real converter quotes DICOM values; writes the volumes named in
# STANDIN_NAMES; with STANDIN_COUNTS, prints the real converter's
# "Convert N DICOM as ..." line for each; exits with STANDIN_EXIT.
STANDIN = r'''#!{python}
import json, os, stat, sys
args = sys.argv[1:]
out = args[args.index("-o") + 1]
src = args[-1]
entries = []
for name in sorted(os.listdir(src)):
    path = os.path.join(src, name)
    entries.append({{"name": name, "link": os.path.islink(path),
                     "target": os.path.realpath(path)}})
modes = {{p: oct(os.stat(p).st_mode & 0o777) for p in (os.path.dirname(out), out, src)}}
def kind(fd):
    mode = os.fstat(fd).st_mode
    if stat.S_ISFIFO(mode):
        return "pipe"
    if stat.S_ISCHR(mode):
        return "null" if os.path.samestat(os.fstat(fd), os.stat(os.devnull)) else "terminal"
    return "file"
streams = {{"stdin": kind(0), "stdout": kind(1), "stderr": kind(2)}}
with open(os.environ["STANDIN_LOG"], "w") as fh:
    json.dump({{"args": args, "entries": entries, "modes": modes, "streams": streams}}, fh)
print("Converting Hendrickx^Marieke PID-7731942 " + src)
sys.stderr.write("Warning: PatientName Hendrickx^Marieke\n")
names = [n for n in os.environ.get("STANDIN_NAMES", "").split(",") if n]
counts = [n for n in os.environ.get("STANDIN_COUNTS", "").split(",") if n]
if names:
    import numpy as np, nibabel as nib
    for index, name in enumerate(names):
        data = (np.arange(6 * 5 * 4).reshape(6, 5, 4) + index).astype(np.int16)
        image = nib.Nifti1Image(data, np.diag([0.8, 0.8, 2.5, 1.0]))
        image.header["descrip"] = b"Time=134517;Hendrickx^Marieke"
        image.header["aux_file"] = b"Comment Hendrickx"
        nib.save(image, os.path.join(out, name))
        if index < len(counts):
            stem = name.split(".nii")[0]
            print("Convert " + counts[index] + " DICOM as " + os.path.join(out, stem)
                  + " (6x5x4x1)")
sys.exit(int(os.environ.get("STANDIN_EXIT", "0")))
'''


def install_standin(tmp_path, monkeypatch, names: str = "vol_s5.nii.gz"):
    """Use the stand-in converter for this test; returns a function that
    reads its log."""
    import json
    import sys

    script = tmp_path / "standin-dcm2niix"
    script.write_text(STANDIN.format(python=sys.executable))
    script.chmod(0o755)
    log = tmp_path / "standin.json"
    monkeypatch.setenv("NALVERA_DCM2NIIX", str(script))
    monkeypatch.setenv("STANDIN_LOG", str(log))
    monkeypatch.setenv("STANDIN_NAMES", names)
    monkeypatch.delenv("STANDIN_EXIT", raising=False)
    monkeypatch.delenv("STANDIN_COUNTS", raising=False)
    return lambda: json.loads(log.read_text())


def hide_converter(tmp_path, monkeypatch) -> None:
    """Make dcm2niix unavailable: no override, the Python package absent,
    nothing on PATH."""
    import sys

    monkeypatch.delenv("NALVERA_DCM2NIIX", raising=False)
    monkeypatch.setitem(sys.modules, "dcm2niix", None)
    monkeypatch.setenv("PATH", str(tmp_path / "nothing-here"))


def read_gzip_members(data: bytes) -> bytes:
    """Decompress every gzip member in ``data`` (a strict reference reader)."""
    out = io.BytesIO()
    while data:
        decompressor = zlib.decompressobj(zlib.MAX_WBITS | 16)
        out.write(decompressor.decompress(data))
        data = decompressor.unused_data
    return out.getvalue()
