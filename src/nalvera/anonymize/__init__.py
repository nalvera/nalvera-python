"""Anonymize scans on this computer before anything is uploaded.

``prepare`` turns a NIfTI file, a folder of NIfTI files or a DICOM folder
into anonymized ``.nii.gz`` files named by random codes, checks each one, and
returns a ``PreparedScan`` per scan. ``inspect`` says what a path holds
without writing anything (for a cost estimate before uploading). ``Mapping``
keeps the local record of which code is which file.

What it does: DICOM is converted here (no DICOM tag leaves the computer); the
NIfTI header is rebuilt from an allow-list of technical fields; header
extensions, text fields and bytes after the voxels are dropped; the file name
is replaced by the code; the gzip stream carries no name or time. The stored
voxels, data type, scaling and geometry are bit-identical to the input. The
output is NIfTI-1, the one version the platform accepts.

What it does not do: remove anything from the image. A face in a head scan and
text burned into an image stay as they are. DICOM screenshots, dose reports and scouts,
which carry the patient's name in their pixels, are left out by the
converter.

There is no off switch, no network access, and no output to the terminal
anywhere in this module: everything it has to say is in what it returns or
raises.

Example::

    from nalvera.anonymize import Mapping, inspect, prepare

    found = inspect("dicom_folder/", convert=True)        # what it would become
    scans = prepare("dicom_folder/", workdir="/tmp/nalvera-work", label="P012 baseline")
    mapping = Mapping("nalvera_scans.csv")
    for scan in scans:
        mapping.record_scan(scan)
        print(scan.original_name, scan.series, "->", scan.filename, scan.shape)
    for volume in scans.dropped:
        print("left out:", volume.series, volume.reason, "kept:", volume.kept)
    scans.discard()                                       # when nothing is uploaded
"""
from __future__ import annotations

import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from typing import Callable, Collection, Iterable, List, Optional, Tuple

from . import dicom
from .errors import AnonymizeError
from .mapping import Mapping
from .names import clean_label, filename_for, new_code
from .nifti import content_key, describe, looks_like_nifti, read_clean, write_clean
from .verify import verify_clean

__all__ = ["prepare", "inspect", "PreparedScan", "PreparedScans", "DroppedVolume",
           "Inspection", "FoundScan", "Mapping", "AnonymizeError"]

# A folder's NIfTI files are found by name, at its top level only.
_NIFTI_NAME_RX = re.compile(r"\.nii(\.gz)?$", re.IGNORECASE)
#: The name ``inspect`` gives its temporary folder: the package's own prefix,
#: so a folder left by a killed process is recognised and removed later.
INSPECT_PREFIX = "nalvera-inspect-"
#: The DICOM conversion's private folder, in the system's temporary folder.
DICOM_PREFIX = "nalvera-dicom-"


@dataclass(frozen=True)
class PreparedScan:
    """One anonymized scan, ready to upload.

    ``code`` is its random code (``scan-7F3K2Q9A``), the name the platform
    shows. ``path`` is the anonymized file inside the work folder, named
    ``filename`` (``<code>.nii.gz``); ``size`` and ``sha256`` describe that
    file; ``shape`` is the image's dimensions (for a cost estimate).
    ``original_path`` and ``original_name`` say where it came from and stay
    on this computer (for DICOM: the folder, or the single file without its
    extension). ``source`` is ``"nifti"`` or ``"dicom"``. ``series`` is the
    DICOM series number with the converter's postfix when one series gave
    several volumes (``"3"``, ``"3a"``, ``"5_e2"``, ``"3_Tilt_1"``; None for
    NIfTI). ``label`` is the user's own label, which does reach the platform.
    ``sha256_original`` is the SHA-256 of the original NIfTI file (None for
    DICOM). ``corrections`` says what the converter corrected in this volume
    (``"gantry_tilt"``, ``"slice_spacing"``; empty for most scans).
    """
    code: str
    filename: str
    path: str
    size: int
    sha256: str
    shape: Tuple[int, ...]
    original_path: str
    original_name: str
    source: str
    series: Optional[str] = None
    label: Optional[str] = None
    sha256_original: Optional[str] = None
    corrections: Tuple[str, ...] = ()


@dataclass(frozen=True)
class DroppedVolume:
    """A volume the DICOM converter wrote that did not become a scan.

    ``series`` is the converter's label for it (``"3"``, ``"3a"``).
    ``reason`` is ``"uncorrected"`` (a copy corrected for gantry tilt or
    uneven slice spacing is used in its place) or ``"duplicate"`` (the same
    voxels, geometry and header as a volume that is used). ``kept`` is the
    series label of the scan used in its place. ``shape`` is its dimensions,
    or None when its header could not be read.
    """
    series: Optional[str]
    reason: str
    kept: Optional[str]
    shape: Optional[Tuple[int, ...]] = None


class PreparedScans(list):
    """What ``prepare`` returns: a list of PreparedScan (``scans``), and what
    did not become a scan.

    ``dropped``: a list of DroppedVolume, each volume the converter wrote
    that is not uploaded, and why. ``left_out``: how many DICOM files are in
    no scan (screenshots, dose reports, scouts, derived and single images,
    duplicates, files the converter could not read); None for NIfTI input or
    when the converter's counts could not be read. ``converter_status``:
    for DICOM input the converter's exit status, 0, or not 0 when it failed
    part-way and the scans are the volumes it wrote (see ``partial``); None
    for NIfTI input.
    """

    def __init__(self, scans: Iterable[PreparedScan] = (), *,
                 dropped: Iterable[DroppedVolume] = (), left_out: Optional[int] = None,
                 converter_status: Optional[int] = None):
        super().__init__(scans)
        self.dropped = list(dropped)
        self.left_out = left_out
        self.converter_status = converter_status

    @property
    def partial(self) -> bool:
        """True when the DICOM converter failed part-way: these scans are
        what it converted, and some files were not (see ``left_out``)."""
        return self.converter_status not in (None, 0)

    def discard(self) -> None:
        """Delete every anonymized file of this result from the work folder
        (when the run stops before uploading them). Harmless when they are
        already gone."""
        for scan in self:
            _remove(scan.path)


@dataclass(frozen=True)
class FoundScan:
    """One scan ``inspect`` found, before anything is written.

    ``path`` is the NIfTI file, or for DICOM the folder or file given.
    ``name`` is its original name (it stays on this computer). ``source`` is
    ``"nifti"`` or ``"dicom"``. ``shape`` is its dimensions (None when its
    header gives none). ``series`` and ``corrections`` are those its
    PreparedScan would have (DICOM only). ``error`` is the AnonymizeError
    code ``prepare`` would raise for it (for example ``nifti2_unsupported``,
    ``unreadable``), or None.
    """
    path: str
    name: str
    source: str
    shape: Optional[Tuple[int, ...]]
    series: Optional[str] = None
    corrections: Tuple[str, ...] = ()
    error: Optional[str] = None


@dataclass(frozen=True)
class Inspection:
    """What ``inspect`` found at ``path``.

    ``kind`` is ``"nifti"`` (one file), ``"nifti_folder"`` (the NIfTI files
    at a folder's top level, each a scan), ``"dicom"`` (DICOM files in a
    folder and its subfolders, or one DICOM file) or ``"mixed"`` (both:
    ``prepare`` refuses it with ``mixed_folder``). ``scans`` holds a
    FoundScan per scan: for DICOM only when ``converted``. ``nifti_files``
    and ``dicom_files`` count the files found. ``converted`` is True when
    the DICOM files were converted to learn their scans. ``dropped``,
    ``left_out`` and ``converter_status`` are as in PreparedScans (DICOM
    converted only).
    """
    path: str
    kind: str
    scans: Tuple[FoundScan, ...]
    nifti_files: int
    dicom_files: int
    converted: bool = False
    dropped: Tuple[DroppedVolume, ...] = ()
    left_out: Optional[int] = None
    converter_status: Optional[int] = None

    @property
    def needs_conversion(self) -> bool:
        """True when DICOM files were found but not converted, so their scans
        and shapes are unknown (call ``inspect`` with ``convert=True``)."""
        return self.kind in ("dicom", "mixed") and not self.converted


def prepare(path, *, workdir, label: Optional[str] = None,
            max_bytes: Optional[int] = None,
            avoid: Optional[Collection[str]] = None,
            check: Optional[Callable[[str], None]] = None) -> PreparedScans:
    """Anonymize the scan(s) at ``path`` into ``workdir`` and return a
    PreparedScans: one PreparedScan per scan, and what was left out.

    ``path`` (required): a ``.nii`` or ``.nii.gz`` file (recognised by
    content); a folder whose top level holds ``.nii``/``.nii.gz`` files (each
    is a scan; the folder is prepared whole or not at all: for one result per
    file, list them with ``inspect`` and prepare each); a folder of DICOM
    files (searched with its subfolders; each series and each distinct
    volume the converter writes becomes a scan, a corrected copy replaces
    its original, identical volumes become one, and screenshots, scouts and
    derived images are left out); or a single DICOM file.
    ``workdir`` (required, keyword): a private folder the caller owns,
    created with mode 0700 when missing; each scan is written there as
    ``<code>.nii.gz`` with mode 0600 and checked before it is returned.
    ``label`` (default None: no label): 1 to 60 letters, digits, spaces,
    dots, dashes or underscores, validated and passed through to each scan.
    ``max_bytes`` (default None: no limit): refuses a scan whose anonymized
    file is larger (the upload limit).
    ``avoid`` (default None): codes that must not be drawn, such as those
    the record already holds; a code among them is drawn again.
    ``check`` (default None): called with the real path of each file just
    before it is read (a NIfTI file) or linked for the converter (a DICOM
    file); it raises to refuse it, such as a file outside allowed folders
    that a link put in place after the folder was listed.

    The scans can be prepared long before they are uploaded, or never: each
    has its ``shape`` for a cost estimate, and ``discard()`` deletes them.

    Raises AnonymizeError with a stable ``code``: ``not_found``,
    ``not_a_scan``, ``unreadable``, ``too_large``, ``nifti2_unsupported``,
    ``mixed_folder``, ``dicom_tool_missing`` (DICOM only: NIfTI needs no
    converter), ``conversion_failed``, ``no_volume``,
    ``verification_failed``, ``invalid_label``. On any error nothing this
    call wrote remains in ``workdir``. An OSError from writing to
    ``workdir`` (a full disk) passes through, with the same clean-up.
    """
    label = clean_label(label)
    source = os.path.abspath(os.fspath(path))
    if not os.path.exists(source):
        raise AnonymizeError("not_found")
    workdir = os.path.abspath(os.fspath(workdir))
    os.makedirs(workdir, mode=0o700, exist_ok=True)

    if os.path.isdir(source):
        niftis = _nifti_files(source)
        dicoms = dicom.list_dicom_files(source)
        if niftis and dicoms:
            raise AnonymizeError("mixed_folder")
        if niftis:
            return _from_niftis(niftis, workdir, label, max_bytes, avoid, check)
        if dicoms:
            return _from_dicom(dicoms, source, _folder_name(source), workdir, label, max_bytes,
                               avoid, check)
        raise AnonymizeError("not_a_scan", "the folder holds no scan")
    if not os.path.isfile(source):
        raise AnonymizeError("not_a_scan")
    if looks_like_nifti(source):
        return _from_niftis([source], workdir, label, max_bytes, avoid, check)
    if dicom.is_dicom_file(source):
        return _from_dicom([source], source, _file_stem(source), workdir, label, max_bytes,
                           avoid, check)
    raise AnonymizeError("not_a_scan")


def inspect(path, *, convert: bool = False, workdir=None) -> Inspection:
    """What ``path`` holds, learned without writing an anonymized file: for
    a cost estimate and a budget question before anything is uploaded.

    ``path`` (required): what ``prepare`` takes. A NIfTI file is read up to
    its header (cheap); a folder lists the NIfTI files at its top level with
    their shapes and what ``prepare`` would refuse in each, and counts its
    DICOM files.
    ``convert`` (default False): DICOM scans and their shapes are known only
    after conversion, which takes seconds to minutes for a large folder.
    False counts the DICOM files and runs nothing (see
    ``Inspection.needs_conversion``). True converts them in a private
    temporary folder, applies the rules of ``prepare`` (corrected copies,
    identical volumes, files left out), and removes the folder; ``prepare``
    converts again, so to convert once, call ``prepare`` and use each
    PreparedScan's ``shape``.
    ``workdir`` (default None: the system's temporary folder): where that
    temporary folder is made when ``convert`` is True.

    Raises AnonymizeError: ``not_found``, ``not_a_scan`` (a file that is no
    scan, an empty folder), ``unreadable`` (a folder that cannot be read),
    ``too_large`` (a folder of more than 100000 DICOM files), and with
    ``convert`` True ``dicom_tool_missing``, ``conversion_failed`` and
    ``no_volume``. A problem with one NIfTI file is that FoundScan's
    ``error``, never an exception.
    """
    source = os.path.abspath(os.fspath(path))
    if not os.path.exists(source):
        raise AnonymizeError("not_found")
    if os.path.isdir(source):
        niftis = _nifti_files(source)
        dicoms = dicom.list_dicom_files(source)
        if niftis:
            kind = "mixed" if dicoms else "nifti_folder"
            return Inspection(path=source, kind=kind,
                              scans=tuple(_found_nifti(p) for p in niftis),
                              nifti_files=len(niftis), dicom_files=len(dicoms))
        if dicoms:
            return _inspect_dicom(dicoms, source, _folder_name(source), convert, workdir)
        raise AnonymizeError("not_a_scan", "the folder holds no scan")
    if not os.path.isfile(source):
        raise AnonymizeError("not_a_scan")
    if looks_like_nifti(source):
        return Inspection(path=source, kind="nifti", scans=(_found_nifti(source),),
                          nifti_files=1, dicom_files=0)
    if dicom.is_dicom_file(source):
        return _inspect_dicom([source], source, _file_stem(source), convert, workdir)
    raise AnonymizeError("not_a_scan")


def _scan_folder(source: str) -> str:
    """The folder the scans are in: a converter found there is not run."""
    return source if os.path.isdir(source) else os.path.dirname(source)


def _folder_name(folder: str) -> str:
    return os.path.basename(os.path.normpath(folder))


def _file_stem(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def _nifti_files(folder: str) -> List[str]:
    """The NIfTI files at the top of ``folder`` (named .nii or .nii.gz, not
    hidden), by name. Subfolders are not searched."""
    try:
        names = sorted(os.listdir(folder))
    except OSError as exc:
        raise AnonymizeError("unreadable", f"the folder cannot be read ({type(exc).__name__})") \
            from None
    found = []
    for name in names:
        if name.startswith(".") or not _NIFTI_NAME_RX.search(name):
            continue
        candidate = os.path.join(folder, name)
        if os.path.isfile(candidate):
            found.append(candidate)
    return found


def _found_nifti(path: str) -> FoundScan:
    name = os.path.basename(path)
    try:
        summary = describe(path)
    except AnonymizeError as error:
        return FoundScan(path=path, name=name, source="nifti", shape=None, error=error.code)
    return FoundScan(path=path, name=name, source="nifti", shape=summary.shape,
                     error=summary.problem)


def _from_niftis(paths: List[str], workdir: str, label: Optional[str],
                 max_bytes: Optional[int], avoid: Optional[Collection[str]] = None,
                 check: Optional[Callable[[str], None]] = None) -> PreparedScans:
    made: List[PreparedScan] = []
    try:
        for path in paths:
            if check is not None:
                check(os.path.realpath(path))
            scan, _info = _anonymize(path, workdir, max_bytes, original_path=path,
                                     original_name=os.path.basename(path), source="nifti",
                                     series=None, label=label, keep_source_hash=True,
                                     avoid=avoid)
            made.append(scan)
        return PreparedScans(made)
    except BaseException:
        for scan in made:
            _remove(scan.path)
        raise


def _from_dicom(files: List[str], source: str, base: str, workdir: str,
                label: Optional[str], max_bytes: Optional[int],
                avoid: Optional[Collection[str]] = None,
                check: Optional[Callable[[str], None]] = None) -> PreparedScans:
    # mkdtemp makes the folder with mode 0700. The converted volumes in it
    # still carry DICOM details in their headers; it is removed in every case.
    # It lives in the system's temporary folder, never in ``workdir``: the
    # caller may have named a folder a person shares (save_anonymized).
    private = tempfile.mkdtemp(prefix=DICOM_PREFIX)
    made: List[PreparedScan] = []

    def examine(volume: dicom.ConvertedVolume):
        scan, info = _anonymize(volume.path, workdir, max_bytes, original_path=source,
                                original_name=base, source="dicom", series=volume.series,
                                label=label, keep_source_hash=False,
                                corrections=volume.corrections, avoid=avoid)
        made.append(scan)
        return scan, content_key(info), scan.shape

    def discard(scan: PreparedScan) -> None:
        made.remove(scan)
        _remove(scan.path)

    try:
        conversion = dicom.convert(files, private, _scan_folder(source), check=check)
        scans, dropped = _distinct(conversion, examine, discard)
        return PreparedScans(scans, dropped=dropped, left_out=conversion.left_out,
                             converter_status=conversion.status)
    except BaseException:
        for scan in made:
            _remove(scan.path)
        raise
    finally:
        shutil.rmtree(private, ignore_errors=True)


def _inspect_dicom(files: List[str], source: str, base: str, convert: bool,
                   workdir) -> Inspection:
    if not convert:
        return Inspection(path=source, kind="dicom", scans=(), nifti_files=0,
                          dicom_files=len(files))
    parent = None
    if workdir is not None:
        parent = os.path.abspath(os.fspath(workdir))
        os.makedirs(parent, mode=0o700, exist_ok=True)
    private = tempfile.mkdtemp(prefix=INSPECT_PREFIX, dir=parent)

    def examine(volume: dicom.ConvertedVolume):
        try:
            info = read_clean(volume.path)
        except AnonymizeError as error:
            found = FoundScan(path=source, name=base, source="dicom", shape=None,
                              series=volume.series, corrections=volume.corrections,
                              error=error.code)
            return found, None, None
        found = FoundScan(path=source, name=base, source="dicom", shape=info.shape,
                          series=volume.series, corrections=volume.corrections)
        return found, content_key(info), info.shape

    try:
        conversion = dicom.convert(files, private, _scan_folder(source))
        scans, dropped = _distinct(conversion, examine, lambda found: None)
        return Inspection(path=source, kind="dicom", scans=tuple(scans), nifti_files=0,
                          dicom_files=len(files), converted=True, dropped=tuple(dropped),
                          left_out=conversion.left_out, converter_status=conversion.status)
    finally:
        shutil.rmtree(private, ignore_errors=True)


def _distinct(volumes: List[dicom.ConvertedVolume], examine: Callable,
              discard: Callable) -> Tuple[list, List[DroppedVolume]]:
    """The rules ``prepare`` and ``inspect`` share: a corrected copy replaces
    its original; of volumes whose anonymized files would be identical, the
    first in series order stays. ``examine(volume)`` gives (item, content
    key or None, shape) for a kept volume; ``discard(item)`` undoes a
    duplicate's item."""
    kept, replaced = dicom.choose(list(volumes))
    dropped = [DroppedVolume(series=gone.series, reason="uncorrected", kept=instead.series,
                             shape=_header_shape(gone.path)) for gone, instead in replaced]
    items, seen = [], {}
    for volume in kept:
        item, key, shape = examine(volume)
        if key is not None and key in seen:
            discard(item)
            dropped.append(DroppedVolume(series=volume.series, reason="duplicate",
                                         kept=seen[key], shape=shape))
            continue
        if key is not None:
            seen[key] = volume.series
        items.append(item)
    dropped.sort(key=lambda d: dicom.series_key(d.series))
    return items, dropped


def _header_shape(path: str) -> Optional[Tuple[int, ...]]:
    try:
        return describe(path).shape
    except AnonymizeError:
        return None


def _anonymize(volume: str, workdir: str, max_bytes: Optional[int], *, original_path: str,
               original_name: str, source: str, series: Optional[str], label: Optional[str],
               keep_source_hash: bool, corrections: Tuple[str, ...] = (),
               avoid: Optional[Collection[str]] = None):
    for _ in range(100):
        code = new_code()
        if avoid and code in avoid:
            continue
        target = os.path.join(workdir, filename_for(code))
        partial = target + ".partial"
        if not os.path.exists(target) and not os.path.exists(partial):
            break
    # Written under a temporary name, checked, then renamed: a file named
    # <code>.nii.gz has always passed the check. write_clean removes its own
    # output on failure, verify_clean a file that fails.
    info = write_clean(volume, partial, max_bytes)
    try:
        sha256, size = verify_clean(partial, info)
        os.replace(partial, target)
    except BaseException:
        _remove(partial)
        raise
    scan = PreparedScan(
        code=code, filename=filename_for(code), path=target, size=size, sha256=sha256,
        shape=info.shape, original_path=original_path, original_name=original_name,
        source=source, series=series, label=label,
        sha256_original=info.source_sha256 if keep_source_hash else None,
        corrections=tuple(corrections))
    return scan, info


def _remove(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass
