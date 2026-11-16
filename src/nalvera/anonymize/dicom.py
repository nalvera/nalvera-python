"""Convert DICOM files to NIfTI on this computer.

No DICOM file and no DICOM tag is ever uploaded. The files are converted
here with ``dcm2niix`` (the converter the Nalvera.AI website uses), into a
private folder the caller creates and removes. Each volume it produces then
goes through the NIfTI anonymizer (nifti.py), which builds a new header:
``dcm2niix`` writes acquisition details into the NIfTI ``descrip`` field and
the DICOM image comment into ``aux_file``, and neither survives.

The converter is given only links to the DICOM files (never copies outside
that private folder), its defaults file is ignored, and it is asked for no
sidecar file and for output names that hold no patient field: only the
series number. It leaves out derived images, scouts and single 2-D images
(``-i y``, as the website's converter): screenshots and dose reports carry
the patient's name in their pixels, which the anonymizer does not change.
Its standard streams are a null input and two pipes this module reads, so
nothing it prints reaches the terminal (or an MCP client on stdout).

One series can give several volumes; their names carry the converter's
postfix after the series number:
  ``_Tilt_1``, ``_Eq_1``, ``_Tilt_Eq_1``  a copy corrected for gantry tilt,
                                          for uneven slice spacing, or both,
                                          written beside the original
  ``_e2``                                 another echo
  ``_ph``, ``_real``, ``_imaginary``      a phase, real or imaginary image
  ``a``, ``b`` ...                        another series with the same number
``choose`` keeps a corrected copy in place of its original.
"""
from __future__ import annotations

import os
import re
import shutil
import struct
import subprocess
from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Tuple

from .errors import AnonymizeError

_DICM_OFFSET = 128
# One conversion may take a while on a large series; never forever.
_TIMEOUT_SECONDS = 1800
_MAX_FILES = 100_000
_PREFIX = "vol_s"
_SUFFIX_RX = re.compile(r"\.nii(\.gz)?$")
# The converter's postfixes for a corrected copy, and what each corrects.
_CORRECTIONS = {"Tilt": "gantry_tilt", "Eq": "slice_spacing"}
_CORRECTION_RX = re.compile(r"_(Tilt_Eq|Tilt|Eq)(?:_\d+)?(?P<dup>[a-z]?)$")
# The converter prints "Convert <n> DICOM as <path> (<size>)" for each volume
# it writes. Its output also quotes paths and DICOM values: only these counts
# are read from it, and nothing of it is kept or shown.
_CONVERTED_RX = re.compile(rb"^Convert (\d+) DICOM as ", re.MULTILINE)


@dataclass(frozen=True)
class ConvertedVolume:
    """One volume the converter wrote.

    ``path`` is the ``.nii`` or ``.nii.gz`` inside the private folder.
    ``series`` is the series number with the converter's postfix ("3",
    "3a", "5_e2", "3_Tilt_1"); ``base`` the same without a correction
    postfix ("3" for "3_Tilt_1"); ``corrections`` what the volume is a
    corrected copy for (``"gantry_tilt"``, ``"slice_spacing"``), empty for an
    original.
    """
    path: str
    series: Optional[str]
    base: Optional[str] = None
    corrections: Tuple[str, ...] = ()


class Conversion(list):
    """The volumes of one conversion (a list of ConvertedVolume in series
    order), with what the converter reported.

    ``volumes``: the ConvertedVolume items. ``files``: how many DICOM files
    it was given. ``left_out``: how many of them are in no volume
    (screenshots, scouts, dose reports, derived and single images,
    duplicates, files it could not read), or None when its counts could not
    be read. ``status``: its exit status, 0, or not 0 when it failed
    part-way and the volumes are those it wrote.
    """

    def __init__(self, volumes: Iterable[ConvertedVolume] = (), *, files: int = 0,
                 left_out: Optional[int] = None, status: int = 0):
        super().__init__(volumes)
        self.files = files
        self.left_out = left_out
        self.status = status


def find_dcm2niix(scan_folder: Optional[str] = None) -> Optional[str]:
    """The converter's executable: ``NALVERA_DCM2NIIX`` when set to an
    absolute path, else the binary the ``dcm2niix`` Python package installs
    (``dcm2niix.bin``), else ``dcm2niix`` on PATH when it lies outside the
    current folder and outside ``scan_folder`` (default None: the current
    folder only): a ``dcm2niix`` planted beside the scans or in a dataset
    folder is never run. None when there is none. A ``dcm2niix`` package
    that fails to import (no build for this platform) counts as absent;
    nothing is imported or run until a DICOM input needs it."""
    override = os.environ.get("NALVERA_DCM2NIIX")
    if override:
        # A bare name would be looked up in the current folder or on PATH.
        return override if os.path.isabs(override) and _executable(override) else None
    try:
        import dcm2niix  # the PyPI package ships the binary for each platform
    except Exception:
        dcm2niix = None
    if dcm2niix is not None:
        candidate = getattr(dcm2niix, "bin", None)
        if candidate and _executable(str(candidate)):
            return str(candidate)
        folder = os.path.dirname(os.path.abspath(getattr(dcm2niix, "__file__", "") or "."))
        for name in ("dcm2niix.exe", "dcm2niix"):
            candidate = os.path.join(folder, name)
            if _executable(candidate):
                return candidate
    here = os.path.realpath(os.getcwd())
    scans = os.path.realpath(scan_folder) if scan_folder else None

    def untrusted(folder: str) -> bool:
        real = os.path.realpath(folder)
        return real == here or (scans is not None and (
            real == scans or real.startswith(scans.rstrip(os.sep) + os.sep)))

    # Only absolute PATH entries other than the current folder and the
    # scans' folders are searched (on Windows the current folder would
    # otherwise come first), and what is found is checked again after links
    # are resolved.
    safe = [d for d in os.get_exec_path() if d and os.path.isabs(d) and not untrusted(d)]
    found = shutil.which("dcm2niix", path=os.pathsep.join(safe)) if safe else None
    if found is None or untrusted(os.path.dirname(os.path.realpath(found))):
        return None
    return found


def _executable(path: str) -> bool:
    return os.path.isfile(path) and os.access(path, os.X_OK)


def is_dicom_file(path: str) -> bool:
    """True when the file at ``path`` is a DICOM Part 10 file ("DICM" at
    byte 128), or a data set without that preamble whose first elements
    lead to a SOP Class UID. Reads at most 64 KiB. Never raises."""
    try:
        with open(path, "rb") as fh:
            head = fh.read(_HEAD_BYTES)
    except OSError:
        return False
    if head[_DICM_OFFSET:_DICM_OFFSET + 4] == b"DICM":
        return True
    return _bare_dataset(head)


# Explicit-VR types whose length takes 4 bytes after 2 reserved ones.
_LONG_VRS = {b"OB", b"OD", b"OF", b"OL", b"OV", b"OW", b"SQ", b"UC", b"UN", b"UR", b"UT"}
_HEAD_BYTES = 64 * 1024
_SOP_CLASS_UID = (0x0008, 0x0016)


def _bare_dataset(head: bytes) -> bool:
    # A data set written without the preamble (older exports) starts with
    # group 0002 or 0008 in little-endian order, in ascending tag order. Walk
    # its elements, explicit or implicit VR, until the SOP Class UID or past
    # it. A small walk of our own: no warning or log line can repeat a path.
    position = 0
    while position + 8 <= len(head):
        tag = struct.unpack_from("<HH", head, position)
        if tag == _SOP_CLASS_UID:
            return True
        if tag[0] not in (0x0002, 0x0008) or tag > _SOP_CLASS_UID:
            return False
        vr = head[position + 4:position + 6]
        if vr.isalpha() and vr.isupper():
            if vr in _LONG_VRS:
                if position + 12 > len(head):
                    return False
                length = struct.unpack_from("<I", head, position + 8)[0]
                position += 12
            else:
                length = struct.unpack_from("<H", head, position + 6)[0]
                position += 8
        else:
            length = struct.unpack_from("<I", head, position + 4)[0]
            position += 8
        if length == 0xFFFFFFFF:
            return False
        position += length
    return False


def list_dicom_files(folder: str, limit: int = _MAX_FILES) -> List[str]:
    """Every DICOM file under ``folder`` (its subfolders included), in a
    stable order. Hidden files and folders are skipped; links to files are
    followed once, links to folders never (no loops). More than ``limit``
    files (default 100000) raises AnonymizeError(``too_large``)."""
    found: List[str] = []
    seen = set()
    for root, dirs, files in os.walk(folder, followlinks=False):
        dirs[:] = sorted(d for d in dirs if not d.startswith("."))
        for name in sorted(files):
            if name.startswith("."):
                continue
            path = os.path.join(root, name)
            real = os.path.realpath(path)
            if real in seen or not os.path.isfile(real):
                continue
            if is_dicom_file(real):
                seen.add(real)
                found.append(real)
                if len(found) > limit:
                    raise AnonymizeError("too_large", "the folder holds too many files")
    return found


def _series_of(name: str) -> Optional[str]:
    stem = _SUFFIX_RX.sub("", name)
    if not stem.startswith(_PREFIX):
        return None
    series = re.sub(r"[^A-Za-z0-9_]", "", stem[len(_PREFIX):])[:40]
    return series or None


def series_key(series: Optional[str]) -> Tuple[int, str]:
    """The sort key of a ``series`` label: by series number, then by the
    label ("3" < "3_Tilt_1" < "3a" < "12"); a label with no number first."""
    number = re.match(r"\d+", series or "")
    return (int(number.group()) if number else -1, series or "")


def _order(volume: ConvertedVolume):
    return series_key(volume.series) + (volume.path,)


def volume_of(path: str) -> ConvertedVolume:
    """The ConvertedVolume for the converter output at ``path``: its series
    label and corrections read from its file name."""
    series = _series_of(os.path.basename(path))
    base, corrections = series, ()
    match = _CORRECTION_RX.search(series or "")
    if match is not None:
        base = series[:match.start()] + match.group("dup")
        corrections = tuple(_CORRECTIONS[part] for part in match.group(1).split("_"))
    return ConvertedVolume(path=path, series=series, base=base, corrections=corrections)


def choose(volumes: List[ConvertedVolume]) -> Tuple[List[ConvertedVolume],
                                                    List[Tuple[ConvertedVolume,
                                                               ConvertedVolume]]]:
    """Apply the rule for corrected copies to ``volumes`` (one conversion's).

    A volume is dropped when another of the same series (the same ``base``)
    is a copy corrected for more (``corrections`` a strict superset): the
    converter writes a corrected copy beside the original, and each scan
    becomes a paid task. Returns (kept volumes in series order, (dropped,
    the kept volume used in its place) pairs). Echoes, phase images and
    same-numbered series are distinct volumes and all kept.
    """
    kept, dropped = [], []
    for volume in volumes:
        better = [other for other in volumes if other.base == volume.base
                  and set(other.corrections) > set(volume.corrections)]
        if better:
            # The most corrected copy; nothing is corrected for more, so it is kept.
            best = min(better, key=lambda v: (-len(v.corrections),) + _order(v))
            dropped.append((volume, best))
        else:
            kept.append(volume)
    return sorted(kept, key=_order), sorted(dropped, key=lambda pair: _order(pair[0]))


def _left_out(stdout: bytes, files: int) -> Optional[int]:
    counts = [int(m.group(1)) for m in _CONVERTED_RX.finditer(stdout or b"")]
    if not counts:
        return None
    return max(files - sum(counts), 0)


def convert(files: List[str], private_dir: str, scan_folder: Optional[str] = None, *,
            check: Optional[Callable[[str], None]] = None) -> Conversion:
    """Convert the DICOM ``files`` inside ``private_dir``: a private, empty
    folder the caller owns and removes afterwards (also on failure).
    ``scan_folder`` (default None) is where the scans are: a converter found
    there is not run (see ``find_dcm2niix``). ``check`` (default None)
    is called with each file's real path, found again now, before anything
    is linked; it raises to refuse one. The converter reads the checked real
    paths.

    Returns a Conversion: the volumes written (every one, corrected copies
    and originals alike: see ``choose``), how many files are in no volume,
    and the converter's exit status. A converter that fails part-way but
    wrote volumes returns them with its status set.

    Raises AnonymizeError: ``not_a_scan`` (no file given),
    ``dicom_tool_missing``, ``no_volume`` (every file left out: screenshots,
    scouts, derived or single images; ``left_out`` says how many),
    ``conversion_failed`` (it failed and wrote nothing, or did not finish).
    """
    if not files:
        raise AnonymizeError("not_a_scan", "no DICOM file")
    # A file listed minutes ago may have become a link to somewhere else
    # since: each real path is found and checked again here, and that path is
    # the one linked.
    reals = [os.path.realpath(path) for path in files]
    if check is not None:
        for real in reals:
            check(real)
    executable = find_dcm2niix(scan_folder)
    if not executable:
        raise AnonymizeError("dicom_tool_missing")

    # The converter reads a folder of neutral links, so stray files beside
    # the DICOM files are never handed to it, and no original name or folder
    # name can reach an output name.
    source = os.path.join(private_dir, "in")
    target = os.path.join(private_dir, "out")
    os.mkdir(source, 0o700)
    os.mkdir(target, 0o700)
    for index, path in enumerate(reals):
        link = os.path.join(source, f"{index:06d}.dcm")
        try:
            os.symlink(path, link)
        except (OSError, NotImplementedError):
            # Windows without the link privilege: a copy, inside the private folder.
            shutil.copyfile(path, link)

    command = [
        executable,
        "-g", "i",          # ignore the user's defaults file (~/.dcm2nii.ini)
        "-b", "n",          # no sidecar file (it would hold acquisition tags)
        "-z", "i",          # gzip with the built-in zlib, no external pigz
        "-i", "y",          # leave out derived, localizer and 2-D images (screenshots)
        "-f", _PREFIX + "%s",   # the series number only: no patient field in a name
        "-w", "2",          # a name conflict gets a suffix, never an overwrite
        "-o", target,
        source,
    ]
    try:
        # A null input, and its output captured in two pipes: the converter
        # never reads this process's input or writes to its terminal.
        done = subprocess.run(command, stdin=subprocess.DEVNULL, capture_output=True,
                              timeout=_TIMEOUT_SECONDS, check=False)
    except subprocess.TimeoutExpired:
        raise AnonymizeError("conversion_failed", "the converter did not finish") from None
    except OSError as exc:
        raise AnonymizeError("dicom_tool_missing", type(exc).__name__) from None

    volumes = sorted((volume_of(os.path.join(target, name))
                      for name in os.listdir(target) if _SUFFIX_RX.search(name)), key=_order)
    if not volumes:
        # The converter's own output quotes paths and DICOM values: keep it
        # out of the error and give only its exit status.
        if done.returncode == 0:
            raise AnonymizeError("no_volume", f"DICOM files left out: {len(files)}",
                                 left_out=len(files))
        raise AnonymizeError("conversion_failed", f"converter exit status {done.returncode}")
    return Conversion(volumes, files=len(files), left_out=_left_out(done.stdout, len(files)),
                      status=done.returncode)
