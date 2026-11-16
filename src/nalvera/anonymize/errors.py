"""The anonymizer's one exception, and the lookup of its texts."""
from __future__ import annotations

import importlib
from typing import Optional

# English texts by code, used when the package's messages module has no text
# of its own under "anonymize.<code>" (it is the one place a text editor
# changes wording; this keeps the module usable on its own).
_TEXT = {
    "not_found": "No file or folder was found at that path. Check the path.",
    "not_a_scan": "This is not a NIfTI file (.nii or .nii.gz) or a folder of DICOM files.",
    "unreadable": "The file could not be read as a NIfTI scan. Check that the file is "
                  "complete.",
    "too_large": "The scan is larger than the upload limit.",
    "nifti2_unsupported": "The platform accepts NIfTI-1 only, and this NIfTI-2 file cannot be "
                          "converted to NIfTI-1 exactly. Save it as NIfTI-1, then try again.",
    "dicom_tool_missing": "The DICOM converter dcm2niix was not found. Install it separately "
                          "with: pip install dcm2niix (or from "
                          "https://github.com/rordenlab/dcm2niix). NIfTI files work without "
                          "it.",
    "conversion_failed": "The DICOM folder could not be converted to a NIfTI file. Check that "
                         "its files are complete.",
    "no_volume": "The DICOM folder holds no image that can become a scan. Screenshots, dose "
                 "reports, scouts, derived images and single images are left out.",
    "mixed_folder": "The folder holds NIfTI files and DICOM files. Keep them in separate "
                    "folders, or pass the NIfTI files one by one.",
    "verification_failed": "The anonymized file did not pass its check and was deleted. "
                           "Nothing was uploaded.",
    "invalid_label": "A label has 1 to 60 letters, digits, spaces, dots, dashes or underscores.",
    "mapping_unreadable": "The scan record is not one this package wrote, so it was "
                          "left unchanged. Choose another file.",
}


def text(key: str, default: str) -> str:
    """The package's text for ``key`` from ``nalvera.messages.MESSAGES``, or
    ``default`` when that module or key does not exist.

    ``key``: a message key such as ``"anonymize.not_found"``.
    ``default``: the text returned when the package has none for ``key``.
    """
    try:
        messages = importlib.import_module("nalvera.messages")
        value = getattr(messages, "MESSAGES", {}).get(key)
    except Exception:
        return default
    return value if isinstance(value, str) and value else default


class AnonymizeError(Exception):
    """Raised by ``nalvera.anonymize``.

    ``code`` is stable and is what callers branch on: ``not_found``,
    ``not_a_scan``, ``unreadable``, ``too_large``, ``nifti2_unsupported``,
    ``dicom_tool_missing``, ``conversion_failed``, ``no_volume``,
    ``mixed_folder``, ``verification_failed``, ``invalid_label``, and
    ``mapping_unreadable`` from ``Mapping``.
    ``detail`` is a fixed technical reason chosen by this module, kept on the
    exception and out of the sentence a person reads (``str(error)``).
    Neither the message nor ``detail`` ever holds header text, DICOM values,
    converter output or file contents.
    ``left_out`` (``no_volume`` only, else None) is the number of DICOM files
    the converter left out: screenshots, scouts, dose reports, derived and
    single images, files it could not read.

    Arguments: ``code`` (one of the codes above), ``detail`` (default None),
    ``left_out`` (keyword only, default None).
    """

    def __init__(self, code: str, detail: Optional[str] = None, *,
                 left_out: Optional[int] = None):
        self.code = code
        self.detail = detail
        self.left_out = left_out
        super().__init__(text("anonymize." + code, _TEXT.get(code, code)))
