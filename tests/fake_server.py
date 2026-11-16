"""A small threaded stand-in for the Nalvera.AI /api/v1 server.

It follows the real server's answers (Platform backend/routes/api_v1.py and
the upload protocol of backend/routes/uploads.py) closely enough to test the
package: resumable uploads with offsets, checksums and Upload-Complete,
every error shape `{"detail", "code", ...}`, idempotent task submits, queue
positions that change between polls, a rejected scan, result downloads, the
estimate by `shape`, the budget `free` (`no_free_tasks`), `cancel_requested`,
`GET /tasks?name=`, and encrypted (NVLT) uploads with the account's
`encryption_required` switch.

Faults are scripted with ``inject()``: a fixed answer (status, body,
headers), a dropped connection, or a body cut short, before or after the
request is handled. Every request is logged in ``requests`` for assertions.
"""
from __future__ import annotations

import base64
import hashlib
import json
import re
import secrets
import sys
import threading
import time
import zlib
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlsplit

ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
GZIP_MAGIC = b"\x1f\x8b"
NVLT_MAGIC = b"NVLT"
_IDEMPOTENCY_RX = re.compile(r"[A-Za-z0-9_-]{1,80}")
_LABEL_RX = re.compile(r"[A-Za-z0-9 ._-]{1,60}")
_LAB_RX = re.compile(r"lab:([1-9][0-9]{0,17})")


def _base62(number: int, width: int) -> str:
    digits = []
    while number:
        number, rem = divmod(number, 62)
        digits.append(ALPHABET[rem])
    return "".join(reversed(digits)).rjust(width, "0")


def make_key() -> str:
    """A well-formed key, built the way backend/api_keys.py builds one
    (independently of the package's own code)."""
    body = "nvk_" + "".join(secrets.choice(ALPHABET) for _ in range(51))
    return body + _base62(zlib.crc32(body.encode("ascii")), 6)


@dataclass
class Fault:
    method: str
    path: str                      # a substring of the path
    status: Optional[int] = None
    body: Any = None
    headers: Dict[str, str] = field(default_factory=dict)
    drop: bool = False             # close the connection without answering
    after: bool = False            # handle the request first, then apply the fault
    truncate: bool = False         # (with after) announce the whole body, send half, close
    times: int = 1
    skip: int = 0                  # let this many matching requests through first


def error_body(code: str, detail: Optional[str] = None, **extra: Any) -> Dict[str, Any]:
    return {"detail": detail or f"server text for {code}", "code": code, **extra}


DEFAULT_MODELS: List[Dict[str, Any]] = [
    {"id": "totalsegmentator_ct", "name": "TotalSegmentator CT", "version": "2.18.0",
     "modality": ["CT"], "type": "img3-seg3", "generative": False, "needs_scan": True,
     "price_eur_per_mvoxel": 0.006,
     "tasks": [
         {"id": "total", "name": "Total: full body", "default": True, "modality": ["CT"],
          "price_eur_per_mvoxel": 0.006, "needs_text": False, "generation": None},
         {"id": "lung_vessels", "name": "Lung vessels", "default": False, "modality": ["CT"],
          "price_eur_per_mvoxel": 0.006, "needs_text": False, "generation": None},
     ]},
    {"id": "merlin", "name": "Merlin", "version": "1", "modality": ["CT"], "type": "img3-txt",
     "generative": False, "needs_scan": True, "price_eur_per_mvoxel": 0.01,
     "tasks": [{"id": "contrastive", "name": "Contrastive", "default": True, "modality": ["CT"],
                "price_eur_per_mvoxel": 0.01, "needs_text": True, "generation": None}]},
    {"id": "nv_generate_ctmr", "name": "NV-Generate", "version": "2", "modality": [],
     "type": "img3-img3", "generative": True, "needs_scan": False, "price_eur_per_mvoxel": 0.01,
     "tasks": [
         {"id": "rflow-ct", "name": "CT", "default": True, "modality": ["CT"],
          "price_eur_per_mvoxel": 0.01, "needs_text": False,
          "generation": {"xy_sizes": [256, 384, 512], "z_sizes": [128, 256],
                         "spacing_mm": {"min": 0.5, "max": 5.0},
                         "modes": ["image-only", "image-mask"],
                         "rules": ["output_size is [X, Y, Z] with X equal to Y"],
                         "defaults": {"num_output_samples": 1, "seed": 0}}},
         {"id": "rflow-mr-brain", "name": "MR brain", "default": False, "modality": ["MR"],
          "price_eur_per_mvoxel": 0.01, "needs_text": False,
          "generation": {"modes": ["image-only", "cross-contrast"], "contrasts": ["t1", "t2"]}},
     ]},
]

# Each GET of a task moves it one step along its script.
DEFAULT_SCRIPT: List[Tuple[str, Optional[int]]] = [
    ("queued", 3), ("queued", 2), ("queued", 1), ("processing", None), ("completed", None)]

# ── The live catalog (``use_catalog()``), for the examples ────────────────────
# Platform models/catalog/registry/*.json as GET /api/v1/models lists them
# (backend/routes/api_v1.py::_model_public): (model id, name, version,
# modality, type, tasks); a task is (id, name, default, its own modality or
# None, price in EUR per million voxels, the result file's extension).
CATALOG: List[Tuple[str, str, str, str, str, List[Tuple[str, str, bool, Optional[str],
                                                         float, str]]]] = [
    ("totalsegmentator_ct", "TotalSegmentator CT", "2.18.0", "CT", "img3-seg3", [
        ("total", "Total — full body", True, None, 0.01, ".nii.gz"),
        ("lung_vessels", "Lung vessels & airways", False, None, 0.007, ".nii.gz"),
        ("body", "Body outline", False, None, 0.007, ".nii.gz"),
        ("cerebral_bleed", "Cerebral bleed", False, None, 0.007, ".nii.gz"),
        ("hip_implant", "Hip implant", False, None, 0.01, ".nii.gz"),
        ("pleural_pericard_effusion", "Pleural / pericardial effusion", False, None, 0.007,
         ".nii.gz"),
        ("head_glands_cavities", "Head glands & cavities", False, None, 0.007, ".nii.gz"),
        ("head_muscles", "Head muscles", False, None, 0.007, ".nii.gz"),
        ("headneck_bones_vessels", "Head/neck bones & vessels", False, None, 0.007, ".nii.gz"),
        ("headneck_muscles", "Head/neck muscles", False, None, 0.007, ".nii.gz"),
        ("liver_vessels", "Liver vessels & tumor", False, None, 0.007, ".nii.gz"),
        ("oculomotor_muscles", "Oculomotor muscles", False, None, 0.007, ".nii.gz"),
        ("lung_nodules", "Lung nodules", False, None, 0.007, ".nii.gz"),
        ("kidney_cysts", "Kidney cysts", False, None, 0.007, ".nii.gz"),
        ("breasts", "Breasts", False, None, 0.007, ".nii.gz"),
        ("liver_segments", "Liver segments (Couinaud)", False, None, 0.007, ".nii.gz"),
        ("craniofacial_structures", "Craniofacial structures", False, None, 0.007, ".nii.gz"),
        ("abdominal_muscles", "Abdominal muscles", False, None, 0.01, ".nii.gz"),
        ("teeth", "Teeth", False, None, 0.007, ".nii.gz"),
        ("trunk_cavities", "Trunk cavities", False, None, 0.007, ".nii.gz"),
        ("ventricle_parts", "Cerebral ventricle parts", False, None, 0.007, ".nii.gz"),
        ("liver_lesions", "Liver lesions", False, None, 0.007, ".nii.gz"),
        ("vertebrae_pp", "Vertebrae — per-vertebra bodies", False, "CT", 0.007, ".nii.gz"),
        ("vertebrae_pp_refined", "Vertebrae — per-vertebra bodies (refined borders)", False,
         "CT", 0.01, ".nii.gz"),
    ]),
    ("totalsegmentator_mr", "TotalSegmentator MRI", "2.18.0", "MRI", "img3-seg3", [
        ("total_mr", "Total — full body", True, None, 0.009, ".nii.gz"),
        ("body_mr", "Body outline", False, None, 0.008, ".nii.gz"),
        ("vertebrae_mr", "Vertebrae", False, None, 0.008, ".nii.gz"),
        ("liver_segments_mr", "Liver segments", False, None, 0.008, ".nii.gz"),
        ("liver_lesions_mr", "Liver lesions", False, None, 0.008, ".nii.gz"),
    ]),
    ("moose_v3_clinical", "MOOSE CT", "3.1.5", "CT", "img3-seg3", [
        ("clin_ct_organs", "Organs", True, None, 0.006, ".nii.gz"),
        ("clin_ct_body", "Body outline", False, None, 0.006, ".nii.gz"),
        ("clin_ct_cardiac", "Cardiac & great vessels", False, None, 0.006, ".nii.gz"),
        ("clin_ct_digestive", "Digestive tract", False, None, 0.006, ".nii.gz"),
        ("clin_ct_lungs", "Lung lobes", False, None, 0.006, ".nii.gz"),
        ("clin_ct_muscles", "Trunk & hip muscles", False, None, 0.006, ".nii.gz"),
        ("clin_ct_peripheral_bones", "Peripheral bones", False, None, 0.006, ".nii.gz"),
        ("clin_ct_ribs", "Ribs & sternum", False, None, 0.006, ".nii.gz"),
        ("clin_ct_vertebrae", "Vertebrae & pelvis", False, None, 0.006, ".nii.gz"),
        ("clin_ct_body_composition", "Body composition", False, None, 0.006, ".nii.gz"),
    ]),
    ("nv_segment_ct", "NV-Segment-CT", "0.6.0", "CT", "img3-seg3", [
        ("everything", "Whole body", True, None, 0.006, ".nii.gz"),
        ("airway", "Airway", False, None, 0.006, ".nii.gz"),
        ("liver_vessels", "Liver vessels", False, None, 0.006, ".nii.gz"),
    ]),
    ("mrsegmentator", "MRSegmentator", "2.0.0", "MRI", "img3-seg3", [
        ("anatomy", "Anatomy (MRI)", True, "MRI", 0.006, ".nii.gz"),
        ("anatomy_ct", "Anatomy (CT)", False, "CT", 0.006, ".nii.gz"),
        ("body_comp", "Body composition", False, "MRI", 0.006, ".nii.gz"),
    ]),
    ("musclemap", "MuscleMap MRI", "2.0", "MRI", "img3-seg3", [
        ("wholebody", "Whole body", True, None, 0.008, ".nii.gz"),
    ]),
    ("nv_generate_ctmr", "MAISI-v2", "1.1", "CT", "img3-img3", [
        ("rflow-ct", "CT generation — image / image + mask", True, "CT", 0.008, ".nii.gz"),
        ("rflow-mr-brain", "MRI brain generation — image / cross-contrast", False, "MRI", 0.008,
         ".nii.gz"),
    ]),
    ("merlin", "Merlin — 3D Abdominal CT Vision-Language", "1.0", "CT", "img3-txt", [
        ("report", "Radiology report", False, None, 0.016, ".txt"),
        ("image_embedding", "Image embedding", False, None, 0.002, ".npy"),
        ("contrastive", "Image–text similarity", False, None, 0.002, ".npy"),
        ("phenotype", "Phenotype findings", True, None, 0.002, ".txt"),
        ("five_year_prediction", "5-year disease risk", False, None, 0.002, ".txt"),
    ]),
    ("nv_reason_ct", "NV-Reason-CT — 3D Chest & Abdominal CT Reasoning", "0.1", "CT",
     "img3-txt", [
         ("chest_report", "Structured chest report", True, None, 0.01, ".txt"),
         ("abdomen_report", "Structured abdominal report", False, None, 0.01, ".txt"),
         ("chest_reasoning", "Chest reasoning walkthrough", False, None, 0.01, ".txt"),
         ("abdomen_reasoning", "Abdominal reasoning walkthrough", False, None, 0.01, ".txt"),
     ]),
]
GENERATIVE_MODELS = ("nv_generate_ctmr",)
#: The kind of result (api_v1 ``output_kind``) by the result file's extension.
OUTPUT_KINDS = {".nii.gz": "volume", ".npy": "vector", ".txt": "text"}

# What a generation accepts: backend/routes/jobs.py (_NVGEN_*), the MR
# contrasts of backend/gen_outputs.py, and api_v1._generation_help, which
# lists them under GET /api/v1/models.
NVGEN_XY_SIZES = [128, 256, 384, 512]
NVGEN_Z_SIZES = [128, 256]
NVGEN_SPACING_MM = {"min": 0.5, "max": 5.0}
NVGEN_CT_MODES = ["image-only", "image-mask"]
NVGEN_MR_MODES = ["image-only", "cross-contrast"]
NVGEN_CT_REGIONS = ["head", "chest", "thorax", "abdomen", "pelvis", "lower"]
NVGEN_CT_TUMOURS = ["hepatic tumor", "bone lesion", "lung tumor", "colon cancer primaries",
                    "pancreatic tumor"]
NVGEN_CT_TUMOURS_BY_REGION = {
    "head": [], "chest": ["lung tumor", "bone lesion"], "thorax": ["lung tumor", "bone lesion"],
    "abdomen": ["hepatic tumor", "pancreatic tumor", "colon cancer primaries", "bone lesion"],
    "pelvis": ["colon cancer primaries", "bone lesion"],
    "lower": ["colon cancer primaries", "bone lesion"],
}
NVGEN_MR_CONTRASTS = ["T1", "T2", "FLAIR", "SWI"]
NVGEN_MR_CFG_DEFAULT = 2.0
#: Limits of backend/schemas.py::GenerateJobRequest that the listing does not name.
GENERATION_LIMITS = {"num_output_samples": (1, 64), "seed": (0, 2147483647),
                     "num_inference_steps": (1, 1000), "cfg_guidance_scale": (0.0, 20.0)}
#: GenerateJobRequest's own defaults for the fields the listing gives none for.
GENERATION_SCHEMA_DEFAULTS = {"body_region": [], "anatomy_list": [], "contrasts": [],
                              "skull_stripped": False, "controllable_anatomy_size": []}
GENERATION_REQUIRED = ("mode", "output_size", "spacing")
#: GET /api/v1/tasks: the accepted ``state`` values and ``limit`` range.
TASK_LIST_STATES = ("active", "finished", "all")
TASK_LIST_LIMIT = (1, 100)


def generation_help(task: str) -> Optional[Dict[str, Any]]:
    """``tasks[].generation`` of GET /api/v1/models (api_v1._generation_help)."""
    base: Dict[str, Any] = {
        "xy_sizes": list(NVGEN_XY_SIZES), "z_sizes": list(NVGEN_Z_SIZES),
        "spacing_mm": dict(NVGEN_SPACING_MM),
        "rules": ["output_size is [X, Y, Z] with X equal to Y",
                  "spacing is [X, Y, Z] in mm with X equal to Y"],
        "defaults": {"num_output_samples": 1, "seed": 0, "num_inference_steps": 30},
    }
    if task == "rflow-ct":
        base.update(modes=list(NVGEN_CT_MODES), body_regions=list(NVGEN_CT_REGIONS),
                    tumours=list(NVGEN_CT_TUMOURS),
                    tumours_by_region={k: list(v) for k, v in NVGEN_CT_TUMOURS_BY_REGION.items()})
        base["rules"] += ["X times its spacing must be at least 256 mm",
                          "image-mask needs one body_region; anatomy_list may name tumors "
                          "available in that region"]
        return base
    if task == "rflow-mr-brain":
        base.update(modes=list(NVGEN_MR_MODES), contrasts=list(NVGEN_MR_CONTRASTS),
                    skull_stripped=[False, True])
        base["defaults"]["cfg_guidance_scale"] = NVGEN_MR_CFG_DEFAULT
        base["rules"] += ["image-only takes exactly one contrast; cross-contrast at least two",
                          "at Z 256 the size must be 256 x 256 x 256"]
        return base
    return None


def catalog_models() -> List[Dict[str, Any]]:
    """The catalog as GET /api/v1/models lists it."""
    out = []
    for model_id, name, version, modality, kind, tasks in CATALOG:
        generative = model_id in GENERATIVE_MODELS
        listed = [{"id": t[0], "name": t[1], "default": t[2], "modality": [t[3] or modality],
                   "price_eur_per_mvoxel": t[4], "needs_text": t[0] == "contrastive",
                   "generation": generation_help(t[0]) if generative else None}
                  for t in tasks]
        default = next(t for t in listed if t["default"])
        out.append({"id": model_id, "name": name, "version": version, "modality": [modality],
                    "type": kind, "generative": generative, "needs_scan": not generative,
                    "price_eur_per_mvoxel": default["price_eur_per_mvoxel"], "tasks": listed})
    return out


def catalog_outputs() -> Dict[Tuple[str, str], str]:
    """(model, task) → the result file's extension."""
    return {(model_id, t[0]): t[5] for model_id, *_, tasks in CATALOG for t in tasks}


def check_generation(task: str, params: Any) -> Tuple[Optional[str], List[str]]:
    """jobs._build_gen_params and schemas.GenerateJobRequest, for the fake:
    (the refusal's explanation or None, the result's parts). Parts follow
    backend/gen_outputs.py: the first image, the masks, the other images; a
    result with one file has none."""
    p = dict(params) if isinstance(params, dict) else {}
    for name in GENERATION_REQUIRED:
        if name not in p:
            return f"params.{name}: Field required", []

    def numbers(value: Any, kind: tuple) -> bool:
        return isinstance(value, list) and len(value) == 3 and all(
            isinstance(v, kind) and not isinstance(v, bool) for v in value)

    if not numbers(p["output_size"], (int,)):
        return "params.output_size: three whole numbers", []
    if not numbers(p["spacing"], (int, float)):
        return "params.spacing: three numbers", []
    for name, (low, high) in GENERATION_LIMITS.items():
        value = p.get(name)
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                                  or not low <= value <= high):
            return f"params.{name}: from {low} to {high}", []
    x, y, z = p["output_size"]
    sx, sy, sz = (float(v) for v in p["spacing"])
    if x != y:
        return "X and Y must be equal.", []
    if x not in NVGEN_XY_SIZES:
        return f"X/Y must be one of {NVGEN_XY_SIZES}.", []
    if z not in NVGEN_Z_SIZES:
        return f"Z must be one of {NVGEN_Z_SIZES}.", []
    if not all(NVGEN_SPACING_MM["min"] <= s <= NVGEN_SPACING_MM["max"] for s in (sx, sy, sz)):
        return "Spacing must be between 0.5 and 5.0 mm.", []
    if sx != sy:
        return "X and Y spacing must be equal.", []
    samples = max(1, int(p.get("num_output_samples") or 1))
    mode = p["mode"]
    if task == "rflow-ct":
        if mode not in NVGEN_CT_MODES:
            return "mode for rflow-ct must be 'image-only' or 'image-mask'.", []
        if x * sx < 256:
            return "CT field of view X·spacing must be ≥ 256 mm.", []
        regions = list(p.get("body_region") or [])
        if any(r not in NVGEN_CT_REGIONS for r in regions):
            return f"body_region must be from {NVGEN_CT_REGIONS}.", []
        masks = 0
        if mode == "image-mask":
            if not regions:
                return "image+mask needs a body_region.", []
            for tumour in p.get("anatomy_list") or []:
                if tumour not in NVGEN_CT_TUMOURS_BY_REGION.get(regions[0], []):
                    return f"{tumour!r} is not available in the {regions[0]!r} region.", []
            masks = samples
        images = samples
    elif task == "rflow-mr-brain":
        if z == 256 and x != 256:
            return "MR brain at Z=256 must be 256×256×256.", []
        if mode not in NVGEN_MR_MODES:
            return "mode for rflow-mr-brain must be 'image-only' or 'cross-contrast'.", []
        contrasts = list(dict.fromkeys(p.get("contrasts") or []))
        if not contrasts:
            return "select at least one MR contrast.", []
        if any(c not in NVGEN_MR_CONTRASTS for c in contrasts):
            return f"contrast must be one of {NVGEN_MR_CONTRASTS}.", []
        if mode == "image-only" and len(contrasts) != 1:
            return "image-only MR generates exactly one contrast.", []
        if mode == "cross-contrast" and len(contrasts) < 2:
            return "cross-contrast MR needs at least two contrasts.", []
        images, masks = samples * len(contrasts), 0
    else:
        return f"model task {task!r} is not a known generative task.", []
    keys = ["image"] + [f"mask{n if n > 1 else ''}" for n in range(1, masks + 1)] \
        + [f"image{n}" for n in range(2, images + 1)]
    return None, keys if len(keys) > 1 else []


class FakeNalvera:
    """The fake server. Change its attributes to set the scene for a test."""

    def __init__(self) -> None:
        self.key = make_key()
        self.email = "Researcher@Lab.example.edu"
        self.scopes = ["tasks:read", "results:read", "tasks:write"]
        self.terms_reaccept_required = False
        self.auth_refusal: Optional[Tuple[int, str, Dict[str, Any]]] = None
        self.limits = {"requests_per_minute": 120, "task_submits_per_minute": 20,
                       "max_active_tasks": 50, "max_uploads_in_progress": 4,
                       "max_file_bytes": 1024 * 1024 * 1024, "chunk_bytes": 32 * 1024 * 1024,
                       "min_poll_seconds": 5}
        self.chunk_bytes = 1000                       # small parts so tests see several
        self.models = json.loads(json.dumps(DEFAULT_MODELS))
        self.minimum_client_version = "0.1.0"
        self.key_expires_at = "2027-11-15T10:00:00Z"
        self.date_header: Optional[str] = None        # the server's clock, when a test sets it
        self.free_tasks_left = 0
        self.free_period = False
        self.personal_credits = 100.0
        self.labs: List[Dict[str, Any]] = []
        self.credits_per_task = 1.25
        self.credits_by_shape: Dict[Tuple[int, ...], float] = {}
        self.cancel_steps = 2                         # polls a running task needs to stop
        # Encryption: off until enable_encryption().
        self.encryption_available = False
        self.encryption_required = False
        self.private_key: Optional[bytes] = None
        self.public_keys: List[bytes] = []
        self.max_active = 50
        self.complete_mode = "inline"                 # or "service": 202 uploaded, then polls
        self.script = list(DEFAULT_SCRIPT)
        self.result_bytes = GZIP_MAGIC + b"segmentation" * 100
        self.result_filename = "job_result_seg.nii.gz"
        self.result_disposition: Optional[str] = None     # a whole header, for tests
        # use_catalog(): result extensions by (model, task), the server's own
        # checks of a generation and of the text a task needs, and result
        # files by (model, task) as (bytes, file name).
        self.outputs: Dict[Tuple[str, str], str] = {}
        self.strict = False
        self.results: Dict[Tuple[str, str], Tuple[bytes, str]] = {}
        self.uploads: Dict[str, Dict[str, Any]] = {}
        self.tasks: Dict[str, Dict[str, Any]] = {}
        self.order: List[str] = []
        self.idempotency: Dict[str, str] = {}
        self.faults: List[Fault] = []
        self.requests: List[Dict[str, Any]] = []
        self.patch_delay = 0.0                         # seconds each PATCH takes (real time)
        self.active_patches = 0
        self.max_active_patches = 0
        self.lock = threading.RLock()
        self._httpd: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self.url = ""

    # ── lifecycle ─────────────────────────────────────────────────────────────
    def start(self) -> FakeNalvera:
        server = self

        class Handler(_Handler):
            fake = server

        class Server(ThreadingHTTPServer):
            def handle_error(self, request, client_address):
                # A client that hung up mid-answer (a test that stops a
                # request on purpose) is no error of the fake.
                if not isinstance(sys.exc_info()[1], (BrokenPipeError, ConnectionResetError)):
                    super().handle_error(request, client_address)

        self._httpd = Server(("127.0.0.1", 0), Handler)
        self._httpd.daemon_threads = True
        self.url = f"http://127.0.0.1:{self._httpd.server_address[1]}"
        self._thread = threading.Thread(target=self._httpd.serve_forever, args=(0.02,),
                                        daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()

    # ── scene helpers ─────────────────────────────────────────────────────────
    def inject(self, method: str, path: str, **kwargs: Any) -> Fault:
        fault = Fault(method=method, path=path, **kwargs)
        with self.lock:
            self.faults.append(fault)
        return fault

    def enable_encryption(self, required: bool = False) -> bytes:
        """Give the platform a key pair (the fake keeps the private key) and
        set the account's switch. Returns the public key."""
        import nvlt_reference
        private, public = nvlt_reference.new_key_pair()
        self.private_key, self.public_keys = private, [public]
        self.encryption_available, self.encryption_required = True, required
        return public

    def _encryption(self) -> Dict[str, Any]:
        return {"available": self.encryption_available, "required": self.encryption_required,
                "format": "NVLT/2",
                "keys": [{"public_key": base64.b64encode(k).decode("ascii"),
                          "sha256": hashlib.sha256(k).hexdigest()} for k in self.public_keys]}

    def use_catalog(self) -> None:
        """List the live catalog's models (``CATALOG``) and check requests as
        the real server does: generation settings, the text a task needs."""
        self.models = catalog_models()
        self.outputs = catalog_outputs()
        self.strict = True

    def add_lab(self, lab_id: int, name: str, balance: float, status: str = "active",
                spending_limit: Optional[float] = None) -> None:
        self.labs.append({"id": lab_id, "name": name, "balance": balance, "member_status": status,
                          "spending_limit": spending_limit, "spent": 0.0})

    def calls(self, method: Optional[str] = None, path: Optional[str] = None) -> List[Dict]:
        with self.lock:
            return [r for r in self.requests if (method is None or r["method"] == method)
                    and (path is None or path in r["path"])]

    def add_task(self, state: str = "completed", **fields: Any) -> str:
        """A task made elsewhere (the website), for listing and downloads."""
        with self.lock:
            task_id = secrets.token_hex(6)
            self.tasks[task_id] = self._task_record(task_id, model="totalsegmentator_ct",
                                                    task="total", name="website-scan.nii.gz",
                                                    source="web", script=[(state, None)],
                                                    **fields)
            self.order.append(task_id)
            return task_id

    # ── state ─────────────────────────────────────────────────────────────────
    def _budgets(self) -> Dict[str, Any]:
        active = [lab for lab in self.labs if lab["member_status"] == "active"]
        best = min(active, key=lambda lab: (-lab["balance"], lab["name"])) if active else None
        return {"free": {"period_active": self.free_period,
                         "free_tasks_left": self.free_tasks_left},
                "personal": {"monthly_left": 0.0, "topup": self.personal_credits,
                             "available_credits": self.personal_credits,
                             "period_end": "2026-12-01T00:00:00Z"},
                "labs": [dict(lab) for lab in self.labs],
                "suggested": f"lab:{best['id']}" if best else "personal"}

    def _task_record(self, task_id: str, *, model: str, task: str, name: str, source: str,
                     script: Optional[List] = None, **fields: Any) -> Dict[str, Any]:
        record = {"id": task_id, "name": name, "model": model, "model_version": "1",
                  "task": task, "task_label": task, "source": source, "step": 0,
                  "script": list(script or self.script), "created_at": "2026-11-15T10:00:00Z",
                  "error": None, "upload": None, "budget": None, "charged": None,
                  "cancel_requested": False}
        record.update(fields)
        return record

    def _task_object(self, record: Dict[str, Any]) -> Dict[str, Any]:
        state, position = record["script"][min(record["step"], len(record["script"]) - 1)]
        finished = state in ("completed", "failed", "cancelled")
        return {"id": record["id"], "name": record["name"], "model": record["model"],
                "model_version": record["model_version"], "task": record["task"],
                "task_label": record["task_label"], "state": state,
                "queue_position": position if state == "queued" else None,
                "created_at": record["created_at"],
                "started_at": "2026-11-15T10:01:00Z" if state != "queued" else None,
                "completed_at": "2026-11-15T10:03:00Z" if finished else None,
                "waited_seconds": 12.0 * (record["step"] + 1),
                "running_seconds": 30.0 if state in ("processing", "completed") else None,
                "error": record["error"] if state == "failed" else None,
                "output_ready": state == "completed",
                "output_kind": record.get("output_kind") or "volume",
                "parts": (record.get("parts") or None) if state == "completed" else None,
                "source": record["source"],
                "cancel_requested": bool(record.get("cancel_requested"))}

    def _advance(self, record: Dict[str, Any]) -> None:
        state = record["script"][min(record["step"], len(record["script"]) - 1)][0]
        if state not in ("completed", "failed", "cancelled"):
            record["step"] = min(record["step"] + 1, len(record["script"]) - 1)

    def _state(self, record: Dict[str, Any]) -> str:
        return record["script"][min(record["step"], len(record["script"]) - 1)][0]

    def _upload_status(self, up: Dict[str, Any]) -> Dict[str, Any]:
        status = {"upload_id": up["id"], "kind": "nifti", "policy": "raw", "state": up["state"],
                  "size": up["size"], "offset": len(up["data"])}
        if up["state"] in ("processing", "uploaded"):
            status["poll_after_ms"] = 1000 if up["state"] == "processing" else 2000
        if up["state"] == "ready":
            status["receipt"] = {"valid": True, "shape": up.get("shape") or [4, 4, 4]}
        if up["state"] == "rejected":
            status["receipt"] = {"valid": False, "error": up["error"]}
            status["error"] = up["error"]
        return status

    def _process(self, up: Dict[str, Any]) -> None:
        data = bytes(up["data"])
        if data.startswith(NVLT_MAGIC):
            # Decrypted at ingest, as backend/vault_crypto.py does; the
            # plaintext then passes every normal check.
            import nvlt_reference
            try:
                if self.private_key is None:
                    raise nvlt_reference.NotDecryptable("no key")
                data = nvlt_reference.decrypt(data, self.private_key)
            except nvlt_reference.NotDecryptable:
                up["state"], up["error"] = "rejected", "The encrypted file could not be read."
                return
            up["plaintext"] = data
            if not data.startswith(GZIP_MAGIC):
                up["state"], up["error"] = "rejected", "The file is not gzip-compressed."
                return
        if b"REJECT" in data:
            up["state"], up["error"] = "rejected", "This file holds a label mask, not a scan."
        else:
            up["state"] = "ready"
            up["shape"] = shape_marker(data)


def shape_marker(data: bytes) -> List[int]:
    """The voxel grid a test wrote into a scan as ``SHAPE=AxBxC;`` (the
    stand-in anonymizer reads the same marker), else [4, 4, 4]."""
    match = re.search(rb"SHAPE=(\d+)x(\d+)x(\d+);", data)
    return [int(g) for g in match.groups()] if match else [4, 4, 4]


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    fake: FakeNalvera

    def log_message(self, *args: Any) -> None:       # quiet
        pass

    def do_GET(self) -> None:
        self._handle("GET")

    def do_POST(self) -> None:
        self._handle("POST")

    def do_PATCH(self) -> None:
        self._handle("PATCH")

    def do_HEAD(self) -> None:
        self._handle("HEAD")

    def do_DELETE(self) -> None:
        self._handle("DELETE")

    # ── plumbing ──────────────────────────────────────────────────────────────
    def _handle(self, method: str) -> None:
        fake = self.fake
        parts = urlsplit(self.path)
        path, query = parts.path, parse_qs(parts.query)
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else b""
        with fake.lock:
            fake.requests.append({"method": method, "path": path, "query": parts.query,
                                  "headers": dict(self.headers.items()), "body_length": len(body)})
            fault = next((f for f in fake.faults if f.method == method and f.path in path
                          and f.times > 0), None)
            if fault is not None and fault.skip > 0:
                fault.skip -= 1
                fault = None
            elif fault is not None:
                fault.times -= 1
        if fault is not None and not fault.after:
            return self._apply(fault)
        if method == "PATCH" and fake.patch_delay:
            with fake.lock:
                fake.active_patches += 1
                fake.max_active_patches = max(fake.max_active_patches, fake.active_patches)
            time.sleep(fake.patch_delay)
            with fake.lock:
                fake.active_patches -= 1
        status, payload, headers = self._route(method, path, query, body)
        if fault is not None and fault.truncate:
            return self._send_short(status, payload, headers)
        if fault is not None:
            return self._apply(fault)
        self._send(status, payload, headers, head=method == "HEAD")

    def _send_short(self, status: int, payload: Any, headers: Dict[str, str]) -> None:
        """Announce the whole body, send half of it, close the connection."""
        data = bytes(payload) if isinstance(payload, (bytes, bytearray)) else \
            json.dumps(payload).encode()
        self.send_response(status)
        for name, value in headers.items():
            self.send_header(name, value)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data[:len(data) // 2])
        self.wfile.flush()
        self.close_connection = True

    def _apply(self, fault: Fault) -> None:
        if fault.drop:
            self.close_connection = True
            try:
                self.connection.close()
            except OSError:
                pass
            return
        self._send(fault.status or 500, fault.body, fault.headers)

    def _send(self, status: int, payload: Any, headers: Optional[Dict[str, str]] = None,
              head: bool = False) -> None:
        headers = dict(headers or {})
        if isinstance(payload, (bytes, bytearray)):
            data = bytes(payload)
            headers.setdefault("Content-Type", "application/octet-stream")
        elif payload is None:
            data = b""
        else:
            data = json.dumps(payload).encode("utf-8")
            headers.setdefault("Content-Type", "application/json")
        self.send_response_only(status)
        self.send_header("Server", self.version_string())
        headers.setdefault("Date", self.fake.date_header or self.date_time_string())
        headers.setdefault("Cache-Control", "no-store")
        for name, value in headers.items():
            self.send_header(name, value)
        self.send_header("Content-Length", str(0 if head else len(data)))
        self.end_headers()
        if not head and data:
            self.wfile.write(data)

    # ── routes ────────────────────────────────────────────────────────────────
    def _route(self, method: str, path: str, query: Dict[str, List[str]], body: bytes):
        fake = self.fake
        auth = self.headers.get("Authorization", "")
        email = self.headers.get("X-Nalvera-Email", "")
        if auth != f"Bearer {fake.key}" or email.strip().lower() != fake.email.lower():
            return 401, error_body("invalid_credentials"), {}
        if fake.auth_refusal is not None:
            status, code, extra = fake.auth_refusal
            return status, error_body(code, **extra), {}
        with fake.lock:
            return self._dispatch(method, path, query, body)

    def _need(self, scope: str):
        if scope not in self.fake.scopes:
            return 403, error_body("scope_missing", required_scope=scope), {}
        return None

    def _json(self, body: bytes) -> Dict[str, Any]:
        try:
            data = json.loads(body or b"{}")
        except ValueError:
            data = None
        return data if isinstance(data, dict) else {}

    def _dispatch(self, method: str, path: str, query: Dict[str, List[str]], body: bytes):
        fake = self.fake
        if method == "GET" and path == "/api/v1/me":
            return 200, {"email": fake.email.lower(), "plan": "tier1", "plan_label": "Guest",
                         "key": {"prefix": fake.key[:12], "name": "laptop", "scopes": fake.scopes,
                                 "expires_at": fake.key_expires_at},
                         "limits": dict(fake.limits),
                         "minimum_client_version": fake.minimum_client_version,
                         "terms_reaccept_required": fake.terms_reaccept_required,
                         "encryption": fake._encryption()}, {}
        if method == "GET" and path == "/api/v1/models":
            return 200, {"models": fake.models}, {}
        if method == "GET" and path == "/api/v1/budgets":
            return 200, fake._budgets(), {}
        if method == "POST" and path == "/api/v1/estimate":
            data = self._json(body)
            budgets = fake._budgets()
            model = next((m for m in fake.models if m["id"] == data.get("model")), None)
            if model is None:
                return 400, error_body("invalid_request", detail="No model with that id."), {}
            credits = fake.credits_per_task
            shape = data.get("shape")
            if shape is not None:
                # api_v1._credits: a 2-3 axis grid of whole numbers 1..100000,
                # never for a generation.
                if model["generative"] or not isinstance(shape, list) \
                        or not 2 <= len(shape) <= 3 \
                        or any(isinstance(n, bool) or not isinstance(n, int)
                               or not 1 <= n <= 100000 for n in shape):
                    return 400, error_body("invalid_request"), {}
                credits = fake.credits_by_shape.get(tuple(shape), fake.credits_per_task)
            elif data.get("upload_id"):
                if data["upload_id"] not in fake.uploads:
                    return 404, error_body("not_found"), {}
            elif not model["generative"]:
                return 400, error_body("invalid_request"), {}
            elif fake.strict:
                task = data.get("task") or next(t["id"] for t in model["tasks"] if t["default"])
                refusal, _parts = check_generation(task, data.get("params"))
                if refusal:
                    return 400, error_body("invalid_request", detail=refusal), {}
            options = [{"budget": "personal", "name": "Personal",
                        "available_credits": fake.personal_credits,
                        "sufficient": fake.personal_credits >= credits}]
            for lab in fake.labs:
                if lab["member_status"] == "active":
                    options.append({"budget": f"lab:{lab['id']}", "name": lab["name"],
                                    "available_credits": lab["balance"],
                                    "sufficient": lab["balance"] >= credits})
            free_source = "global" if fake.free_period else (
                "wallet" if fake.free_tasks_left else None)
            return 200, {"credits": credits,
                         "free": {"available": free_source is not None, "source": free_source,
                                  "free_tasks_left": fake.free_tasks_left},
                         "budgets": options, "suggested": budgets["suggested"]}, {}
        if path == "/api/v1/uploads" or path.startswith("/api/v1/uploads/"):
            refused = self._need("tasks:write")
            return refused or self._uploads(method, path, body)
        if method == "POST" and path == "/api/v1/tasks":
            refused = self._need("tasks:write")
            return refused or self._submit(body)
        if method == "GET" and path == "/api/v1/tasks":
            return self._list(query)
        match = re.fullmatch(r"/api/v1/tasks/([0-9a-f]+)(/cancel|/result)?", path)
        if match:
            record = fake.tasks.get(match.group(1))
            if record is None:
                return 404, error_body("not_found"), {}
            action = match.group(2)
            if method == "GET" and action is None:
                fake._advance(record)
                return 200, fake._task_object(record), {}
            if method == "POST" and action == "/cancel":
                refused = self._need("tasks:write")
                if refused:
                    return refused
                current = fake._state(record)
                if current in ("completed", "failed", "cancelled"):
                    return 409, error_body("not_cancellable"), {}
                if not record["cancel_requested"]:
                    record["cancel_requested"] = True
                    # A running task stops at the worker's next check: it stays
                    # "processing" for a few polls (api_v1._task).
                    record["script"] = ([("processing", None)] * fake.cancel_steps
                                        if current == "processing" else []) \
                        + [("cancelled", None)]
                    record["step"] = 0
                return 200, fake._task_object(record), {}
            if method == "GET" and action == "/result":
                refused = self._need("results:read")
                if refused:
                    return refused
                state = fake._state(record)
                if state != "completed":
                    return 409, error_body("not_ready", state=state), {}
                part = (query.get("part") or [None])[0]
                if record.get("parts"):
                    return self._parts_result(record, part)
                data, name = record.get("result") or fake.results.get(
                    (record["model"], record["task"]), (fake.result_bytes, fake.result_filename))
                name = name if not part else f"job_result_{part}.nii.gz"
                return 200, data, {
                    "Content-Disposition": fake.result_disposition
                    or f'attachment; filename="{name}"',
                    "Content-Type": "application/gzip"}
        if method == "GET":
            return 404, error_body("not_found"), {}
        return 405, error_body("invalid_request"), {}

    def _uploads(self, method: str, path: str, body: bytes):
        fake = self.fake
        if method == "POST" and path == "/api/v1/uploads":
            data = self._json(body)
            filename, size = data.get("filename"), data.get("size")
            if not isinstance(filename, str) or not filename.endswith((".nii.gz",
                                                                       ".nii.gz.nvlt")):
                return 400, error_body("invalid_request"), {}
            if not isinstance(size, int) or size <= 0:
                return 400, error_body("invalid_request", errors=[{"field": "size"}]), {}
            if size > fake.limits["max_file_bytes"]:
                return 413, error_body("too_large", max_bytes=fake.limits["max_file_bytes"]), {}
            live = [u for u in fake.uploads.values() if u["state"] == "receiving"]
            if len(live) >= fake.limits["max_uploads_in_progress"]:
                return 429, error_body("too_many_uploads", retry_after=3,
                                       limit=fake.limits["max_uploads_in_progress"]), \
                    {"Retry-After": "3"}
            upload_id = secrets.token_hex(8)
            fake.uploads[upload_id] = {"id": upload_id, "filename": filename, "size": size,
                                       "data": bytearray(), "state": "receiving", "error": None}
            return 201, {"upload_id": upload_id, "state": "receiving", "offset": 0,
                         "chunk_bytes": fake.chunk_bytes, "expires_after_secs": 21600}, {}
        match = re.fullmatch(r"/api/v1/uploads/([0-9a-f]{16})(/complete)?", path)
        up = fake.uploads.get(match.group(1)) if match else None
        if up is None:
            return 404, error_body("not_found"), {}
        if method == "HEAD":
            return 204, None, {"Upload-Offset": str(len(up["data"])),
                               "Upload-Length": str(up["size"]), "Upload-State": up["state"]}
        if method == "GET" and not match.group(2):
            if up["state"] == "uploaded":
                up["state"] = "processing"
            elif up["state"] == "processing":
                fake._process(up)
            return 200, fake._upload_status(up), {}
        if method == "DELETE":
            cancelled = up["state"] in ("receiving", "uploaded", "processing", "ready")
            if cancelled:
                up["state"] = "cancelled"
            return 200, {"cancelled": cancelled}, {}
        if method == "PATCH":
            if up["state"] != "receiving":
                return 404, error_body("not_found"), {}
            if self.headers.get("Content-Type") != "application/offset+octet-stream":
                return 415, error_body("invalid_request"), {}
            offset = int(self.headers.get("Upload-Offset", "-1"))
            committed = len(up["data"])
            if offset == 0 and committed == 0 and fake.encryption_required \
                    and body[:4] != NVLT_MAGIC:
                # The account's switch: checked on the first bytes, before any
                # byte is stored; the upload session is cancelled.
                up["state"] = "cancelled"
                return 422, error_body("encryption_required"), {}
            if offset != committed:
                return 409, error_body("offset_mismatch", offset=committed), {}
            checksum = self.headers.get("Upload-Checksum")
            if checksum is not None:
                algo, _, b64 = checksum.partition(" ")
                if algo != "sha256" or base64.b64decode(b64) != hashlib.sha256(body).digest():
                    return 460, error_body("checksum_mismatch", offset=committed), {}
            if committed + len(body) > up["size"]:
                return 413, error_body("too_large", max_bytes=fake.limits["max_file_bytes"]), {}
            if committed == 0 and body[:2] != GZIP_MAGIC and body[:4] != NVLT_MAGIC:
                up["state"], up["error"] = "rejected", "The file is not gzip-compressed."
                return 422, error_body("scan_rejected", reason=up["error"]), {}
            up["data"] += body
            if self.headers.get("Upload-Complete") == "?1" and len(up["data"]) == up["size"]:
                return self._complete(up)
            return 200, {"offset": len(up["data"])}, {}
        if method == "POST" and match.group(2):
            if up["state"] != "receiving":
                if up["state"] == "uploaded":
                    return 202, {"upload_id": up["id"], "state": "uploaded",
                                 "poll_after_ms": 2000}, {}
                return 200, fake._upload_status(up), {}
            if len(up["data"]) < up["size"]:
                return 409, error_body("offset_mismatch", offset=len(up["data"])), {}
            return self._complete(up)
        return 405, error_body("invalid_request"), {}

    def _complete(self, up: Dict[str, Any]):
        fake = self.fake
        if fake.complete_mode == "service":
            up["state"] = "uploaded"
            return 202, {"upload_id": up["id"], "state": "uploaded", "poll_after_ms": 2000}, {}
        fake._process(up)
        return 200, fake._upload_status(up), {}

    def _parts_result(self, record: Dict[str, Any], part: Optional[str]):
        """A generation with several files: one ZIP, or one file with ?part=
        (jobs.download_result through gen_outputs.list_output_parts)."""
        import io
        import zipfile
        stem = f"NV-Generate_{record['task']}_{record['id']}"
        if part is not None:
            if part not in record["parts"]:
                return 404, error_body("not_found"), {}
            return 200, self.fake.result_bytes, {
                "Content-Disposition": f'attachment; filename="{stem}_{part}.nii.gz"',
                "Content-Type": "application/gzip"}
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            for key in record["parts"]:
                archive.writestr(f"{stem}_{key}.nii.gz", self.fake.result_bytes)
        return 200, buffer.getvalue(), {
            "Content-Disposition": f'attachment; filename="{stem}.zip"',
            "Content-Type": "application/zip"}

    def _submit(self, body: bytes):
        fake = self.fake
        key = self.headers.get("Idempotency-Key", "")
        if not _IDEMPOTENCY_RX.fullmatch(key):
            return 400, error_body("invalid_request"), {}
        data = self._json(body)
        if key in fake.idempotency:
            record = fake.tasks[fake.idempotency[key]]
            return 202, {"task": fake._task_object(record), "charged": record["charged"],
                         "idempotent_replay": True}, {}
        model = next((m for m in fake.models if m["id"] == data.get("model")), None)
        if model is None:
            return 400, error_body("invalid_request", detail="No model with that id."), {}
        task_ids = [t["id"] for t in model["tasks"]]
        task = data.get("task") or next(t["id"] for t in model["tasks"] if t["default"])
        if task not in task_ids:
            return 400, error_body("invalid_request", detail="No task with that id."), {}
        label = data.get("label")
        if label is not None and not _LABEL_RX.fullmatch(label.strip()):
            return 400, error_body("invalid_request"), {}
        budget = data.get("budget") if data.get("budget") is not None else "personal"
        lab = None
        if budget not in ("personal", "free"):
            if not isinstance(budget, str) or not _LAB_RX.fullmatch(budget):
                return 400, error_body("invalid_request"), {}
            lab = next((x for x in fake.labs if f"lab:{x['id']}" == budget
                        and x["member_status"] == "active"), None)
            if lab is None:
                return 403, error_body("lab_not_allowed"), {}
        parts: List[str] = []
        if data.get("upload_id"):
            up = fake.uploads.get(data["upload_id"])
            if up is None or up["state"] in ("consumed", "cancelled"):
                return 404, error_body("not_found"), {}
            if up["state"] in ("receiving", "uploaded", "processing"):
                return 409, error_body("not_ready", state=up["state"]), {}
            if up["state"] == "rejected":
                return 422, error_body("scan_rejected", reason=up["error"]), {}
            if fake.strict and model["generative"]:
                return 400, error_body("invalid_request"), {}
            if fake.strict and task == "contrastive" and not (data.get("text") or "").strip():
                # jobs.submit_scan_task: the contrastive task needs its query.
                return 400, error_body("invalid_request"), {}
        elif data.get("params") is None:
            return 400, error_body("invalid_request"), {}
        elif not isinstance(data["params"].get("output_size"), list):
            return 400, error_body("invalid_request",
                                   detail="params.output_size: Field required"), {}
        elif fake.strict:
            refusal, parts = check_generation(task, data["params"])
            if refusal:
                return 400, error_body("invalid_request", detail=refusal), {}
        active = sum(1 for r in fake.tasks.values() if r["source"] == "api"
                     and fake._state(r) in ("queued", "processing"))
        if active >= fake.max_active:
            return 429, error_body("too_many_active_tasks", retry_after=60,
                                   limit=fake.max_active), {"Retry-After": "60"}
        credits_due = fake.credits_per_task
        if data.get("upload_id"):
            shape = tuple((fake.uploads[data["upload_id"]].get("shape") or [4, 4, 4])[:3])
            credits_due = fake.credits_by_shape.get(shape, fake.credits_per_task)
        if fake.free_period or fake.free_tasks_left > 0:
            if not fake.free_period:
                fake.free_tasks_left -= 1
            charged = {"source": "free", "lab_id": None, "credits": 0.0}
        elif budget == "free":
            # api_v1: nothing queued, nothing charged, the upload is kept.
            return 402, error_body("no_free_tasks", budget="free"), {}
        elif lab is not None:
            if lab["balance"] < credits_due:
                return 402, error_body("insufficient_credits", budget=budget), {}
            lab["balance"] -= credits_due
            charged = {"source": "lab", "lab_id": lab["id"], "credits": credits_due}
        else:
            if fake.personal_credits < credits_due:
                return 402, error_body("insufficient_credits", budget=budget), {}
            fake.personal_credits -= credits_due
            charged = {"source": "personal", "lab_id": None, "credits": credits_due}
        if data.get("upload_id"):
            fake.uploads[data["upload_id"]]["state"] = "consumed"
            name = label.replace(" ", "_") + ".nii.gz" if label else (
                data.get("name") or f"upload-{data['upload_id']}.nii.gz")
            if name.endswith(".nvlt"):
                name = name[:-len(".nvlt")]
        else:
            # jobs.queue_generation: the label as given, else a name of its own.
            name = label.strip() if label else \
                f"NV-Generate {task} ({data['params'].get('mode')})"
        task_id = secrets.token_hex(6)
        extension = fake.outputs.get((model["id"], task), ".nii.gz")
        record = fake._task_record(task_id, model=model["id"], task=task, name=name,
                                   source="api", budget=budget, charged=charged,
                                   upload=data.get("upload_id"), body=data,
                                   output_kind=OUTPUT_KINDS.get(extension, "file"), parts=parts)
        fake.tasks[task_id] = record
        fake.order.append(task_id)
        fake.idempotency[key] = task_id
        return 202, {"task": fake._task_object(record), "charged": charged,
                     "idempotent_replay": False}, {}

    def _list(self, query: Dict[str, List[str]]):
        fake = self.fake
        state = (query.get("state") or ["all"])[0]
        limit = int((query.get("limit") or ["50"])[0])
        offset = int((query.get("offset") or ["0"])[0])
        name = (query.get("name") or [None])[0]
        if state not in TASK_LIST_STATES or not TASK_LIST_LIMIT[0] <= limit <= TASK_LIST_LIMIT[1]:
            return 400, error_body("invalid_request"), {}
        newest = list(reversed(fake.order))
        chosen = []
        for task_id in newest:
            record = fake.tasks[task_id]
            if name is not None and record["name"] != name:
                continue                       # an exact match, no wildcard
            current = fake._state(record)
            if state == "active" and current not in ("queued", "processing"):
                continue
            if state == "finished" and current not in ("completed", "failed", "cancelled"):
                continue
            chosen.append(record)
        page = chosen[offset:offset + limit]
        for record in page:
            fake._advance(record)
        return 200, {"tasks": [fake._task_object(r) for r in page], "total": len(chosen)}, {}
