import json
from datetime import datetime
from typing import Any
from urllib import request
import re
from pathlib import Path
import tarfile
from utils import is_python_file, get_cleanData
import zipfile
import ast
import asttokens
import vermin

patternNumeric = re.compile("[0-9]+\\.[0-9]+")
patternLiteral = re.compile("cp[0-9]+")

ROOT_DIR = Path(__file__).parent
EXAMPLES_DIR = ROOT_DIR / 'examples'
TMP_DIR = ROOT_DIR / 'tmp'

PYPI_ENDPOINT = "https://pypi.org"

PYTHON_RELEASES = {shortned: actual for shortned, actual in {
    "cp20": "2.0",
    "cp21": "2.1",
    "cp22": "2.2",
    "cp23": "2.3",
    "cp24": "2.4",
    "cp25": "2.5",
    "cp26": "2.6",
    "cp27": "2.7",
    "cp30": "3.0",
    "cp31": "3.1",
    "cp32": "3.2",
    "cp33": "3.3",
    "cp34": "3.4",
    "cp35": "3.5",
    "cp36": "3.6",
    "cp37": "3.7",
    "cp38": "3.8",
    "cp39": "3.9",
    "cp310": "3.10",
    "cp311": "3.11"
}.items()}

PYTHON_VERSIONS = {shortned: actual for shortned, actual in {
    "2.0": set(),
    "2.1": set(),
    "2.2": set(),
    "2.3": set(),
    "2.4": set(),
    "2.5": set(),
    "2.6": set(),
    "2.7": set(),
    "3.0": set(),
    "3.1": set(),
    "3.2": set(),
    "3.3": set(),
    "3.4": set(),
    "3.5": set(),
    "3.6": set(),
    "3.7": set(),
    "3.8": set(),
    "3.9": set(),
    "3.10": set(),
    "3.11": set()
}.items()}


class Release:
    def __init__(self, project_name: str, version: str, files: list[dict[str, Any]],
                 re_download: bool, re_calculate: bool):
        sdist_file = next(file for file in files if file['packagetype'] == "sdist")

        self.project_name = project_name.lower()
        self.version = version
        self.re_download = re_download
        self.re_calculate = re_calculate
        self.filename: str = sdist_file['filename']
        self.requires_python: str = sdist_file['requires_python'] or ''
        self.upload_date = datetime.fromisoformat(sdist_file['upload_time'])
        self.url: str = sdist_file['url']

    def download_files(self):
        out_dir = EXAMPLES_DIR / self.project_name / self.version
        final_dir = ROOT_DIR
        tmp_file = TMP_DIR / self.filename
        request.urlretrieve(self.url, tmp_file)
        # Optimisation: Only keep the Python files
        if tarfile.is_tarfile(tmp_file):
            with tarfile.open(tmp_file) as tar:
                for entry in tar.getmembers():
                    if is_python_file(entry.name):
                        try:
                            tar.extract(entry, final_dir)
                            file=final_dir / entry.name
                            with open(file, encoding="latin-1") as f:
                                cleaned_data = get_cleanData(f.readlines())
                                try:
                                    atok = asttokens.ASTTokens(cleaned_data, parse=True)
                                    for n in ast.walk(atok.tree):
                                        if not isinstance(n, ast.Module) and not isinstance(n,
                                                                                            ast.Name) and not isinstance(
                                            n,
                                            ast.Load) and not isinstance(
                                            n, ast.Store) \
                                                and not isinstance(n, ast.Constant):
                                            py = vermin.version_strings(vermin.detect(atok.get_text(n)))
                                            py2 = py.split(',')[0]
                                            py3 = py.split(',')[1]
                                            if '~' in py2 and '~' in py3:
                                                continue
                                                # PYTHON_VERSIONS['2.0'].add(atok.get_text(n))
                                            elif '!' in py2:
                                                PYTHON_VERSIONS[py3.strip()].add(atok.get_text(n))
                                            elif '3.0' in py3:
                                                PYTHON_VERSIONS[py2.strip()].add(atok.get_text(n))
                                except:
                                    continue
                            file.unlink()
                        except KeyError:
                            continue

        else:
            with zipfile.ZipFile(tmp_file) as tmp_zip:
                for entry in tmp_zip.namelist():
                    if is_python_file(entry):
                        tmp_zip.extract(entry, final_dir)
                # tmp_zip.extractall(out_dir, filter(is_python_file, tmp_zip.namelist()))

        tmp_file.unlink()
        return PYTHON_VERSIONS


class PyPIProject:
    def __init__(self, project_name: str, re_download_releases: bool, re_calculate: bool):
        with request.urlopen(f"{PYPI_ENDPOINT}/pypi/{project_name}/json") as f:
            meta_data = json.load(f)
            self.name = meta_data['info']['name']
            releases = []
            release_version = {}
            for version, files in meta_data['releases'].items():
                try:
                    if len(meta_data['releases'][version]) > 0:
                        python_version_ = meta_data['releases'][version][0]["python_version"]
                        if patternNumeric.match(python_version_) or patternLiteral.match(python_version_):
                            releases.append(Release(self.name, version, files, re_download_releases, re_calculate))
                            if patternLiteral.match(python_version_):
                                python_version_ = PYTHON_RELEASES[python_version_]
                            release_version[version] = \
                                python_version_
                except StopIteration:
                    # Not all releases have a sdist file, skip those
                    continue

            self.releases = releases
            self.releases_version = release_version
