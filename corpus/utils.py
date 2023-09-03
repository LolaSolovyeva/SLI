import json
from datetime import datetime
from typing import Iterable
from urllib import request
from pathlib import Path
from collections import defaultdict
from operator import itemgetter
import vermin as vermin

PYTHON_RELEASES = {version: datetime.fromisoformat(d) for version, d in {
    "2.0": "2000-10-16",
    "2.1": "2001-04-15",
    "2.2": "2001-12-21",
    "2.3": "2003-06-29",
    "2.4": "2004-11-30",
    "2.5": "2006-09-19",
    "2.6": "2008-10-01",
    "2.7": "2010-07-03",
    "3.0": "2008-12-03",
    "3.1": "2009-06-27",
    "3.2": "2011-02-20",
    "3.3": "2012-09-29",
    "3.4": "2014-03-16",
    "3.5": "2015-09-13",
    "3.6": "2016-12-23",
    "3.7": "2018-06-27",
    "3.8": "2019-10-14",
    "3.9": "2020-10-05",
    "3.10": "2021-10-04",
    "3.11": "2022-10-24"
}.items()}


class Config:
    vermin = vermin.Config.parse_file(vermin.Config.detect_config_file())


def parse_vermin_version(version: str):
    if target := vermin.utility.parse_target(version):
        return f"{target[1][0]}.{target[1][1]}"


def is_python_file(path: str) -> bool:
    return Path(path).suffix in {".py", ".py3", ".pyw", ".pyj", ".pyi"}


def sort_features(features):
    """
    :param features: Features to sort
    :return: Return a new dict where features are sorted on version,
    and within each version it is sorted on how often it occurs (descending)
    """
    return {py_v: dict(sorted(features[py_v].items(), key=itemgetter(1), reverse=True)) for py_v in PYTHON_RELEASES}


def create_vector(features):
    result = {py_v: 0 for py_v in PYTHON_RELEASES}
    minversion = find_minversion(features)
    probability = 1 / (len(PYTHON_RELEASES.keys()) - list(PYTHON_RELEASES).index(minversion))
    check = True
    for py_v in result:
        if py_v >= minversion and check:
            check = False
            result[py_v] = probability
        if not check:
            result[py_v] = probability
    return result


def find_minversion(features):
    preliminary = list(features.keys())[0]
    for version in features:
        if len(features[version]) > 0:
            preliminary = version
    return preliminary


def autopct(pct):  # only show the label when it's > 10%
    return ('%.2f' % pct + "%") if pct > 5 else ''


def get_most_popular_projects(n: int, commit_hash: str) -> Iterable[str]:
    """
    See: https://github.com/hugovk/top-pypi-packages, dumps monthly the 5,000 most-downloaded packages from PyPI
    :param commit_hash: The hash of the commit to load the file from (use 'main' for the latest status)
    :param n: Amount of project to return.
    :return: The n most popular projects (of previous) on PyPI.
    """
    url = f"https://raw.githubusercontent.com/hugovk/top-pypi-packages/{commit_hash}/top-pypi-packages-30-days.min.json"
    with request.urlopen(request.Request(url)) as f:
        res = json.load(f)
        return (row['project'] for row in res['rows'][:n])
