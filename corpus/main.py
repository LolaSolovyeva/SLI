# This is a sample Python script.
import json

from pypi_crawler import PyPIProject
from utils import get_most_popular_projects
from pathlib import Path

ROOT_DIR = Path(__file__).parent


# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def main():
    projects = get_most_popular_projects(50, "fa998b797a5300a240e2b4c042f9a438ab91c7f5")
    for project_name in projects:
        project = PyPIProject(project_name, True, True)
        if len(project.releases) > 0:
            for release in project.releases:
                release.download_files(project.releases_version)


if __name__ == '__main__':
    main()
