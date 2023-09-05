# This is a sample Python script.
import json

from pypi_crawler import PyPIProject
from utils import get_most_popular_projects
from pathlib import Path

ROOT_DIR = Path(__file__).parent


# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def main():
    projects = get_most_popular_projects(1, "fa998b797a5300a240e2b4c042f9a438ab91c7f5")
    result = {}
    for project_name in projects:
        project = PyPIProject(project_name, True, True)
        if len(project.releases) > 0:
            for release in project.releases:
                result = release.download_files()

    for version in result:
        result[version] = list(result[version])
    json_object = json.dumps(result, indent=4)

    # Writing to sample.json
    with open("all_data.json", "w") as outfile:
        outfile.write(json_object)


if __name__ == '__main__':
    main()
