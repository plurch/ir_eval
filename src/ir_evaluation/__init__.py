import tomllib
from pathlib import Path

pyproject_path = Path(__file__).parent / '..' / '..' / "pyproject.toml"

with open(pyproject_path, "rb") as f:
  pyproject_data = tomllib.load(f)

metadata = pyproject_data["project"]
custom_metadata = pyproject_data["tool"]["ir_evaluation"]["metadata"]

__version__ = metadata["version"]
__description__ = metadata["description"]
__author__ = metadata["authors"][0]["name"]
__license__ = custom_metadata["license"]
__url__ = custom_metadata["url"]
