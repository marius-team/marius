from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlretrieve
import tarfile
import zipfile
from zipfile import ZipFile
import gzip
import shutil
import os


def download_url(url, output_dir, overwrite):
    output_dir = Path(output_dir)

    url_components = urlparse(url)
    filename = Path(url_components.path + url_components.query).name
    filepath = output_dir / filename

    if filepath.is_file() and not overwrite:
        print(f"File already exists: {filepath}")
    else:
        try:
            print(f"Downloading {filename} to {filepath}")
            urlretrieve(url, str(filepath))
        except OSError:
            raise RuntimeError(f"Failed to download {filename}")

    return filepath


def extract_file(filepath, remove_input=True):
    try:
        if tarfile.is_tarfile(str(filepath)):
            if (str(filepath).endswith(".gzip") or
                    str(filepath).endswith(".gz")):
                with tarfile.open(filepath, "r:gz") as tar:
                    tar.extractall(path=filepath.parent)
            elif (str(filepath).endswith(".tar.gz") or
                  str(filepath).endswith(".tgz")):
                with tarfile.open(filepath, "r:gz") as tar:
                    tar.extractall(path=filepath.parent)
            elif str(filepath).endswith(".tar"):
                with tarfile.open(filepath, "r:") as tar:
                    tar.extractall(path=filepath.parent)
            elif str(filepath).endswith(".bz2"):
                with tarfile.open(filepath, "r:bz2") as tar:
                    tar.extractall(path=filepath.parent)
            else:
                try:
                    with tarfile.open(filepath, "r:gz") as tar:
                        tar.extractall(path=filepath.parent)
                except tarfile.TarError:
                    raise RuntimeError(
                        "Unrecognized file format, may need to perform extraction manually with a custom dataset.")
        elif zipfile.is_zipfile(str(filepath)):
            with ZipFile(filepath, "r") as zip:
                zip.extractall(filepath.parent)
        else:
            try:
                with filepath.with_suffix("").open("wb") as output_f, \
                        gzip.GzipFile(filepath) as gzip_f:
                    shutil.copyfileobj(gzip_f, output_f)
            except gzip.BadGzipFile:
                raise RuntimeError("Undefined file format.")
            except:
                raise RuntimeError("Undefined exception.")
    except EOFError:
        raise RuntimeError("Dataset file isn't complete. Try downloading again.")

    if filepath.exists() and remove_input:
        filepath.unlink()

    return filepath.parent


def strip_header(filepath, num_lines):
    cmd = "tail -n +{} {} > tmp.txt".format(num_lines+1, filepath)
    os.system(cmd)

    cmd = "mv tmp.txt {}".format(filepath)
    os.system(cmd)