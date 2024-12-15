import importlib.util
import importlib.metadata


def _is_package_available(package_name: str, version: str = None) -> bool:
    """
    Check if a package is installed and if a version is specified, check if the package
    matches the given version.

    Args:
        package_name (str): The name of the package to check.
        version (str, optional): The version to check against. Defaults to None.

    Returns:
        bool: True if the package is installed and the version matches if given, False otherwise.
    """
    exist = importlib.util.find_spec(package_name) is not None
    if exist and version is not None:
        try:
            # check version
            packge_version = importlib.metadata.version(package_name)
            return packge_version == version
        except importlib.metadata.PackageNotFoundError:
            return False
    else:
        return exist


def is_autogen_available(version: str = None) -> bool:
    return _is_package_available('autogen', version)


def is_open_interpreter_available() -> bool:
    return _is_package_available('interpreter')


def is_metagpt_available() -> bool:
    return _is_package_available('metagpt')

