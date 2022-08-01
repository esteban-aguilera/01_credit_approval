import calendar
import re

from datetime import datetime


# --------------------------------------------------------------------------------
# Datetime
# --------------------------------------------------------------------------------
def str2date(s:str) -> datetime:
    """Transforms a string to a datetime object

    Parameters
    ----------
    s : str
        string that represents a date

    Returns
    -------
    datetime
        transformed datetime

    Raises
    ------
    ValueError
        The format of the argument does not match a valid date format
    """
    s = s.strip()
    if match:=re.match(r"(\d{4})-(\d{1,2})-(\d{1,2})", s):
        year, month, day = match.groups()
    elif match:=re.match(r"(\d{1,2})-(\d{1,2})-(\d{4})", s):
        day, month, year = match.groups()
    elif match:=re.match(r"(\d{4})-(\d{1,2})", s):
        year, month = match.groups()
        day = calendar.monthrange(int(year), int(month))[1]
    elif match:=re.match(r"(\d{1,2})-(\d{4})", s):
        month, year = match.groups()
        day = calendar.monthrange(int(year), int(month))[1]
    else:
        raise ValueError(f"The format of '{s}' does not match a valid date format")

    return datetime(int(year), int(month), int(day))


# --------------------------------------------------------------------------------
# Standarize Strings
# --------------------------------------------------------------------------------
def capitalize(s:str, sep:str=None) -> str:
    """capitalizes each word of a string

    Parameters
    ----------
    s : str
        string that needs to be capitalized
    sep : str, optional
        separator used to identify each word, by default r"\s"

    Returns
    -------
    str
        capitalized string
    """
    if sep is None:
        sep = r"\s"

    return ' '.join([w.strip().capitalize() for w in re.split(sep, s)])


def snake_case(s:str) -> str:
    s = re.sub(r"\s+", "_", s.strip().lower())

    return s
