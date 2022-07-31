import os


# --------------------------------------------------------------------------------
# Functions
# --------------------------------------------------------------------------------
def mkdir(path:str, verbose=False) -> None:
    """Create new directory

    Parameters
    ----------
    path : str
        Path to be created
    verbose : bool, optional
        If True informative prints are created, by default False
    """
    if os.path.exists(path):
        return
    
    subpaths = path.split("/")
    for i, subpath in enumerate(subpaths):
        if i == 0:
            cumpath = subpath
        else:
            cumpath = f"{cumpath}/{subpath}"
            
        if not os.path.exists(cumpath):
            os.mkdir(cumpath)

    if verbose:
        print(f"The directory '{path}' has been created succesfully.")
