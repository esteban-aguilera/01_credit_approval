import os
import numpy as np
import pandas as pd

from typing import List


# package imports
from . import paths, strings, excel


# --------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------
list_str = List[str]


# --------------------------------------------------------------------------------
# Loggger Class
# --------------------------------------------------------------------------------
class Logger:
    
    content: dict
    
    def __init__(self) -> None:
        self.content = {}
    
    def append(self, section:str, msg_type:str, msg:str) -> None:
        section = strings.capitalize(section)
        msg_type = strings.capitalize(msg_type)

        if section not in list(self.content.keys()):
            self.content[section] = []

        self.content[section].append([msg_type, msg])
        
    def save(self, filename:str=None, mode:str=None, sections:list_str=None):
        if filename is None or filename.lower().endswith(".txt"):
            self.save_text(filename=filename, mode=mode, sections=sections)
        elif filename.lower().endswith(".xlsx"):
            self.save_excel(filename=filename, mode=mode, sections=sections)
        
    def save_excel(self, filename:str=None, mode:str=None, sections:list_str=None) -> None:
        if filename is None:
            filename = 'logger.xlsx'

        if mode is None or mode == "error":
            if os.path.exists(filename):
                raise ValueError(f"The file '{filename}' already exists.")
            
            delete_workbook = False
            delete_worksheet = True
        elif mode == "append":
            delete_workbook = False
            delete_worksheet = False
        elif mode == "overwrite":
            delete_workbook = True
            delete_worksheet = True
        else:
            raise ValueError(f"The value '{mode}' is invalid for argument mode.")
        
        if sections is None:
            sections = list(self.content.keys())
            
        if "/" in filename:
            paths.mkdir('/'.join(filename.split("/")[:-1]), verbose=True)
            
        content = {}
        for section in sections:
            content[section] = self.content[section]
        
        for section in sections:
            df = pd.DataFrame(content[section], columns=["Type", "Message"])
            
            if delete_workbook:
                excel.save_dataframe(
                    df, filename, section,
                    delete_workbook=delete_workbook
                )
            elif delete_worksheet:
                excel.save_dataframe(
                    df, filename, section,
                    delete_worksheet=delete_worksheet
                )
            else:
                if section in excel.get_sheetnames(filename):
                    header = False
                    startrow = excel.get_nrows(filename, section)
                    df.index += startrow - 1
                else:
                    header = True
                    startrow = 0
                
                excel.save_dataframe(
                    df, filename, section,
                    header=header, startrow=startrow
                )
        
    def save_text(self, filename:str=None, mode:str=None, sections:list_str=None) -> None:
        if filename is None:
            filename = 'logger.txt'

        if mode is None or mode == "error":
            if os.path.exists(filename):
                raise ValueError(f"The file '{filename}' already exists.")
            
            mode = "w"
        elif mode == "append":
            mode = "a+"
        elif mode == "overwrite":
            self.remove(filename)
            mode = "w"
        else:
            raise ValueError(f"The value '{mode}' is invalid for argument mode.")
        
        if sections is None:
            sections = list(self.content.keys())
            
        if "/" in filename:
            paths.mkdir('/'.join(filename.split("/")[:-1]), verbose=True)
            
        content = {}
        for section in sections:
            content[section] = self.content[section]
    
        new_file = not os.path.exists(filename)

        with open(filename, mode) as f:
            for section in sections:
                if not new_file:
                    f.write("\n\n")
                    new_file = False
                
                # crear titulo de la sección
                f.write(section + "\n" + len(section)*"-" + "\n\n")
                
                for msg_type, msg in content[section]:
                    f.write(f"{msg_type}: {msg}\n\n")
                    
        
    def remove(self, filename:str) -> None:
        if os.path.exists(filename):
            print(f"Removing '{filename}'")
            os.remove(filename)

    def clear(self) -> None:
        self.content = {}
