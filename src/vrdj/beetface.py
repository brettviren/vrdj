'''
Non-plugin interface to beets
'''
import os
from beets.library import Library
from beets import ui, config, util
from pathlib import Path

def dbpath():
    '''
    Return the path to the beets library
    '''
    return Path(config["library"].as_filename())

def music_directory():
    return Path(config["directory"].as_filename())

def library():
    '''
    Return your beets library
    '''
    path = util.bytestring_path(dbpath().absolute())

    lib = Library(
        path,
        config["directory"].as_filename(),
        ui.get_path_formats(),
        ui.get_replacements(),
    )
    return lib
    

def item_at_path(lib, item_path):
    '''
    Query
    '''
    normalized_path = os.path.abspath(item_path)
    query = f'path:"{normalized_path}"'
    got = lib.items(query)
    if got:
        return got[0]
    return None

