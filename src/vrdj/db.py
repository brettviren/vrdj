'''
vrdj database

An "item" refers to a unit of audio content (eg a song).  An item has and "id"
provided externally from vrdj, but it is intended to be a beets item.id.
'''

#fixme: rename this "store" or "main" or something.  It's more than a dbi.

import os
import time
import sqlite3
import numpy as np
from pathlib import Path
from vrdj.scheme import Scheme
import vrdj.embeddings

from vrdj.util import sqlite_cursor

def tensor_to_blob(tensor: np.ndarray) -> bytes:
    """Converts a NumPy array into a raw byte BLOB for SQLite storage."""
    # Ensure it's stored as little-endian float32 for portability and consistency
    return tensor.astype('<f4').tobytes()

def blob_to_tensor(blob: bytes, vector_size: int) -> np.ndarray:
    """Reconstitutes a NumPy array from a BLOB."""
    if not blob:
        return None
    # Read as little-endian float32 and reshape to (N, D)
    return np.frombuffer(blob, dtype='<f4').reshape(-1, vector_size)
    # fixme: store shape in DB

class Store:
    '''
    A store object holds the vrdj state as central sqlite DB file and FAISS indices.

    The store directly manages the embeddings table and delegates vector tables
    and FAISS indices to the "Scheme".
    '''

    def __init__(self, dirpath: str|Path,
                 metric: str = 'cosine',
                 embedding: str = 'vggish',
                 device: str ='cpu'):
        '''
        Create a vrdj store.

        This consists of a general database file which caches embeddings and
        scheme vector indices.  Unique table is made for embeddings of a given
        name and the vector indexing is done on a per scheme basis.  Multiple
        stores can share the embeddings and the unique vector index tables will
        be kept distinct by their name.
        '''

        dirpath = Path(dirpath)
        dirpath.mkdir(parents=True, exist_ok=True)
        #print(f'vrdj store in: {dirpath}')
        self.dirpath = dirpath

        emod = getattr(vrdj.embeddings, embedding)
        self.vector_length = emod.vector_length
        self.model = emod.Model(device)

        self.sqlite_filepath = dirpath / "store.sqlite"
        self.tablename = f'embedding_{embedding}'

        self._init_sqlite()

        self.scheme = Scheme(dirpath, db=self.db,
                             metric=metric, embedding=embedding)


    def get_embedding(self, item_id):
        '''
        Return item's embedding or None if no item.
        '''
        with sqlite_cursor(self.db) as cursor:
            cursor.execute(
                f"SELECT embedding FROM {self.tablename} WHERE item_id = ?",
                (item_id,))
            result = cursor.fetchone()
            if not result:
                return
            return blob_to_tensor(result[0], self.vector_length)

    def get_many_embeddings(self, item_ids):
        '''
        Return embeddings for item_ids
        '''
        return map(self.get_embedding, item_ids)


    def add_embedding(self, item_id, source, force=False):
        '''
        Store an item's embedding and index its vectors.

        If item_id is already stored, this will not restore unless force=True

        The source may be an embedding tensor or a audio filename.
        '''
        embedding = self.get_embedding(item_id)
        if embedding is not None and not force:
            # print(f"already have embedding for {item_id=}")
            return

        if isinstance(source, np.ndarray):
            embedding = source
        else:
            embedding = self.model.embedding(source)
            
        blob = tensor_to_blob(embedding)
        with sqlite_cursor(self.db) as cursor:
            cursor.execute(
                f"""
                INSERT OR REPLACE INTO {self.tablename}
                (item_id, embedding, created)
                VALUES (?, ?, ?)
                """,
                (item_id, blob, time.time()))
        # forward to scheme no matter what
        self.scheme.add_embedding(item_id, embedding)

    def _init_sqlite(self):
        """Initializes the SQLite connection and creates the mapping tables."""
        if hasattr(self, 'db'):
            return
            
        self.db = sqlite3.connect(self.sqlite_filepath.absolute())
        with sqlite_cursor(self.db) as cursor:

            # The vggish source.  Each item data is fed to VGGish and the embedding
            # that spans multiple segments is stored.  The item_id is an external
            # ID, ie beets item ID ($id).
            cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.tablename} (
            id INTEGER PRIMARY KEY,
            item_id INTEGER,
            embedding BLOB NOT NULL,
            created REAL
            );
            """)

    
