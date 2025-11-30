'''
vrdj database

An "item" refers to a unit of audio content (eg a song).  An item has and "id"
provided externally from vrdj, but it is intended to be a beets item.id.
'''
import os
import time
import sqlite3
import numpy as np
from pathlib import Path
from vrdj.scheme import Scheme


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
    A store object holds the vrdj state as central sqlite DB file.
    '''

    def __init__(self, dirpath: str|Path, scheme: str|int = 1, device: str ='cpu'):
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
        print(f'vrdj store in: {dirpath}')
        self.dirpath = dirpath
        self.scheme = Scheme(dirpath, scheme=scheme, device=device)
        self.sqlite_filepath = dirpath / "store.sqlite"
        self.embedding_table = f'embedding_{self.scheme.embedding}'
        self.vector_table = f'vector_{self.scheme.name.replace("-","_")}'

        pass

    def flush(self):
        self.scheme.save_index()
        self.conn.commit()

    @property
    def cursor(self):
        self._init_sqlite()
        return self._cursor

    @property
    def conn(self):
        self._init_sqlite()
        return self._conn

    def get_embedding(self, item_id):
        '''
        Return item's embedding or None if no item.
        '''
        print(f'{item_id=} {type(item_id)}')
        self.cursor.execute(
            f"SELECT embedding FROM {self.embedding_table} WHERE item_id = ?",
            (item_id,))
        result = self.cursor.fetchone()
        if not result:
            return
        return blob_to_tensor(result[0], self.scheme.vector_length)

    def set_embedding(self, item_id, embedding):
        '''
        Store an item's embedding and index its vectors.
        '''
        blob = tensor_to_blob(embedding)
        self.cursor.execute(
            f"""
            INSERT OR REPLACE INTO {self.embedding_table}
            (item_id, embedding, created)
            VALUES (?, ?, ?)
            """,
            (item_id, blob, time.time()))

    def get_item_vectors(self, item_id):
        '''
        Return FAISS vector IDs for item, ordered by segment.
        '''
        self.cursor.execute(
            f"""
            SELECT vector_id FROM {self.vector_table}
            WHERE item_id = ?
            ORDER BY segment
            """, (item_id,))
        return self.cursor.fetchall()

    def set_item_vectors(self, item_id, vector_ids):
        '''
        Set vector ids for item.  Vector ids are expected in segment order.
        '''
        for segment, vector_id in enumerate(vector_ids):
            self.cursor.execute(
                f"""
                INSERT or REPLACE INTO {self.vector_table}
                (vector_id, item_id, segment)
                VALUES (?, ?, ?)
                """,
                (vector_id, item_id, segment))

    def get_item_with_vector(self, vector_id):
        '''
        Return item ID that has a FAISS vector id.
        '''
        self.cursor.execute(
            f"""
            SELECT item_id FROM {self.vector_table}
            WHERE vector_id = ?
            """, (vector_id,))
        return self.cursor.fetchone()

    def get_items_by_vectors(self, vector_ids):
        '''
        Return item IDs that have the FAISS vector ids.
        '''
        prm_list = ", ".join(['?']*len(vector_ids))
        self.cursor.execute(
            f"""
            SELECT DISTINCT item_id FROM {self.vector_table}
            WHERE vector_is IN ({prm_list})
            """, tuple(vector_ids))
        return self.cursor.fetchall()

    def _init_sqlite(self):
        """Initializes the SQLite connection and creates the mapping tables."""
        if hasattr(self, '_conn'):
            return
            
        self._conn = sqlite3.connect(self.sqlite_filepath.absolute())
        self._cursor = self._conn.cursor()
        
        # The vggish source.  Each item data is fed to VGGish and the embedding
        # that spans multiple segments is stored.  The item_id is an external
        # ID, ie beets item ID ($id).
        self._cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {self.embedding_table} (
        id INTEGER PRIMARY KEY,
        item_id INTEGER,
        embedding BLOB NOT NULL,
        created REAL
        );
        """)

        # Associate a vector in a faiss index with a item and a time segment.
        # The combination of source and metric implies the faiss database.
        self._cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {self.vector_table} (
        id INTEGER PRIMARY KEY,
        vector_id INTEGER NOT NULL,
        item_id INTEGER NOT NULL,
        segment INTEGER NOT NULL
        );
        """)

        # Want to find faiss indices for given item and source/metric.
        self._cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_item_{self.vector_table} ON {self.vector_table} (item_id);")
        self._conn.commit()
    
