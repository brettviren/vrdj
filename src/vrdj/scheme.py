'''
The vrdj "scheme" covers embeddings of audio data and their comparisons.

An "embedding" is a 2D array shaped as (nsegments, vlen).  A "segment" is a
portion of audio data spanning on time sample period.  The "vlen" is the length
of the vector that represents one segment.

For VGGish, a segment spans 0.96 seconds with 50% overlap.  Other embeddings
that produce 2D array of segments may be added in the future.

The vrdj scheme will maintain two FAISS indices: "average" and "segment".  The
"average" index holds vectors which are the average over the segments while the
"segment" index holds the individual segment vectors.  

The scheme also have a "metric" used to compare vectors.  The metric is baked
into the FAISS index and so different metrics require different "average" and
"segment" indices.  The 'cosine' metric is default while 'l2' is also possible.
'''

import vrdj.embeddings
import faiss
from pathlib import Path
import numpy
from vrdj.util import sqlite_cursor

class Index:
    def __init__(self, kind, dirpath, db, 
                 metric='cosine', embedding='vggish'):
        self.kind = kind
        self.db = db
        emod = getattr(vrdj.embeddings, embedding)
        self.vector_length = emod.vector_length
        self._metric = metric
        self._embedding = embedding
        
        dirpath = Path(dirpath)
        self.filepath = dirpath / f'{kind}-{embedding}-{metric}.faiss'
        self.tablename = f'vectors_{kind}_{embedding}_{metric}'

        self._init_db()

    @property
    def index(self):
        if not hasattr(self, '_index'):
            if self.filepath.exists():
                filename = str(self.filepath.absolute())
                index = faiss.read_index(filename)
                if index.d != self.vector_length:
                    raise ValueError(f'Vector length mismatch: {self._embedding} produces {self.vector_length} while index expects {index.d}')
            else:
                if self._metric == 'cosine':
                    maker = faiss.IndexFlatIP
                elif self._metric == 'l2':
                    maker = faiss.IndexFlagL2
                else:
                    raise ValueError(f'unsupported metric: {self._metric}')
                index = maker(self.vector_length)
            setattr(self, '_index', index)
        return self._index

    def vectorize(self, emb):
        '''
        Return vectorized embedding as shape (nvectors, vector_length)
        '''
        if self._metric == 'cosine':
            emb = numpy.ascontiguousarray(emb)
            faiss.normalize_L2(emb)

        if self.kind == 'segment':
            vec = emb.astype('float32')
        else:
            vec = numpy.mean(emb, axis=0).reshape(1,-1).astype('float32')
        return vec

    def save(self):
        '''
        Save the index.
        '''
        index = getattr(self, '_index', None)
        if index is None:
            print(f'no {self.kind} index to save')
            return
        faiss.write_index(index, str(self.filepath.absolute()))
        
    def query_one(self, vector, count=1, return_scores=False):
        '''
        Return at most count vector IDs similar to vector.

        The vector is 1D array.

        If return_scores is True, return tuple of (vector_ids, scores).
        '''
        if self.index.ntotal == 0:
            print(f'index for {self.filepath} has no entries')
            return None

        # search interface expects (nvectors, vector_length) shape
        if vector.ndim == 1:
            vector = vector.reshape(1, -1).astype('float32')
        count = min(count, self.index.ntotal)
        got = self.index.search(vector, count)
        if got is None:
            return None
        scores, indices = got        
        if return_scores:
            return (indices[0], scores[0])
        return indices[0]

    def query_many(self, vectors, count=1, return_scores=False):
        '''
        Like query_one but vectors is a 2D (nvectors, vector_length)

        Return is a sequence along nvectors.
        '''
        count = min(count, self.index.ntotal)
        scores, indices = self.index.search(vectors, count)
        if return_scores:
            return (indices, scores)
        return indices

    def add_embedding(self, item_id, embedding):
        '''
        Insert the embedding for the item.
        '''
        vecs = self.vectorize(embedding)
        start_size = self.index.ntotal
        self.index.add(vecs)
        end_size = self.index.ntotal
        self.save()
        vector_ids = tuple(range(start_size, end_size))
        with sqlite_cursor(self.db) as cursor:
            for segment, vector_id in enumerate(vector_ids):
                cursor.execute(
                    f"""
                    INSERT or REPLACE INTO {self.tablename}
                    (vector_id, item_id, segment)
                    VALUES (?, ?, ?)
                    """,
                    (vector_id, item_id, segment))

    def get_item_vectors(self, item_id):
        '''
        Return FAISS vector IDs for item, ordered by segment.
        '''
        with sqlite_cursor(self.db) as cursor:
            cursor.execute(
                f"""
                SELECT vector_id FROM {self.tablename}
                WHERE item_id = ?
                ORDER BY segment
                """, (item_id,))
            return cursor.fetchall()

    def get_item_with_vector(self, vector_id):
        '''
        Return item ID that has a FAISS vector id.
        '''
        with sqlite_cursor(self.db) as cursor:
            cursor.execute(
                f"""
                SELECT item_id FROM {self.tablename}
                WHERE vector_id = ?
                """, (vector_id,))
            got = cursor.fetchone()
            if got is None:
                return
            return got[0]

    def get_items_by_vectors(self, vector_ids):
        '''
        Return item IDs that have the FAISS vector ids.
        '''
        if isinstance(vector_ids, numpy.ndarray):
            vector_ids = vector_ids.tolist()

        item_ids = list()
        for vector_id in vector_ids:
            item_id = self.get_item_with_vector(vector_id)
            if item_id is None:
                print(f'No item for {vector_id=}')
                continue
            item_ids.append(item_id)
        return item_ids

    def _init_db(self):
        '''
        Create sqlite table.
        '''
        with sqlite_cursor(self.db) as cursor:
            cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.tablename} (
            id INTEGER PRIMARY KEY,
            vector_id INTEGER NOT NULL,
            item_id INTEGER NOT NULL,
            segment INTEGER NOT NULL
            );
            """)
            cursor.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_item_{self.tablename}
            ON {self.tablename} (item_id);""")

class Scheme:
    '''
    Collect operations and persistent info for one scheme.

    This manages a number of FAISS indices and sqlite tables.
    '''

    def __init__(self, dirpath, db,
                 metric='cosine', embedding='vggish'):
        '''
        Construct a scheme.

        The scheme's vector indices may be saved under dirpath.
        '''
        self._metric = metric
        self._embedding = embedding

        self.index_average = Index("average", dirpath, db, metric, embedding)
        self.index_segment = Index("segment", dirpath, db, metric, embedding)
        self.indices = dict(
            average = self.index_average,
            segment = self.index_segment)

    def save(self):
        '''
        Save the index.
        '''
        for ind in self.indices.values():
            ind.save()

    def add_embedding(self, item_id, embedding):
        '''
        Insert an embedding and return vector IDs: (average, segments)
        '''
        for ind in self.indices.values():
            ind.add_embedding(item_id, embedding)
