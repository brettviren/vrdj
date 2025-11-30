'''
The vrdj "scheme" is a combination of
- embedding :: how we form a tensor from an audio file
- vectorize :: how we form a vector from an embedding
- metric :: how we compare vectors

A vector database instance is specific to this trio.

We hard-wire the vector database technology to be FAISS and default to using
VGGish for embeddings but allow to extend this later to others.
'''

import vrdj.embeddings
import faiss
from pathlib import Path
import numpy as np

def name(embedding='vggish', vectorize='mean', metric='cosine'):
    '''
    Form canonical scheme name.
    '''
    return f"{embedding}-{vectorize}-{metric}"

# Here we hard-wire and number supported schemes.  The index number is used in
# the DB so only extend this map, do not change entries, or risk conflicting
# with existing databases.
INDEX_SCHEME_MAP = {
    1: {'embedding': 'vggish', 'vectorize': 'mean', 'metric': 'cosine'},
    2: {'embedding': 'vggish', 'vectorize': 'mean', 'metric': 'l2'},
    3: {'embedding': 'vggish', 'vectorize': 'segment', 'metric': 'cosine'},
    4: {'embedding': 'vggish', 'vectorize': 'segment', 'metric': 'l2'},
}

# Look up index scheme number given string construction
INDEX_SCHEME_REVERSE_MAP = {
    name(**val): key
    for key, val in INDEX_SCHEME_MAP.items()
}

def vectorize_segment(emb):
    '''
    Return (nsegments, vector) tensor (just emb)
    '''
    vectors = emb.astype('float32')
    print(f'vectorize_segment: {vectors.shape} {vectors.dtype}')
    return vectors

def vectorize_mean(emb):
    '''
    Return (1, vector) tensor of mean emb
    '''
    vectors = np.mean(emb, axis=0).reshape(1, -1).astype('float32')
    print(f'vectorize_mean: {vectors.shape} {vectors.dtype}')
    return vectors
    

def normalize_l2(vecs):
    faiss.normalize_L2(vecs)
    return vecs

def normalize_none(vecs):
    return vecs

# Function to return sequence of vectors given embedding tensor
VECTORIZE_SCHEMES = {
    'segment': vectorize_segment,
    'mean': vectorize_mean,
}

METRIC_SCHEMES = {
    'cosine': {
        'index_factory': lambda D: faiss.IndexFlatIP(D),
        'normalize': True,
        'pre_normalize': True,
    },
    'l2': {
        'index_factory': lambda D: faiss.IndexFlatL2(D),
        'normalize': False, 
        'pre_normalize': False,
    }
}

class Scheme:
    '''
    Collect operations and persistent info for one scheme.
    '''

    def __init__(self, dirpath, scheme='vggish-mean-cosine', device='cpu'):
        '''
        Construct a scheme.

        The scheme's vector index may be saved under dirpath.

        The scheme is specified with its canonical name or number.
        '''
        if isinstance(scheme, str):
            scheme = INDEX_SCHEME_REVERSE_MAP[scheme]
        self.number = scheme
        self.ism = INDEX_SCHEME_MAP[scheme]
        self.name = name(**self.ism)

        emod = getattr(vrdj.embeddings, self.ism['embedding'])
        self.vector_length = emod.vector_length
        self.model = emod.Model(device)
        self.device = device
        self.index_filepath = Path(dirpath) / (self.name + ".faiss")
        print(f'vrdj scheme: {self.index_filepath}')

        if METRIC_SCHEMES[self.metric]['normalize']:
            self.normalizer = normalize_l2
        else:
            self.normalizer = normalize_none
        self.vectorize_func = VECTORIZE_SCHEMES[self.vectorize]

    @property
    def embedding(self):
        return self.ism['embedding']
    @property
    def vectorize(self):
        return self.ism['vectorize']
    @property
    def metric(self):
        return self.ism['metric']

    @property
    def index(self):
        if not hasattr(self, '_index'):
            if self.index_filepath.exists():
                self._index = faiss.read_index(self.index_filepath.abspath())
                if self._index.d != self.vector_length:
                    raise ValueError(f'Vector length mismatch: {self.embedding} produces {self.vector_length} while index expects {self._index.d}')
            else:
                ms = METRIC_SCHEMES[self.metric]
                maker = ms['index_factory']
                self._index = maker(self.vector_length)
        return self._index

    def save_index(self):
        '''
        Save current index object to the file path for this scheme.
        '''
        if not hasattr(self, '_index'):
            print(f'scheme {self.name} has no index to save')
            return
        print(f'scheme {self.name} saving index to {self.index_filepath}')
        faiss.write_index(self.index, str(self.index_filepath.absolute()))

    def embed_audio(self, audio):
        '''
        Return an embedding from audio.

        The audio may be a filename or a waveform.
        '''
        return self.model.embedding(audio);

    def vectorize_embedding(self, emb):
        '''
        Return vectors for the embedding as 2D tensor.

        The embedding is pre-processed based on the scheme.
        '''
        vectors = self.vectorize_func(emb)
        if vectors is None:
            raise ValueError(f'no vectors from {emb.shape=} {emb.dtype=}')
        normed = self.normalizer(vectors)
        if normed is None:
            raise ValueError(f'no normalizer from {vectors.shape=} {vectors.dtype=}')

        return normed

    def insert_embedding(self, emb):
        '''
        Insert an embedding and return vector indices.

        The indices are provided as a number array.

        Note, the index is not (yet) saved to disk.
        '''
        vectors = self.vectorize_embedding(emb)
        if vectors is None:
            raise ValueError(f'no vectors from {emb.shape=} {emb.dtype=}')
        self.index.add(vectors)
        return vectors

    def query(self, vector_id, count=1):
        '''
        Return no more than count vector ids that are near vector_id.
        '''
        # fixme: add support for batched vector_id.
        count = min(count, self.index.ntotal)
        scores, indices = self.index.search(vector_id, count)
        return indices[0]

