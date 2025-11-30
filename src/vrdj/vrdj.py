import vrdj.embeddings
import vrdj.vectors
import vrdj.db
import vrdj.store

class VRDJ:
    def __init__(dbfile, embedding="vggish", vectors="faiss", comparison="mean", device='cpu'):

        

        emod = getattr(vrdj.embeddings, embedding)

        self.embedding = emod.Model(device)
        self.vector_length = emod.vector_length

        vmod = getattr(vrdj.vectors, vectors)
        
        
