'''
vrdj operations
'''

def ingest(store, item_path, item_id):
    '''
    Ingest the item.

    This is idempotent.  If does not recalculate / resave embeddings or vectors
    that already exist.
    '''
    emb = store.get_embedding(item_id)
    if emb is None:
        emb = store.scheme.embed_audio(item_path)
        store.set_embedding(item_id, emb)
        store.conn.commit()
    indices = store.get_item_vectors(item_id)
    if not indices:
        indices = store.scheme.insert_embedding(emb)
        store.set_item_vectors(item_id, indices)
        store.flush()
        
        
        

    # if item does not have embeddings, make, store
    # if item does not have vectors, make, store
    
