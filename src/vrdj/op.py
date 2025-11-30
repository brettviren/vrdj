'''
vrdj operations
'''

import numpy as np

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
        
        
def similar_average_item(store, item_id, count):
    '''
    Return item IDs for items similar to average vector of item_id.
    '''
    emb = store.get_embedding(item_id)

    index = store.scheme.index_average

    vec = index.vectorize(emb)
    vids, scores = index.query_one(vec, count, return_scores=True)
    return index.get_items_by_vectors(vids)

def similar_average_many(store, item_ids, count):
    '''
    Return item IDs for items similar to average vector of item_id.
    '''
    assert count > 0

    index = store.scheme.index_average

    vectors = list()
    for emb in store.get_many_embeddings(item_ids):
        vec = index.vectorize(emb)
        print(f'similar: {emb.shape=} {vec.shape=} {vec.dtype=}')
        vectors.append(vec)
    vecs = np.vstack(vectors)
    print(f'similar: {vecs.shape=} {vecs.dtype=}')
    vec = np.mean(vecs, axis=0)
    print(f'similar: {vec.shape=} {vec.dtype=} {count=}')
    vids, scores = index.query_one(vec, count, return_scores=True)
    for v,s in zip(vids, scores):
        print(f'vector_id={v} {type(v)} score={s}')
    return index.get_items_by_vectors(vids)



