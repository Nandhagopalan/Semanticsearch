""" Search method with retrieval-reranking
"""
import time
import numpy as np

def search(query,index,bi_encoder,cross_encoder):
    print("Input question:", query)

    ##### Sematic Search #####
    # Encode the query using the bi-encoder and find potentially relevant passages
    t=time.time()
    query_vector = bi_encoder.encode([query])
    top_k = index.search(query_vector, 3)
    top_k_ids = top_k[1].tolist()[0]
    top_k_ids = list(np.unique(top_k_ids))
    print('>>>> Results in Total Time: {}'.format(time.time()-t))

    ##### Re-Ranking #####
    # Now, score all retrieved passages with the cross_encoder
    t=time.time()
    cross_inp = [[query, passages[hit]] for hit in top_k_ids]
    bienc_op=[passages[hit] for hit in top_k_ids]
    cross_scores = cross_encoder.predict(cross_inp)
    print('>>>> Results in Total Time: {}'.format(time.time()-t))

    # Output of top-5 hits from bi-encoder
    print("\n-------------------------\n")
    print("Top-3 Bi-Encoder Retrieval hits")
    for result in bienc_op:
        print("\t{}".format(result.replace("\n", " ")))
        
#     for idx in range(len(cross_scores)):
#         hits[idx]['cross-score'] = cross_scores[idx]
    
    # Output of top-5 hits from re-ranker
    print("\n-------------------------\n")
    print("Top-3 Cross-Encoder Re-ranker hits")
    for hit in np.argsort(np.array(cross_scores))[::-1]:
        print("\t{}".format(bienc_op[hit].replace("\n", " ")))