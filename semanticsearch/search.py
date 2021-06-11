""" Search method with retrieval-reranking
"""
import time
import numpy as np
from semanticsearch import utils 

LOGGER = utils.init_logger()

def search(query,index,bi_encoder,cross_encoder,passages):
    LOGGER.info(f"Input question: {query}")

    ##### Sematic Search #####
    # Encode the query using the bi-encoder and find potentially relevant passages
    t=time.time()
    query_vector = bi_encoder.encode([query])
    top_k = index.search(query_vector, 3)
    top_k_ids = top_k[1].tolist()[0]
    top_k_ids = list(np.unique(top_k_ids))
    LOGGER.info('>>>> Results in Total Time: {}'.format(time.time()-t))

    ##### Re-Ranking #####
    # Now, score all retrieved passages with the cross_encoder
    t=time.time()
    cross_inp = [[query, passages[hit]] for hit in top_k_ids]
    bienc_op=[passages[hit] for hit in top_k_ids]
    cross_scores = cross_encoder.predict(cross_inp)
    LOGGER.info('>>>> Results in Total Time: {}'.format(time.time()-t))

    # Output of top-5 hits from bi-encoder
    LOGGER.info("\n-------------------------\n")
    LOGGER.info("Top-3 Bi-Encoder Retrieval hits")
    for result in bienc_op:
        LOGGER.info("\t{}".format(result.replace("\n", " ")))
        
    # Output of top-5 hits from re-ranker
    LOGGER.info("\n-------------------------\n")
    LOGGER.info("Top-3 Cross-Encoder Re-ranker hits")
    results=[]
    for hit in np.argsort(np.array(cross_scores))[::-1]:
        LOGGER.info("\t{}".format(bienc_op[hit].replace("\n", " ")))
        results.append(bienc_op[hit].replace("\n", " "))

    json_resp={}
    for rank,result in enumerate(results):
        json_resp[f'rank_{rank+1}']=result

    return json_resp