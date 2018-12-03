from core_gpfa.em import em

def test_em(seq, params, kernSDList):

    # Run inference
    (est_params, seq, LL, iterTime) = em(seq, params, kernSDList)
    
    return LL