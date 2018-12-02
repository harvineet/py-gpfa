from core_gpfa.exact_inference_with_LL import exact_inference_with_LL

def test_inference(seq, params):

    # Run inference
    (seq_out, LLtrain) = exact_inference_with_LL(seq, params, True)
    
    return LLtrain