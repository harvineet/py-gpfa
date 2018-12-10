from core_gpfa.exact_inference_with_LL import exact_inference_with_LL
from core_gpfa.postprocess import postprocess

def test_inference(seq, params):

    # Run inference
    (seq_out, LLtrain) = exact_inference_with_LL(seq, params, True)

    return seq_out, LLtrain

def test_orthonormalize(LL, params, seq):
    
    (est_params, seq_train, seq_test) = postprocess(params, seq, [], method='gpfa', kern_SD=30)

    return est_params, seq_train, seq_test