module HMMBase

import Distributions: Distributions, Univariate, Multivariate, VariateForm

export
    # hmm.jl
    HMM,
    assert_hmm,
    sample_hmm,
    compute_transition_matrix,
    # decoding.jl
    viterbi,
    # filtering.jl
    messages_backward,
    messages_backward_log,
    messages_forward,
    messages_forward_log

end
