module HMMBase

import Distributions: Distribution, Univariate, Multivariate, VariateForm
import StatsFuns: logsumexp

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

include("hmm.jl")
include("decoding.jl")
include("filtering.jl")

end
