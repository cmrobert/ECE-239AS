import tensorflow as tf


def logsigmoid(input_tensor):
    # return -tf.log(tf.sigmoid(-input_tensor))
    return -tf.nn.softplus(-input_tensor)


def logit_bernoulli_entropy(logits):
    # Reference: https://github.com/openai/imitation/blob/99fbccf3e060b6e6c739bdf209758620fcdefd3c/policyopt/thutil.py#L48-L51
    ent = (1.0 - tf.nn.sigmoid(logits)) * logits - logsigmoid(logits)
    return ent
