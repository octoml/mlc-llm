import torch
import pytest
from mlc_serve.model.sampler import SamplingState, adjust_logits
from mlc_serve.engine import SamplingParams, SAMPLING_EPS

dtype = torch.float32
dev = "cuda"
vocab_size = 32000


def get_sampling_state(sampling_params, past_output_tokens=None):
    batch_size = len(sampling_params)
    if past_output_tokens is None:
        past_output_tokens = [[] for _ in range(batch_size)]
    _copy_stream: torch.cuda.Stream = torch.cuda.Stream()
    with torch.cuda.stream(_copy_stream):
        sampling_state = SamplingState.from_sampling_params(
            sampling_params,
            list_past_output_tokens=past_output_tokens,
            dtype=dtype,
            dev=dev,
            vocab_size=vocab_size,
        )
    torch.cuda.current_stream().wait_stream(_copy_stream)
    return sampling_state


def _test_temperature(temp=0, batch_size=1):
    shape = (batch_size, vocab_size)
    logits = torch.rand(shape, dtype=dtype, device=dev)
    sampling_param = SamplingParams(
        temperature=temp,
    )

    sampling_state = get_sampling_state([sampling_param])

    expected = logits / temp if abs(temp) > SAMPLING_EPS else logits
    new_logits = adjust_logits(logits, sampling_state, vocab_size)
    assert torch.allclose(expected, new_logits)


def _test_logit_bias_checker():
    # logit bias must be [-100, 100]
    with pytest.raises(ValueError):
        logit_bias = {1: 2, 3: 105, 2: 2}
        sampling_param = SamplingParams(logit_bias=logit_bias)
        get_sampling_state([sampling_param])

    with pytest.raises(ValueError):
        logit_bias = {1: 99, 3: -101, 2: 2}
        sampling_param = SamplingParams(logit_bias=logit_bias)
        get_sampling_state([sampling_param])

    logit_bias = {1: 100, 3: -100, 2: 2}
    sampling_param = SamplingParams(logit_bias=logit_bias)
    get_sampling_state([sampling_param])

    # TODO(@team): it seems like the valid range is [1,vocab_size]. Double check.
    logit_bias = {1: 10, 3: -10, vocab_size: 2}
    sampling_param = SamplingParams(logit_bias=logit_bias)
    get_sampling_state([sampling_param])

    with pytest.raises(ValueError):
        logit_bias = {0: 10, 3: -10}
        sampling_param = SamplingParams(logit_bias=logit_bias)
        get_sampling_state([sampling_param])

    with pytest.raises(ValueError):
        logit_bias = {1: 10, 3: -10, vocab_size + 100: 2}
        sampling_param = SamplingParams(logit_bias=logit_bias)
        get_sampling_state([sampling_param])

    with pytest.raises(ValueError):
        logit_bias = {1: 10, -1: -10}
        sampling_param = SamplingParams(logit_bias=logit_bias)
        get_sampling_state([sampling_param])


def _test_logit_bias():
    # test single batch
    batch_size = 1
    shape = (batch_size, vocab_size)
    logits = torch.rand(shape, dtype=dtype, device=dev)
    logit_bias = {1: -1, 3: 1, 2: 2}
    sampling_param = SamplingParams(logit_bias=logit_bias)
    sampling_state = get_sampling_state([sampling_param])

    expected = torch.clone(logits)
    for idx, val in logit_bias.items():
        expected[0][idx - 1] += val
    new_logits = adjust_logits(logits, sampling_state, vocab_size)
    assert torch.allclose(expected, new_logits)

    # test multi-batch
    batch_size = 3
    shape = (batch_size, vocab_size)
    logits = torch.rand(shape, dtype=dtype, device=dev)
    list_logit_bias = [{1: -1, 3: 1, 2: 2}, {4: 2, 5: 1}, {1: -10}]
    sampling_params = [
        SamplingParams(logit_bias=logit_bias) for logit_bias in list_logit_bias
    ]
    sampling_state = get_sampling_state(sampling_params)

    expected = torch.clone(logits)
    for batch_size in range(batch_size):
        logit_bias = list_logit_bias[batch_size]
        for idx, val in logit_bias.items():
            expected[batch_size][idx - 1] += val
    new_logits = adjust_logits(logits, sampling_state, vocab_size)
    assert torch.allclose(expected, new_logits)


def _test_penalties_checker():
    get_sampling_state([SamplingParams(presence_penalty=-1.0)])
    get_sampling_state([SamplingParams(frequency_penalty=-1.0)])
    get_sampling_state([SamplingParams(repetition_penalty=0.7)])

    with pytest.raises(ValueError):
        get_sampling_state([SamplingParams(presence_penalty=-2.1)])

    with pytest.raises(ValueError):
        get_sampling_state([SamplingParams(frequency_penalty=-2.1)])

    with pytest.raises(ValueError):
        get_sampling_state([SamplingParams(repetition_penalty=-2.1)])

    with pytest.raises(ValueError):
        get_sampling_state([SamplingParams(presence_penalty=2.1)])

    with pytest.raises(ValueError):
        get_sampling_state([SamplingParams(frequency_penalty=2.1)])

    with pytest.raises(ValueError):
        get_sampling_state(
            [
                SamplingParams(frequency_penalty=1.1),
                SamplingParams(repetition_penalty=2.1),
                SamplingParams(presence_penalty=1.1),
                SamplingParams(presence_penalty=3.1),
            ]
        )


def _test_penalties():
    # TODO(vvchernov): Add test for repetition penalty
    batch_size = 1
    shape = (batch_size, vocab_size)
    logits = torch.rand(shape, dtype=dtype, device=dev)
    presence_penalties = [0.8]
    frequency_penalties = [0.3]
    past_output_tokens = [[2, 2, 2, 3]]

    def prepare_metadata(past_output_tokens):
        count_map = []
        for past_output_tokens_per_req in past_output_tokens:
            # TODO: Check if this is the right range
            cnt = [0] * (vocab_size)
            for tok in past_output_tokens_per_req:
                cnt[tok] += 1
            count_map.append(cnt)

        count_tensor = torch.tensor(count_map, device=dev)
        mask_tensor = count_tensor > 0
        return count_tensor, mask_tensor

    count_map, mask = prepare_metadata(past_output_tokens)

    def get_expected_result(
        logits, count_map, mask, frequency_penalties, presence_penalties
    ):
        expected = torch.clone(logits)
        for i in range(batch_size):
            expected[i] = (
                expected[i]
                - count_map[i] * frequency_penalties[i]
                - mask[i] * presence_penalties[i]
            )
        return expected

    expected = get_expected_result(
        logits, count_map, mask, frequency_penalties, presence_penalties
    )

    sampling_param = [
        SamplingParams(
            presence_penalty=presence_penalties[0],
            frequency_penalty=frequency_penalties[0],
        )
    ]
    sampling_state = get_sampling_state(
        sampling_param, past_output_tokens=past_output_tokens
    )
    new_logits = adjust_logits(logits, sampling_state, vocab_size)
    assert torch.allclose(expected, new_logits)

    batch_size = 3
    shape = (batch_size, vocab_size)
    logits = torch.rand(shape, dtype=dtype, device=dev)
    presence_penalties = [0.8, 0.7, -0.8]
    frequency_penalties = [-0.3, 2.0, 1.2]
    past_output_tokens = [[2, 2, 2, 3, 5], [3, 1, 2, 4], [3, 3, 1]]

    count_map, mask = prepare_metadata(past_output_tokens)
    expected = get_expected_result(
        logits, count_map, mask, frequency_penalties, presence_penalties
    )

    sampling_params = [
        SamplingParams(
            presence_penalty=presence_penalties[i],
            frequency_penalty=frequency_penalties[i],
        )
        for i in range(batch_size)
    ]
    sampling_state = get_sampling_state(
        sampling_params, past_output_tokens=past_output_tokens
    )
    new_logits = adjust_logits(logits, sampling_state, vocab_size)
    assert torch.allclose(expected, new_logits)


def _test_top_p_top_k_checker():
    get_sampling_state([SamplingParams(top_p=0.8)])
    get_sampling_state([SamplingParams(top_k=3)])

    get_sampling_state([SamplingParams(top_k=-1)])
    get_sampling_state([SamplingParams(top_k=1)])

    with pytest.raises(ValueError):
        get_sampling_state([SamplingParams(top_p=0.0)])

    with pytest.raises(ValueError):
        get_sampling_state([SamplingParams(top_p=-0.8)])

    with pytest.raises(ValueError):
        get_sampling_state([SamplingParams(top_k=0)])

    with pytest.raises(ValueError):
        get_sampling_state([SamplingParams(top_k=0.8)])

    with pytest.raises(ValueError):
        get_sampling_state([SamplingParams(top_k=-2)])


def _test_top_p_top_k():
    def get_expected_result(logits, top_pks, filter_value=-float("Inf")):
        """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
            top_k >0: keep only top k tokens with highest probability (top-k filtering).
        """
        batch_size = len(top_pks)
        lst_logits = []
        for ii in range(batch_size):
            _logits = logits[ii]
            top_p, top_k = top_pks[ii]
            if top_p > 0.0:
                sorted_logits, sorted_indices = torch.sort(_logits, descending=True)
                cumulative_probs = torch.cumsum(
                    torch.softmax(sorted_logits, dim=-1), dim=-1
                )

                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift the indices to the right to keep also the first token above the threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                    ..., :-1
                ].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                _logits[indices_to_remove] = filter_value

            if top_k > 0:
                # Remove all tokens with a probability less than the last token of the top-k
                top_k_values = torch.topk(_logits, top_k)[0]
                # Use `None` to insert a singleton dimension
                # Equivalent to apply `squeeze` to the given dimension
                # e.g., arr.shape = [3,3]
                #       arr[:,:,None].shape = [3,3,1]
                indices_to_remove = _logits < top_k_values[..., -1, None]
                _logits[indices_to_remove] = filter_value

            lst_logits.append(_logits)
        return torch.stack(lst_logits)

    batch_size = 1
    top_p, top_k = 0.7, 5
    shape = (batch_size, vocab_size)
    logits = torch.rand(shape, dtype=dtype, device=dev)
    sampling_params = [
        SamplingParams(top_p=top_p, top_k=top_k) for _ in range(batch_size)
    ]
    sampling_state = get_sampling_state(sampling_params)
    new_logits = adjust_logits(logits, sampling_state, vocab_size)
    expected = logits.clone()
    expected = get_expected_result(expected, top_pks=[(top_p, top_k)])
    assert torch.allclose(expected, new_logits)

    batch_size = 3
    shape = (batch_size, vocab_size)
    logits = torch.rand(shape, dtype=dtype, device=dev)
    top_pks = [(0.7, 3), (0.5, 2), (0.8, 5)]
    sampling_params = [
        SamplingParams(top_p=top_p, top_k=top_k) for top_p, top_k in top_pks
    ]
    sampling_state = get_sampling_state(sampling_params)

    new_logits = adjust_logits(logits, sampling_state, vocab_size)
    expected = logits.clone()
    expected = get_expected_result(expected, top_pks)
    assert torch.allclose(expected, new_logits)


if __name__ == "__main__":
    _test_temperature()
    _test_logit_bias_checker()
    _test_logit_bias()
    _test_penalties_checker()
    _test_penalties()
    _test_top_p_top_k_checker()
    _test_top_p_top_k()
