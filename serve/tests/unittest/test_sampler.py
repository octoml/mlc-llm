import torch
import pytest
from mlc_serve.model.sampler import (
    SamplingState,
    adjust_logits, 
    sample, 
    SamplingOutput
)
from mlc_serve.engine import SamplingParams, SAMPLING_EPS

dtype = torch.float32
dev = "cuda"
vocab_size = 32000


def get_sampling_metadata(sampling_params, past_output_tokens=None):
    batch_size = len(sampling_params)
    if past_output_tokens is None:
        past_output_tokens = [[] for _ in range(batch_size)]
    _copy_stream: torch.cuda.Stream = torch.cuda.Stream()
    with torch.cuda.stream(_copy_stream):
        sampling_metadata = SamplingState.from_sampling_params(
            sampling_params,
            list_past_output_tokens=past_output_tokens,
            dtype=dtype,
            dev=dev,
            vocab_size=vocab_size,
        )
    torch.cuda.current_stream().wait_stream(_copy_stream)
    return sampling_metadata


def _test_temperature_checker():
    # temperature must be in [0, 2]
    get_sampling_metadata([SamplingParams(temperature=0.0)])
    get_sampling_metadata([SamplingParams(temperature=0.8)])
    get_sampling_metadata([SamplingParams(temperature=1.3)])
    get_sampling_metadata([SamplingParams(temperature=2.0)])

    with pytest.raises(ValueError):
        temperature = -0.1
        sampling_param = SamplingParams(temperature=temperature)
        get_sampling_metadata([sampling_param])

    with pytest.raises(ValueError):
        temperature = 2.1
        sampling_param = SamplingParams(temperature=temperature)
        get_sampling_metadata([sampling_param])

def _test_temperature():
    for batch_size in [1, 4, 8]:
        shape = (batch_size, vocab_size)
        logits = torch.rand(shape, dtype=dtype, device=dev)

        for temperature in [0, 0.5, 1.0, 1.5, 2.0]:
            # use same temperature
            sampling_params = [SamplingParams(temperature=temperature) for _ in range(batch_size)]
            expected = logits / temperature if abs(temperature) > SAMPLING_EPS else logits
            sampling_metadata = get_sampling_metadata(sampling_params)
            new_logits = adjust_logits(logits, sampling_metadata, vocab_size)
            for idx, response in enumerate(new_logits):
                assert torch.allclose(expected[idx], response)

        # use different temperature
        if batch_size > 1:
            temperature = [i % 3 for i in range(batch_size)]
            sampling_params = [SamplingParams(temperature=val) for val in temperature]
            for idx, val in enumerate(temperature):
                expected[idx] = logits[idx] / val if abs(val) > SAMPLING_EPS else logits[idx]
            sampling_metadata = get_sampling_metadata(sampling_params)
            new_logits = adjust_logits(logits, sampling_metadata, vocab_size)
            for idx, response in enumerate(new_logits):
                assert torch.allclose(expected[idx], response)

def _test_logit_bias_checker():
    # logit bias must be [-100, 100]
    with pytest.raises(ValueError):
        logit_bias = {1: 2, 3: 105, 2: 2}
        sampling_param = SamplingParams(logit_bias=logit_bias)
        get_sampling_metadata([sampling_param])

    with pytest.raises(ValueError):
        logit_bias = {1: 99, 3: -101, 2: 2}
        sampling_param = SamplingParams(logit_bias=logit_bias)
        get_sampling_metadata([sampling_param])

    logit_bias = {1: 100, 3: -100, 2: 2}
    sampling_param = SamplingParams(logit_bias=logit_bias)
    get_sampling_metadata([sampling_param])

    # TODO(@team): it seems like the valid range is [1,vocab_size]. Double check.
    logit_bias = {1: 10, 3: -10, vocab_size: 2}
    sampling_param = SamplingParams(logit_bias=logit_bias)
    get_sampling_metadata([sampling_param])

    with pytest.raises(ValueError):
        logit_bias = {0: 10, 3: -10}
        sampling_param = SamplingParams(logit_bias=logit_bias)
        get_sampling_metadata([sampling_param])

    with pytest.raises(ValueError):
        logit_bias = {1: 10, 3: -10, vocab_size + 100: 2}
        sampling_param = SamplingParams(logit_bias=logit_bias)
        get_sampling_metadata([sampling_param])

    with pytest.raises(ValueError):
        logit_bias = {1: 10, -1: -10}
        sampling_param = SamplingParams(logit_bias=logit_bias)
        get_sampling_metadata([sampling_param])


def _test_logit_bias():
    # test single batch
    batch_size = 1
    shape = (batch_size, vocab_size)
    logits = torch.rand(shape, dtype=dtype, device=dev)
    logit_bias = {1: -1, 3: 1, 2: 2}
    sampling_param = SamplingParams(logit_bias=logit_bias)
    sampling_metadata = get_sampling_metadata([sampling_param])

    expected = torch.clone(logits)
    for idx, val in logit_bias.items():
        expected[0][idx - 1] += val
    new_logits = adjust_logits(logits, sampling_metadata, vocab_size)
    assert torch.allclose(expected, new_logits)

    # test multi-batch
    batch_size = 3
    shape = (batch_size, vocab_size)
    logits = torch.rand(shape, dtype=dtype, device=dev)
    list_logit_bias = [{1: -1, 3: 1, 2: 2}, {4: 2, 5: 1}, {1: -10}]
    sampling_params = [
        SamplingParams(logit_bias=logit_bias) for logit_bias in list_logit_bias
    ]
    sampling_metadata = get_sampling_metadata(sampling_params)

    expected = torch.clone(logits)
    for batch_size in range(batch_size):
        logit_bias = list_logit_bias[batch_size]
        for idx, val in logit_bias.items():
            expected[batch_size][idx - 1] += val
    new_logits = adjust_logits(logits, sampling_metadata, vocab_size)
    assert torch.allclose(expected, new_logits)


def _test_penalties_checker():
    get_sampling_metadata([SamplingParams(presence_penalty=-2.0)])
    get_sampling_metadata([SamplingParams(frequency_penalty=-2.0)])
    get_sampling_metadata([SamplingParams(repetition_penalty=0.1)])
    get_sampling_metadata([SamplingParams(presence_penalty=2.0)])
    get_sampling_metadata([SamplingParams(frequency_penalty=2.0)])
    get_sampling_metadata([SamplingParams(repetition_penalty=2.0)])

    with pytest.raises(ValueError):
        get_sampling_metadata([SamplingParams(presence_penalty=-2.1)])

    with pytest.raises(ValueError):
        get_sampling_metadata([SamplingParams(frequency_penalty=-2.1)])

    with pytest.raises(ValueError):
        get_sampling_metadata([SamplingParams(repetition_penalty=0.0)])

    with pytest.raises(ValueError):
        get_sampling_metadata([SamplingParams(presence_penalty=2.1)])

    with pytest.raises(ValueError):
        get_sampling_metadata([SamplingParams(frequency_penalty=2.1)])

    with pytest.raises(ValueError):
        get_sampling_metadata(
            [
                SamplingParams(frequency_penalty=2.1),
                SamplingParams(repetition_penalty=1.1),
                SamplingParams(presence_penalty=1.1),
                SamplingParams(frequency_penalties=1.1)
            ]
        )

    with pytest.raises(ValueError):
        get_sampling_metadata(
            [
                SamplingParams(frequency_penalty=1.1),
                SamplingParams(repetition_penalty=1.1),
                SamplingParams(presence_penalty=1.1),
                SamplingParams(repetition_penalty=0.0),
            ]
        )

    with pytest.raises(ValueError):
        get_sampling_metadata(
            [
                SamplingParams(frequency_penalty=1.1),
                SamplingParams(repetition_penalty=1.1),
                SamplingParams(presence_penalty=1.1),
                SamplingParams(presence_penalty=2.1),
            ]
        )


def _test_penalties():
    def _prepare_metadata(past_output_tokens):
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

    def _get_expected_result(
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

    for batch_size in [1, 4, 8]:
        shape = (batch_size, vocab_size)
        logits = torch.rand(shape, dtype=dtype, device=dev)
        past_output_tokens = [[2, 2, 2, 3, 5]] * batch_size
        count_map, mask = _prepare_metadata(past_output_tokens)

        # presence_penalty
        presence_penalties = [-2.0, -1.4, -0.8, 0.0, 0.5, 1.0, 1.5, 2.0]
        for idx in range(len(presence_penalties)):
            sampling_params = [
                SamplingParams(
                    presence_penalty=presence_penalties[i % len(presence_penalties)]
                ) for i in range(idx, batch_size + idx)
            ]
            expected = _get_expected_result(
                logits, 
                count_map, 
                mask, 
                [0] * batch_size, 
                (2 * presence_penalties)[idx : batch_size + idx]
            )
            sampling_metadata = get_sampling_metadata(
                sampling_params, past_output_tokens=past_output_tokens
            )
            new_logits = adjust_logits(logits, sampling_metadata, vocab_size)
            assert torch.allclose(expected, new_logits)

        # frequency_penalty
        frequency_penalties = [-2.0, -1.4, -0.8, 0.0, 0.5, 1.0, 1.5, 2.0]
        for idx in range(len(frequency_penalties)):
            sampling_params = [
                SamplingParams(
                    frequency_penalty=frequency_penalties[i % len(frequency_penalties)]
                ) for i in range(idx, batch_size + idx)
            ]
            expected = _get_expected_result(
                logits, 
                count_map, 
                mask, 
                (2 * frequency_penalties)[idx : batch_size + idx],
                [0] * batch_size
            )
            sampling_metadata = get_sampling_metadata(
                sampling_params, past_output_tokens=past_output_tokens
            )
            new_logits = adjust_logits(logits, sampling_metadata, vocab_size)
            assert torch.allclose(expected, new_logits)

        # repetition_penalty
        for temperature in [0.0, 0.5, 1.0, 1.5, 2.0]:
            repetition_penalties = [0.1, 0.6, 1.0, 1.5, 1.8, 2.1, 2.5, 3.0]
            for idx in range(len(repetition_penalties)):
                sampling_params = [
                    SamplingParams(
                        temperature=temperature, 
                        repetition_penalty=repetition_penalties[i % len(repetition_penalties)]
                    ) for i in range(idx, batch_size + idx)
                ]
                expected = torch.clone(logits)
                for i in range(batch_size):
                    expected[i] = logits[i] / (temperature * repetition_penalties[(idx + i) % len(repetition_penalties)]) \
                            if abs(temperature) > SAMPLING_EPS else logits[i] / repetition_penalties[(idx + i) % len(repetition_penalties)]
                sampling_metadata = get_sampling_metadata(sampling_params)
                new_logits = adjust_logits(logits, sampling_metadata, vocab_size)
                for batch_idx, response in enumerate(new_logits):
                    assert torch.allclose(expected[batch_idx], response)

def _test_top_p_top_k_checker():
    get_sampling_metadata([SamplingParams(top_p=0.8)])
    get_sampling_metadata([SamplingParams(top_k=3)])

    get_sampling_metadata([SamplingParams(top_k=-1)])
    get_sampling_metadata([SamplingParams(top_k=1)])

    with pytest.raises(ValueError):
        get_sampling_metadata([SamplingParams(top_p=0.0)])

    with pytest.raises(ValueError):
        get_sampling_metadata([SamplingParams(top_p=-0.8)])

    with pytest.raises(ValueError):
        get_sampling_metadata([SamplingParams(top_k=0)])

    with pytest.raises(ValueError):
        get_sampling_metadata([SamplingParams(top_k=0.8)])

    with pytest.raises(ValueError):
        get_sampling_metadata([SamplingParams(top_k=-2)])


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
    sampling_metadata = get_sampling_metadata(sampling_params)
    new_logits = adjust_logits(logits, sampling_metadata, vocab_size)
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
    sampling_metadata = get_sampling_metadata(sampling_params)

    new_logits = adjust_logits(logits, sampling_metadata, vocab_size)
    expected = logits.clone()
    expected = get_expected_result(expected, top_pks)
    assert torch.allclose(expected, new_logits)

def _test_logprobs_checker():
    get_sampling_metadata([SamplingParams(logprobs=False)])
    get_sampling_metadata([SamplingParams(logprobs=True)])
    get_sampling_metadata([SamplingParams(logprobs=True, top_logprobs=0)])
    get_sampling_metadata([SamplingParams(logprobs=True, top_logprobs=5)])

    # TODO: Shouldn't ValueError be raised when logprobs is False but top_logprobs is also set?
    # with pytest.raises(ValueError):
    #     get_sampling_metadata([SamplingParams(logprobs=False, top_logprobs=0)])

    # with pytest.raises(ValueError):
    #     get_sampling_metadata([SamplingParams(logprobs=False, top_logprobs=5)])

    # with pytest.raises(ValueError):
    #     get_sampling_metadata([SamplingParams(logprobs=False, top_logprobs=-1)])

    with pytest.raises(ValueError):
        get_sampling_metadata([SamplingParams(logprobs=True, top_logprobs=-1)])

    with pytest.raises(ValueError):
        get_sampling_metadata([SamplingParams(logprobs=True, top_logprobs=6)])

    with pytest.raises(ValueError):
        get_sampling_metadata([SamplingParams(logprobs=True, top_logprobs=2.5)])


def _test_logprobs():
    for batch_size in [1, 4, 8]:
        shape = (batch_size, vocab_size)
        logits = torch.rand(shape, dtype=dtype, device=dev)

        # No logprobs
        sampling_params = [
            SamplingParams(logprobs=False) for _ in range(batch_size)
        ]
        sampling_metadata = get_sampling_metadata(sampling_params)
        output: SamplingOutput = sample(logits, sampling_metadata)
        assert all([logprob_response is None for logprob_response in output.logprob_infos])

        # Logprob only of a current token
        sampling_params = [
            SamplingParams(logprobs=True) for _ in range(batch_size)
        ]
        sampling_metadata = get_sampling_metadata(sampling_params)
        output: SamplingOutput = sample(logits, sampling_metadata)
        assert len(output.logprob_infos) == batch_size
        for idx in range(batch_size):
            assert isinstance(output.logprob_infos[idx].current_token_id, int)
            assert isinstance(output.logprob_infos[idx].current_logprob, float)
            assert output.logprob_infos[idx].top_token_ids.nelement() == 0
            assert output.logprob_infos[idx].top_logprobs.nelement() == 0

        # Top-k logprobs
        for top_logprobs in [1, 3, 5]:
            sampling_params = [
                SamplingParams(logprobs=True, top_logprobs=top_logprobs) for _ in range(batch_size)
            ]
            sampling_metadata = get_sampling_metadata(sampling_params)
            output: SamplingOutput = sample(logits, sampling_metadata)
            assert len(output.logprob_infos) == batch_size
            for idx in range(batch_size):
                assert isinstance(output.logprob_infos[idx].current_token_id, int)
                assert isinstance(output.logprob_infos[idx].current_logprob, float)
                assert output.logprob_infos[idx].top_token_ids.nelement() != 0
                assert len(output.logprob_infos[idx].top_token_ids) == top_logprobs
                assert output.logprob_infos[idx].top_logprobs.nelement() != 0
                assert len(output.logprob_infos[idx].top_logprobs) == top_logprobs

if __name__ == "__main__":
    _test_temperature()
    _test_logit_bias_checker()
    _test_logit_bias()
    _test_penalties_checker()
    _test_penalties()
    _test_top_p_top_k_checker()
    _test_top_p_top_k()
    _test_logprobs_checker()
    _test_logprobs()