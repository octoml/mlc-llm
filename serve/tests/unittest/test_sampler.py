from itertools import product, permutations
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
        get_sampling_metadata([SamplingParams(temperature=-0.1)])

    with pytest.raises(ValueError):
        get_sampling_metadata([SamplingParams(temperature=2.1)])

def _test_temperature():
    # single-batch
    shape = (1, vocab_size)
    logits = torch.rand(shape, dtype=dtype, device=dev)
    for temperature in [0, 0.5, 1.0, 1.5, 2.0]:
        sampling_params = [SamplingParams(temperature=temperature)]
        expected = logits / temperature if abs(temperature) > SAMPLING_EPS else logits
        sampling_metadata = get_sampling_metadata(sampling_params)
        new_logits = adjust_logits(logits, sampling_metadata, vocab_size)
        assert torch.allclose(expected, new_logits)

    # multi-batch
    for batch_size in [4, 8, 12]:
        shape = (batch_size, vocab_size)
        logits = torch.rand(shape, dtype=dtype, device=dev)
        temperature = [i % 3 for i in range(batch_size)] # temperature from {0, 1, 2}
        sampling_params = [SamplingParams(temperature=val) for val in temperature]
        expected = []
        for idx, val in enumerate(temperature):
            expected.append(logits[idx] / val if abs(val) > SAMPLING_EPS else logits[idx])
        sampling_metadata = get_sampling_metadata(sampling_params)
        new_logits = adjust_logits(logits, sampling_metadata, vocab_size)
        for idx, response in enumerate(new_logits):
            assert torch.allclose(expected[idx], response)

def _test_logit_bias_checker():
    # logit bias values must be [-100, 100]
    get_sampling_metadata([SamplingParams(logit_bias={1: 100, 3: -100, 2: 2})])
    get_sampling_metadata([SamplingParams(logit_bias={34: 0, 23: -0.5})])
    # TODO(@team): it seems like the valid range is [1,vocab_size]. Double check.
    get_sampling_metadata([SamplingParams(logit_bias={1: 10, 3: -10, vocab_size: 2})])
    get_sampling_metadata([SamplingParams(logit_bias={})])

    with pytest.raises(ValueError):
        get_sampling_metadata([
            SamplingParams(logit_bias={1: 2, 3: 105, 2: 2})
        ])

    with pytest.raises(ValueError):
        get_sampling_metadata([
            SamplingParams(logit_bias={1: 99, 3: -101, 2: 2})
        ])

    with pytest.raises(ValueError):
        get_sampling_metadata([
            SamplingParams(logit_bias={0: 10, 3: -10})
        ])

    with pytest.raises(ValueError):
        get_sampling_metadata([
            SamplingParams(logit_bias={1: 10, 3: -10, vocab_size + 100: 2})
        ])

    with pytest.raises(ValueError):
        get_sampling_metadata([
            SamplingParams(logit_bias={1: 10, -1: -10})
        ])


def _test_logit_bias():
    for batch_size in [1, 4]:
        shape = (batch_size, vocab_size)
        logits = torch.rand(shape, dtype=dtype, device=dev)
        sampling_param = [dict() for _ in range(batch_size)]
        for logit_bias_combination in permutations(
            product(
                [1, 32000, 724, 223],
                [100, -100, -12.5, 0.05]
            ),
            batch_size
        ):
            for num_batch in range(len(logit_bias_combination)):
                logit_index, logit_bias = logit_bias_combination[num_batch]
                sampling_param[num_batch].update({logit_index: logit_bias})
        expected = torch.clone(logits)
        for num_batch in range(batch_size):
            for idx, val in sampling_param[num_batch].items():
                expected[num_batch][idx - 1] += val
        for idx, logit_bias in enumerate(sampling_param):
            sampling_param[idx] = SamplingParams(logit_bias=logit_bias)
        sampling_metadata = get_sampling_metadata(sampling_param)
        new_logits = adjust_logits(logits, sampling_metadata, vocab_size)
        assert torch.allclose(expected, new_logits)

def _test_penalties_checker():
    # repetition_penalty
    get_sampling_metadata([SamplingParams(repetition_penalty=0.1)])
    get_sampling_metadata([SamplingParams(repetition_penalty=2.0)])

    with pytest.raises(ValueError):
        get_sampling_metadata([SamplingParams(repetition_penalty=0.0)])
    
    with pytest.raises(ValueError):
        get_sampling_metadata([SamplingParams(repetition_penalty=-2.0)])

    # frequency_penalty
    get_sampling_metadata([SamplingParams(frequency_penalty=-2.0)])
    get_sampling_metadata([SamplingParams(frequency_penalty=2.0)])

    with pytest.raises(ValueError):
        get_sampling_metadata([SamplingParams(frequency_penalty=-2.1)])

    with pytest.raises(ValueError):
        get_sampling_metadata([SamplingParams(frequency_penalty=2.1)])

    # presence_penalty
    get_sampling_metadata([SamplingParams(presence_penalty=-2.0)])
    get_sampling_metadata([SamplingParams(presence_penalty=2.0)])

    with pytest.raises(ValueError):
        get_sampling_metadata([SamplingParams(presence_penalty=-2.1)])

    with pytest.raises(ValueError):
        get_sampling_metadata([SamplingParams(presence_penalty=2.1)])

    # combinations of penalties with valid values
    get_sampling_metadata([SamplingParams(
        repetition_penalty=0.5, 
        presence_penalty=0.5, 
        frequency_penalty=0.0)
    ])

    # combinations of penalties with invalid values
    with pytest.raises(ValueError):
        get_sampling_metadata([SamplingParams(
            repetition_penalty=-0.5, 
            presence_penalty=0.5, 
            frequency_penalty=0.0)
        ])

    with pytest.raises(ValueError):
        get_sampling_metadata([SamplingParams(
            repetition_penalty=0.5, 
            presence_penalty=2.5, 
            frequency_penalty=0.0)
        ])

    with pytest.raises(ValueError):
        get_sampling_metadata([SamplingParams(
            repetition_penalty=0.5, 
            presence_penalty=0.5, 
            frequency_penalty=-3.0)
        ])

    # penalties with valid values in multi-batch
    get_sampling_metadata(
        [
            SamplingParams(repetition_penalty=1.5),
            SamplingParams(presence_penalty=0.5),
            SamplingParams(frequency_penalty=0.0)
        ]
    )

    # penalties with invalid values in multi-batch
    with pytest.raises(ValueError):
        get_sampling_metadata(
            [
                SamplingParams(frequency_penalty=2.1),
                SamplingParams(repetition_penalty=1.1),
                SamplingParams(presence_penalty=1.1),
                SamplingParams(frequency_penalty=1.1)
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
    # top_p must be in (0, 1]
    # top_k must be in (0, vocab_size] (use -1 to consider all tokens)

    # top_p
    get_sampling_metadata([SamplingParams(top_p=0.6)])
    get_sampling_metadata([SamplingParams(top_p=0.1)])
    get_sampling_metadata([SamplingParams(top_p=1.0)])

    # top_k
    get_sampling_metadata([SamplingParams(top_k=3)])
    get_sampling_metadata([SamplingParams(top_k=-1)])
    get_sampling_metadata([SamplingParams(top_k=1)])

    # combinations of top_p, top_k with valid values
    get_sampling_metadata([SamplingParams(top_p=0.1, top_k=128)])
    get_sampling_metadata([SamplingParams(top_p=0.6, top_k=1)])
    get_sampling_metadata([SamplingParams(top_p=1.0, top_k=-1)])

    # combinations of top_p, top_k with invalid values
    with pytest.raises(ValueError):
        get_sampling_metadata([SamplingParams(top_p=0.0, top_k=128)])
    with pytest.raises(ValueError):
        get_sampling_metadata([SamplingParams(top_p=-1, top_k=-5)])
    with pytest.raises(ValueError):
        get_sampling_metadata([SamplingParams(top_p=5, top_k=0)])

    # top_p, top_k with valid values in multi-batch
    get_sampling_metadata([
        SamplingParams(top_p=0.1, top_k=128),
        SamplingParams(top_p=0.5, top_k=1024),
        SamplingParams(top_p=1.0, top_k=8)
    ])
    get_sampling_metadata([
        SamplingParams(top_p=0.1),
        SamplingParams(top_p=0.5, top_k=1024),
        SamplingParams(top_k=8)
    ])
    get_sampling_metadata([
        SamplingParams(top_p=1.0, top_k=-1),
        SamplingParams(top_p=0.5, top_k=32000),
    ])

    # top_p, top_k with invalid values in multi-batch
    with pytest.raises(ValueError):
        get_sampling_metadata([
            SamplingParams(top_p=-1, top_k=128),
            SamplingParams(top_p=0.5, top_k=12),
        ])
    with pytest.raises(ValueError):
        get_sampling_metadata([
            SamplingParams(top_p=0.1),
            SamplingParams(top_k=-2),
        ])
    with pytest.raises(ValueError):
        get_sampling_metadata([
            SamplingParams(top_p=1.1, top_k=-1),
            SamplingParams(top_p=0.5, top_k=64),
        ])

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

    for batch_size in [1, 4]:
        shape = (batch_size, vocab_size)
        logits = torch.rand(shape, dtype=dtype, device=dev)
        for top_pks in permutations(
            product(
                [0.3, 0.7],    # top_p
                [128, 2048, 32000]  # top_k
            ),
            batch_size
        ):
            sampling_params = [
                SamplingParams(top_p=top_p, top_k=top_k) for top_p, top_k in top_pks
            ]
            sampling_metadata = get_sampling_metadata(sampling_params)
            new_logits = adjust_logits(logits, sampling_metadata, vocab_size)
            expected = get_expected_result(logits.clone(), top_pks)
            assert torch.allclose(expected, new_logits)

def _test_logprobs_checker():
    get_sampling_metadata([SamplingParams(logprobs=False)])
    get_sampling_metadata([SamplingParams(logprobs=True)])
    get_sampling_metadata([SamplingParams(logprobs=True, top_logprobs=0)])
    get_sampling_metadata([SamplingParams(logprobs=True, top_logprobs=5)])

    with pytest.raises(ValueError):
        get_sampling_metadata([SamplingParams(logprobs=True, top_logprobs=-1)])

    with pytest.raises(ValueError):
        get_sampling_metadata([SamplingParams(logprobs=True, top_logprobs=6)])

    with pytest.raises(TypeError):
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
    _test_temperature_checker()
    _test_temperature()
    _test_logit_bias_checker()
    _test_logit_bias()
    _test_penalties_checker()
    _test_penalties()
    _test_top_p_top_k_checker()
    _test_top_p_top_k()
    _test_logprobs_checker()
    _test_logprobs()