import random
from itertools import product, permutations
from typing import List

import torch
import pytest
from mlc_serve.model.sampler import SamplingState, adjust_logits, sample, SamplingOutput
from mlc_serve.engine import SamplingParams, SAMPLING_EPS

dtype = torch.float32
dev = "cuda"


def get_sampling_state(sampling_params, past_output_tokens=None, prompt_masks=None, vocab_size=32000):
    batch_size = len(sampling_params)
    if past_output_tokens is None:
        past_output_tokens = [[] for _ in range(batch_size)]
    if prompt_masks is None:
        # Prepare empty prompt mask
        prompt_mask = torch.zeros((vocab_size,), dtype=torch.bool)
        prompt_masks = [prompt_mask] * batch_size
    _copy_stream: torch.cuda.Stream = torch.cuda.Stream()
    with torch.cuda.stream(_copy_stream):
        sampling_state = SamplingState.from_sampling_params(
            sampling_params,
            list_past_output_tokens=past_output_tokens,
            list_mask_prompt=prompt_masks,
            dtype=dtype,
            dev=dev,
            vocab_size=vocab_size,
        )
    torch.cuda.current_stream().wait_stream(_copy_stream)
    return sampling_state


def test_temperature_checker():
    # temperature must be in [0, 2]
    get_sampling_state([SamplingParams(temperature=0.0)])
    get_sampling_state([SamplingParams(temperature=0.8)])
    get_sampling_state([SamplingParams(temperature=1.3)])
    get_sampling_state([SamplingParams(temperature=2.0)])

    with pytest.raises(ValueError):
        get_sampling_state([SamplingParams(temperature=-0.1)])

    with pytest.raises(ValueError):
        get_sampling_state([SamplingParams(temperature=2.1)])

@pytest.mark.parametrize("batch_size", [1, 4, 8, 12])
def test_temperature(batch_size: int):
    vocab_size = 32000
    shape = (batch_size, vocab_size)
    logits = torch.rand(shape, dtype=dtype, device=dev)
    temperature = [0, 0.5, 1.0, 1.5, 2.0]
    for batch_temp in permutations(temperature, batch_size):
        sampling_params = [
            SamplingParams(temperature=val)
            for val in batch_temp
        ]
        expected = []
        for idx, val in enumerate(batch_temp):
            expected.append(logits[idx] / val if abs(val) > SAMPLING_EPS else logits[idx])
        sampling_state = get_sampling_state(sampling_params)
        new_logits = adjust_logits(logits, sampling_state, vocab_size)
        for idx, response in enumerate(new_logits):
            assert torch.allclose(expected[idx], response)


def test_logit_bias_checker():
    # logit bias values must be [-100, 100]
    # and indices in [0, vocab_size)

    vocab_size = 32000
    get_sampling_state([SamplingParams(logit_bias={1: 100, 3: -100, 2: 2})])
    get_sampling_state([SamplingParams(logit_bias={34: 0, 23: -0.5})])
    get_sampling_state([SamplingParams(logit_bias={1: 10, 3: -10, vocab_size - 1: 2})])
    get_sampling_state([SamplingParams(logit_bias={})])

    with pytest.raises(ValueError):
        get_sampling_state([SamplingParams(logit_bias={1: 2, 3: 105, 2: 2})])

    with pytest.raises(ValueError):
        get_sampling_state([SamplingParams(logit_bias={1: 99, 3: -101, 2: 2})])

    with pytest.raises(ValueError):
        get_sampling_state([SamplingParams(logit_bias={1: 10, 3: -10, vocab_size: 2})])

    with pytest.raises(ValueError):
        get_sampling_state([SamplingParams(logit_bias={1: 10, 3: -10, vocab_size + 100: 2})])

    with pytest.raises(ValueError):
        get_sampling_state([SamplingParams(logit_bias={1: 10, -1: -10})])

@pytest.mark.parametrize("batch_size", [1, 4])
def test_logit_bias(batch_size: int):
    vocab_size = 32000
    shape = (batch_size, vocab_size)
    logits = torch.rand(shape, dtype=dtype, device=dev)
    sampling_param = [{} for _ in range(batch_size)]
    for logit_bias_combination in permutations(
        product(
            [0, 31999, 724, 223],
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
            expected[num_batch][idx] += val
    for idx, logit_bias in enumerate(sampling_param):
        sampling_param[idx] = SamplingParams(logit_bias=logit_bias)
    sampling_state = get_sampling_state(sampling_param)
    new_logits = adjust_logits(logits, sampling_state, vocab_size)
    assert torch.allclose(expected, new_logits)


def test_penalties_checker():
    # repetition_penalty must be >0
    # frequency_penalty must be in [-2, 2]
    # precense_penalty must be in [-2, 2]

    # repetition_penalty
    get_sampling_state(
        [SamplingParams(repetition_penalty=0.1)], 
    )

    get_sampling_state(
        [SamplingParams(repetition_penalty=2.0)],
    )

    with pytest.raises(ValueError):
        get_sampling_state(
            [SamplingParams(repetition_penalty=0.0)],
        )

    with pytest.raises(ValueError):
        get_sampling_state(
            [SamplingParams(repetition_penalty=-2.0)],
        )

    # frequency_penalty
    get_sampling_state([SamplingParams(frequency_penalty=-2.0)])
    get_sampling_state([SamplingParams(frequency_penalty=2.0)])

    with pytest.raises(ValueError):
        get_sampling_state([SamplingParams(frequency_penalty=-2.1)])

    with pytest.raises(ValueError):
        get_sampling_state([SamplingParams(frequency_penalty=2.1)])

    # presence_penalty
    get_sampling_state([SamplingParams(presence_penalty=-2.0)])
    get_sampling_state([SamplingParams(presence_penalty=2.0)])

    with pytest.raises(ValueError):
        get_sampling_state([SamplingParams(presence_penalty=-2.1)])

    with pytest.raises(ValueError):
        get_sampling_state([SamplingParams(presence_penalty=2.1)])

    # combinations of penalties with valid values
    get_sampling_state(
        [SamplingParams(repetition_penalty=0.5, presence_penalty=0.5, frequency_penalty=0.0)],
    )

    # combinations of penalties with invalid values
    with pytest.raises(ValueError):
        get_sampling_state(
            [SamplingParams(repetition_penalty=-0.5, presence_penalty=0.5, frequency_penalty=0.0)],
        )

    with pytest.raises(ValueError):
        get_sampling_state(
            [SamplingParams(repetition_penalty=0.5, presence_penalty=2.5, frequency_penalty=0.0)],
        )

    with pytest.raises(ValueError):
        get_sampling_state(
            [SamplingParams(repetition_penalty=0.5, presence_penalty=0.5, frequency_penalty=-3.0)],
        )

    # penalties with valid values in multi-batch
    get_sampling_state(
        [
            SamplingParams(repetition_penalty=1.5),
            SamplingParams(presence_penalty=0.5),
            SamplingParams(frequency_penalty=0.0),
        ],
    )

    # penalties with invalid values in multi-batch
    with pytest.raises(ValueError):
        get_sampling_state(
            [
                SamplingParams(frequency_penalty=2.1),
                SamplingParams(repetition_penalty=1.1),
                SamplingParams(presence_penalty=1.1),
                SamplingParams(frequency_penalty=1.1),
            ],
        )

    with pytest.raises(ValueError):
        get_sampling_state(
            [
                SamplingParams(frequency_penalty=1.1),
                SamplingParams(repetition_penalty=1.1),
                SamplingParams(presence_penalty=1.1),
                SamplingParams(repetition_penalty=0.0),
            ],
        )

    with pytest.raises(ValueError):
        get_sampling_state(
            [
                SamplingParams(frequency_penalty=1.1),
                SamplingParams(repetition_penalty=1.1),
                SamplingParams(presence_penalty=1.1),
                SamplingParams(presence_penalty=2.1),
            ],
        )

@pytest.mark.parametrize("batch_size", [1, 3])
def test_penalties(batch_size: int):
    def _prepare_metadata(past_output_tokens, vocab_size):
        count_map = []
        for past_output_tokens_per_req in past_output_tokens:
            cnt = [0] * vocab_size
            for tok in past_output_tokens_per_req:
                cnt[tok] += 1
            count_map.append(cnt)

        count_tensor = torch.tensor(count_map, device=dev)
        mask_tensor = count_tensor > 0
        return count_tensor, mask_tensor


    def _get_expected_result(
        logits,
        count_map,
        mask,
        temperatures,
        repetition_penalties,
        presence_penalties,
        frequency_penalties,
    ):
        expected = torch.clone(logits)
        for i in range(batch_size):
            for j in range(len(expected[i])):
                if mask[i][j]:
                    expected[i][j] *= 1 / repetition_penalties[i] if expected[i][j] > 0 else repetition_penalties[i]
            temperature = 1.0 if temperatures[i] < SAMPLING_EPS else temperatures[i]
            expected[i] = (
                (expected[i]
                - count_map[i] * frequency_penalties[i]
                - mask[i] * presence_penalties[i])
                / temperature
            )
            print("DEBUG")
        return expected

    vocab_size = 512
    shape = (batch_size, vocab_size)
    logits = torch.rand(shape, dtype=dtype, device=dev)
    past_output_tokens = [[2, 2, 2, 3, 5]] * batch_size
    count_map, mask = _prepare_metadata(past_output_tokens, vocab_size)

    temperatures = [0.6]
    presence_penalties = [-2.0, 2.0]
    frequency_penalties = [-2.0, 2.0]
    repetition_penalties = [0.4, 1.0]
    for batch_params in permutations(
        product(
            temperatures,
            repetition_penalties,
            presence_penalties,
            frequency_penalties
        ),
        batch_size
    ):
        sampling_params = [
            SamplingParams(
                temperature=temp,
                repetition_penalty=rep_pen,
                presence_penalty=pr_pen,
                frequency_penalty=fr_pen,
                vocab_size=vocab_size
            )
            for temp, rep_pen, pr_pen, fr_pen in batch_params
        ]
        expected = _get_expected_result(
            logits,
            count_map,
            mask,
            [temp for temp, _, _, _ in batch_params],
            [rep_pen for _, rep_pen, _, _ in batch_params],
            [pr_pen for _, _, pr_pen, _ in batch_params],
            [fr_pen for _, _, _, fr_pen in batch_params],
        )
        sampling_state = get_sampling_state(
            sampling_params, past_output_tokens=past_output_tokens, vocab_size=vocab_size
        )
        new_logits = adjust_logits(logits, sampling_state, vocab_size)
        assert torch.allclose(expected, new_logits)


def test_top_p_top_k_checker():
    # top_p must be in (0, 1]
    # top_k must be in (0, vocab_size] (use -1 to consider all tokens)

    # top_p
    get_sampling_state([SamplingParams(top_p=0.6)])
    get_sampling_state([SamplingParams(top_p=0.1)])
    get_sampling_state([SamplingParams(top_p=1.0)])

    # top_k
    get_sampling_state([SamplingParams(top_k=3)])
    get_sampling_state([SamplingParams(top_k=-1)])
    get_sampling_state([SamplingParams(top_k=1)])

    # combinations of top_p, top_k with valid values
    get_sampling_state([SamplingParams(top_p=0.1, top_k=128)])
    get_sampling_state([SamplingParams(top_p=0.6, top_k=1)])
    get_sampling_state([SamplingParams(top_p=1.0, top_k=-1)])

    # combinations of top_p, top_k with invalid values
    with pytest.raises(ValueError):
        get_sampling_state([SamplingParams(top_p=0.0, top_k=128)])
    with pytest.raises(ValueError):
        get_sampling_state([SamplingParams(top_p=-1, top_k=-5)])
    with pytest.raises(ValueError):
        get_sampling_state([SamplingParams(top_p=5, top_k=0)])

    # top_p, top_k with valid values in multi-batch
    get_sampling_state(
        [
            SamplingParams(top_p=0.1, top_k=128),
            SamplingParams(top_p=0.5, top_k=1024),
            SamplingParams(top_p=1.0, top_k=8),
        ]
    )
    get_sampling_state(
        [SamplingParams(top_p=0.1), SamplingParams(top_p=0.5, top_k=1024), SamplingParams(top_k=8)]
    )
    get_sampling_state(
        [
            SamplingParams(top_p=1.0, top_k=-1),
            SamplingParams(top_p=0.5, top_k=32000),
        ]
    )

    # top_p, top_k with invalid values in multi-batch
    with pytest.raises(ValueError):
        get_sampling_state(
            [
                SamplingParams(top_p=-1, top_k=128),
                SamplingParams(top_p=0.5, top_k=12),
            ]
        )
    with pytest.raises(ValueError):
        get_sampling_state(
            [
                SamplingParams(top_p=0.1),
                SamplingParams(top_k=-2),
            ]
        )
    with pytest.raises(ValueError):
        get_sampling_state(
            [
                SamplingParams(top_p=1.1, top_k=-1),
                SamplingParams(top_p=0.5, top_k=64),
            ]
        )

def get_expected_result_by_top_pks(logits, top_pks, temps=None, filter_value=-float("Inf")):
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    Args:
        logits: logits distribution shape (vocabulary size)
        top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        top_k >0: keep only top k tokens with highest probability (top-k filtering).
    """
    batch_size = len(top_pks)
    lst_logits = []
    if temps is None:
        temps = [1.0] * batch_size
    for ii in range(batch_size):
        if temps[ii] < SAMPLING_EPS:
            temps[ii] = 1.0
        _logits = logits[ii] / temps[ii]
        top_p, top_k = top_pks[ii]
        if top_p > 0.0:
            sorted_logits, sorted_indices = torch.sort(_logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
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

@pytest.mark.parametrize("batch_size", [1, 4])
def test_top_p_top_k(batch_size: int):
    vocab_size = 32000
    shape = (batch_size, vocab_size)
    logits = torch.rand(shape, dtype=dtype, device=dev)
    for top_pks in permutations(
        product(
            [0.3, 0.7],          # top_p
            [128, 2048, 32000]   # top_k
        ),
        batch_size
    ):
        sampling_params = [SamplingParams(top_p=top_p, top_k=top_k) for top_p, top_k in top_pks]
        sampling_state = get_sampling_state(sampling_params)
        new_logits = adjust_logits(logits, sampling_state, vocab_size)
        expected = get_expected_result_by_top_pks(logits.clone(), top_pks)
        assert torch.allclose(expected, new_logits)


def test_logprobs_checker():
    get_sampling_state([SamplingParams(logprobs=False)])
    get_sampling_state([SamplingParams(logprobs=True)])
    get_sampling_state([SamplingParams(logprobs=True, top_logprobs=0)])
    get_sampling_state([SamplingParams(logprobs=True, top_logprobs=5)])

    with pytest.raises(ValueError):
        get_sampling_state([SamplingParams(logprobs=True, top_logprobs=-1)])

    with pytest.raises(ValueError):
        get_sampling_state([SamplingParams(logprobs=True, top_logprobs=6)])

    with pytest.raises(TypeError):
        get_sampling_state([SamplingParams(logprobs=True, top_logprobs=2.5)])

@pytest.mark.parametrize("batch_size", [1, 4, 8])
def test_logprobs(batch_size: int):
    vocab_size = 32000
    shape = (batch_size, vocab_size)
    logits = torch.rand(shape, dtype=dtype, device=dev)

    # No logprobs
    sampling_params = [SamplingParams(logprobs=False) for _ in range(batch_size)]
    sampling_state = get_sampling_state(sampling_params)
    output: SamplingOutput = sample(logits, sampling_state)
    assert all([logprob_response is None for logprob_response in output.logprob_infos])

    # Logprob only of a current token
    sampling_params = [SamplingParams(logprobs=True) for _ in range(batch_size)]
    sampling_state = get_sampling_state(sampling_params)
    output: SamplingOutput = sample(logits, sampling_state)
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
        sampling_state = get_sampling_state(sampling_params)
        output: SamplingOutput = sample(logits, sampling_state)
        assert len(output.logprob_infos) == batch_size
        for idx in range(batch_size):
            assert isinstance(output.logprob_infos[idx].current_token_id, int)
            assert isinstance(output.logprob_infos[idx].current_logprob, float)
            assert output.logprob_infos[idx].top_token_ids.nelement() != 0
            assert len(output.logprob_infos[idx].top_token_ids) == top_logprobs
            assert output.logprob_infos[idx].top_logprobs.nelement() != 0
            assert len(output.logprob_infos[idx].top_logprobs) == top_logprobs

@pytest.mark.skip(reason="""
    This test is currently broken. Need to validate correctness of this check
    and make sure that _apply_top_p_top_k from sampler.py does not produce too many -inf values
    """)
@pytest.mark.parametrize("batch_size", [1, 4, 8, 12])
def test_mixture_of_requests(batch_size: int):
    # Mixed temperature & top_p/top_ks
    vocab_size = 32000
    top_ps = list(torch.arange(1, 0, -0.01))
    top_ks = list(range(1, vocab_size + 1))
    temperatures = list(torch.arange(0, 2.1, 0.1))
    temp_weights = [0.5]
    temp_weights.extend([1 / (len(temperatures) - 1)] * (len(temperatures) - 1))
    top_ks.append(-1)
    for _ in range(10):
        shape = (batch_size, vocab_size)
        logits = torch.rand(shape, dtype=dtype, device=dev)
        top_pks = [(random.choice(top_ps), random.choice(top_ks)) for _ in range(batch_size)]
        temps = random.choices(temperatures, weights=temp_weights, k=batch_size)
        sampling_params = [
            SamplingParams(temperature=temps[i], top_p=top_p, top_k=top_k)
            for i, (top_p, top_k) in enumerate(top_pks)
        ]
        sampling_state = get_sampling_state(sampling_params)
        new_logits = adjust_logits(logits, sampling_state, vocab_size)
        expected = get_expected_result_by_top_pks(logits.clone(), top_pks, temps)
        assert torch.allclose(expected, new_logits)