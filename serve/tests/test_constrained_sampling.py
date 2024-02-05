import time

import torch
from transformers import AutoTokenizer
from pydantic import BaseModel

from mlc_serve.engine.constrained_sampling import JSONLogitsProcessor

class Structure(BaseModel):
    field1: int
    field2: str
    field3: float


def test_cs():
    schema = Structure.model_json_schema()
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
    start = time.perf_counter()
    proc = JSONLogitsProcessor(schema, tokenizer)
    end = time.perf_counter()
    print(f"Time to initialize Logits processor: {end-start}")
    logits = torch.full((32000,), 0.0)

    json_str = Structure(field1=1234, field2="Something in the meadow calls to me", field3=1e-3).model_dump_json()

    print(json_str)
    # Initialize the FSM state
    # start = time.perf_counter()
    # proc(0, [], logits)
    # end = time.perf_counter()
    # print(f"First round of logit processing: {end-start}")


    # Small out degree of fsm
    for i in range(50):
        proc.fsm_state = {0: 1}
        start = time.perf_counter()
        out = proc(0, [371], logits)
        end = time.perf_counter()
        if i==0:
            print(f"Number of entries greater than -10: {sum(out>-10)}")
    print(f"Low out degree of logit processing took: {(end-start)/50} seconds")

    # Large out degree of fsm
    out_degrees = [len(v) for v in proc.fsm.states_to_token_maps.values()]
    max_i = out_degrees.index(max(out_degrees))
    # print(f"{out_degrees=}, {max_i=}, {out_degrees[max_i]}")
    input_ids = tokenizer.encode("apple banana")
    for i in range(50):
        proc.fsm_state={0: max_i}
        start = time.perf_counter()
        out = proc(0, input_ids, logits)
        end = time.perf_counter()
        if i==0:
            print(f"Number of entries greater than -10: {sum(out>-10)}")
    print(f"High out degree round of logit processing took: {(end-start)/50} seconds")
    return
    for i in range(len(json_str)):
        # Incrementally tokenize and chop off start sequence
        print(f"trying to tokenize {json_str[:i]}")
        token_ids = tokenizer(json_str[i]).input_ids[0]
        print(token_ids)
        start = time.perf_counter()
        out = proc(0, token_ids, logits)
        end = time.perf_counter()
        print(f"Number of entries greater than -10: {sum(out>-10)}")
        print(f"{i}th round of logit processing of list {token_ids[:i]} took: {end-start} seconds")


if __name__ == "__main__":
    test_cs()
