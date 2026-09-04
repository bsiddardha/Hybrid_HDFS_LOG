import re


MAX_LEN = 128


# Vocabulary
event2id = {
    "<PAD>": 0,
    "<UNK>": 1
}

id2event = {
    0: "<PAD>",
    1: "<UNK>"
}


# E1 → E29
for i in range(1, 30):

    event = f"E{i}"

    event2id[event] = i + 1
    id2event[i + 1] = event


def encode(sequence: str):

    tokens = re.findall(
        r"E\d+",
        sequence
    )

    return [
        event2id.get(token, 1)
        for token in tokens
    ]


def pad_sequence(
    sequence,
    max_len=MAX_LEN
):

    sequence = sequence[:max_len]

    attention_mask = [1] * len(sequence)

    pad_length = max_len - len(sequence)

    sequence += [0] * pad_length

    attention_mask += [0] * pad_length

    return sequence, attention_mask