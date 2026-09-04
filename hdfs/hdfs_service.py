import os
import torch

from groq import Groq

from hdfs.model import LogTransformer
from hdfs.utils import encode, pad_sequence


# ---------------------------------------
# Paths
# ---------------------------------------

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

CHECKPOINT_PATH = os.path.join(
    BASE_DIR,
    "checkpoints",
    "best_model.pt"
)


# ---------------------------------------
# Device
# ---------------------------------------

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)


# ---------------------------------------
# Groq
# ---------------------------------------

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

client = Groq(
    api_key=GROQ_API_KEY
)


# ---------------------------------------
# Load Transformer Model
# ---------------------------------------

VOCAB_SIZE = 31

model = LogTransformer(
    vocab_size=VOCAB_SIZE
).to(DEVICE)


if os.path.exists(CHECKPOINT_PATH):

    model.load_state_dict(
        torch.load(
            CHECKPOINT_PATH,
            map_location=DEVICE
        )
    )

    model.eval()

    print("✅ HDFS Transformer model loaded!")

else:

    raise FileNotFoundError(
        f"HDFS checkpoint not found: {CHECKPOINT_PATH}"
    )


# ---------------------------------------
# Predict Log
# ---------------------------------------

def predict_log(log_sequence: str):

    tokens = encode(log_sequence)

    # Prevent completely empty input
    if not tokens:

        return {
            "prediction": "Invalid Input",
            "confidence": 0
        }

    tokens, mask = pad_sequence(tokens)

    input_ids = torch.tensor(
        [tokens],
        dtype=torch.long
    ).to(DEVICE)

    attention_mask = torch.tensor(
        [mask],
        dtype=torch.long
    ).to(DEVICE)

    with torch.no_grad():

        outputs = model(
            input_ids,
            attention_mask
        )

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        confidence = probabilities.max().item()

        prediction = torch.argmax(
            probabilities,
            dim=1
        ).item()

    return {
        "prediction": (
            "Anomaly"
            if prediction == 1
            else "Normal"
        ),

        "confidence": round(
            confidence * 100,
            2
        )
    }


# ---------------------------------------
# Root Cause Analysis
# ---------------------------------------

def analyze_anomaly(log_sequence: str):

    prompt = f"""
You are an expert Hadoop Distributed File System (HDFS)
Site Reliability Engineer.

The following HDFS event sequence was detected
as an ANOMALY:

{log_sequence}

Perform a Root Cause Analysis.

Provide the response in this format:

ROOT CAUSE:
Explain the most probable root cause.

SEVERITY:
Low / Medium / High / Critical

ESTIMATED IMPACT:
Give an estimated impact percentage.

FAILURE POINT:
Explain where the sequence likely failed.

RECOMMENDED FIX:
Give clear resolution steps.

PREVENTION:
Give preventive measures.
"""

    try:

        response = client.chat.completions.create(

            model="openai/gpt-oss-20b",

            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],

            temperature=0.2
        )

        return response.choices[0].message.content

    except Exception as e:

        return f"Groq API Error: {str(e)}"


# ---------------------------------------
# Complete HDFS Pipeline
# ---------------------------------------

def analyze_hdfs_log(log_sequence: str):

    prediction_result = predict_log(
        log_sequence
    )

    # Invalid input
    if prediction_result["prediction"] == "Invalid Input":

        return {
            "success": False,

            "prediction": "Invalid Input",

            "confidence": 0,

            "rca": (
                "No valid HDFS event IDs found. "
                "Please provide sequences such as: "
                "E1 E5 E10 E3"
            )
        }


    # Normal log
    if prediction_result["prediction"] == "Normal":

        return {
            "success": True,

            "prediction": "Normal",

            "confidence": prediction_result["confidence"],

            "rca": (
                "No anomaly detected. "
                "The HDFS event sequence appears "
                "to be within normal operational behavior."
            )
        }


    # Anomaly
    rca = analyze_anomaly(
        log_sequence
    )

    return {
        "success": True,

        "prediction": "Anomaly",

        "confidence": prediction_result["confidence"],

        "rca": rca
    }