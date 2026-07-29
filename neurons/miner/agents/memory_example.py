"""
Example agent showing how to carry a belief across intervals using memory.

Re-forecast events are run once per interval until cutoff, and the validator
gives you back what you stored last time:

    in   ->  event_data["memory"]   your blob from the previous interval, or
                                    None on the first run for this event
    out  ->  the "memory" key       whatever you want handed back next time

The blob is yours. The validator never reads it - it only transports it - so
the format is entirely your choice. Two constraints: it must be a string, and
anything past 32768 characters is truncated, so keep it bounded rather than
appending forever.

Memory is scoped per (miner, event), and `agent_main(event_data)` is unchanged.
An agent that ignores memory keeps working exactly as before.

Replace `forecast_from_evidence` with your real research + inference; the rest
of this file is the memory pattern itself.
"""

import json

PRIOR_WEIGHT = 0.6
MAX_HISTORY = 10


def load_memory(raw_memory: str | None) -> dict:
    if not raw_memory:
        return {}

    try:
        return json.loads(raw_memory)
    except json.JSONDecodeError:
        return {}


def forecast_from_evidence(event_data: dict) -> float:
    return 0.5


def update_belief(prior_belief: float | None, fresh_estimate: float) -> float:
    if prior_belief is None:
        return fresh_estimate

    return PRIOR_WEIGHT * prior_belief + (1 - PRIOR_WEIGHT) * fresh_estimate


def build_memory(belief: float, history: list[float], fresh_estimate: float) -> str:
    return json.dumps(
        {
            "belief": round(belief, 4),
            "history": (history + [round(fresh_estimate, 4)])[-MAX_HISTORY:],
        }
    )


def agent_main(event_data: dict) -> dict:
    memory = load_memory(event_data.get("memory"))
    prior_belief = memory.get("belief")
    history = memory.get("history", [])

    print(f"[AGENT] {event_data['event_id']} prior={prior_belief} seen={len(history)}")

    fresh_estimate = forecast_from_evidence(event_data)
    belief = update_belief(prior_belief, fresh_estimate)

    print(f"[AGENT] fresh={fresh_estimate:.4f} belief={belief:.4f}")

    return {
        "event_id": event_data["event_id"],
        "prediction": belief,
        "reasoning": (
            f"Blended a fresh estimate of {fresh_estimate:.2f} with a prior belief of "
            f"{prior_belief:.2f} from {len(history)} earlier interval(s)."
            if prior_belief is not None
            else f"First forecast for this event, estimated {fresh_estimate:.2f}."
        ),
        "memory": build_memory(belief, history, fresh_estimate),
    }
