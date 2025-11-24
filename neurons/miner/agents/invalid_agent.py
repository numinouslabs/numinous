import random

import bittensor as bt  # Non-supported library


def agent(event_data: dict) -> dict:  # Wrong entrypoint name -> "agent_main"
    # Dummy usage
    keypair = bt.Keypair.create_from_uri("//Alice")
    forecast = random.random()
    reasoning = f"I used random.random() to predict {forecast}. Keypair: {keypair.ss58_address}"
    return {
        "event_id": event_data["event_id"],
        "probability": forecast,
        "reasoning": reasoning,
    }
