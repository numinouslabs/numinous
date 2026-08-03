def agent_main(event_data: dict) -> dict:
    return {
        "event_id": event_data["event_id"],
        "prediction": 0.5,
        "reasoning": "I think the probability of the event is 0.5",
    }
