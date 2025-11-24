import importlib.util
import json
import sys
import traceback


def main():
    print("[AGENT_RUNNER] Starting agent execution")

    try:
        # Read event data from input.json
        print("[AGENT_RUNNER] Reading input.json")
        with open("/sandbox/input.json", "r") as f:
            event_data = json.load(f)
        print(f"[AGENT_RUNNER] Event data loaded: {event_data.get('event_id', 'unknown')}")

        # Import agent module
        print("[AGENT_RUNNER] Loading agent.py")
        spec = importlib.util.spec_from_file_location("agent", "/sandbox/agent.py")
        if spec is None or spec.loader is None:
            raise Exception("Failed to load agent.py.")

        agent_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(agent_module)
        print("[AGENT_RUNNER] Agent module loaded successfully")

        # Check for agent_main() function
        if not hasattr(agent_module, "agent_main"):
            raise Exception(
                "Agent must have an 'agent_main' function. "
                "Expected signature: def agent_main(event_data: Dict[str, Any]) -> Dict[str, Any]."
            )

        print("[AGENT_RUNNER] Found agent_main() function")

        # Execute agent_main
        print("[AGENT_RUNNER] Calling agent_main()")
        prediction = agent_module.agent_main(event_data)
        print("[AGENT_RUNNER] agent_main() completed")

        # Validate prediction output
        if not isinstance(prediction, dict):
            raise Exception(f"agent_main() must return a dict, got {type(prediction).__name__}.")

        if "event_id" not in prediction:
            raise Exception("Prediction dict must include 'event_id' field.")

        if "prediction" not in prediction:
            raise Exception("Prediction dict must include 'prediction' field.")

        # Validate prediction value
        pred_value = prediction["prediction"]
        if not isinstance(pred_value, (int, float)):
            raise Exception(f"Prediction value must be a number, got {type(pred_value).__name__}.")

        if not (0.0 <= pred_value <= 1.0):
            raise Exception(f"Prediction must be between 0.0 and 1.0, got {pred_value}.")

        print(f"[AGENT_RUNNER] Prediction validated: {pred_value}")

        # Write successful output
        output = {"status": "success", "output": prediction}

        print("[AGENT_RUNNER] Writing output.json")
        with open("/sandbox/output.json", "w") as f:
            json.dump(output, f, indent=2)
        print("[AGENT_RUNNER] output.json written successfully")

    except Exception as e:
        # Handle any errors
        print(f"[AGENT_RUNNER] ERROR: {e}")
        traceback.print_exc(file=sys.stdout)

        output = {"status": "error", "error": str(e), "traceback": traceback.format_exc()}

        try:
            print("[AGENT_RUNNER] Writing error output.json")
            with open("/sandbox/output.json", "w") as f:
                json.dump(output, f, indent=2)
            print("[AGENT_RUNNER] Error output.json written")
        except Exception as write_error:
            print(f"[AGENT_RUNNER] FATAL: Failed to write output.json: {write_error}")

    print("[AGENT_RUNNER] Agent runner finished")


if __name__ == "__main__":
    main()
