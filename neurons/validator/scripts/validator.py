import argparse

from bittensor import Subtensor, Wallet


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--subtensor.network", type=str, default="finney")
    parser.add_argument("--wallet.name", type=str, default="default")
    parser.add_argument("--wallet.hotkey", type=str, default="default")
    parser.add_argument("--wallet.path", type=str, default="~/.bittensor/wallets/")

    return parser.parse_args()


def main(netuid: int):
    args = get_args()

    client = Subtensor(getattr(args, "subtensor.network"))
    metagraph = client.subnets.metagraph(netuid)
    wallet = Wallet(
        name=getattr(args, "wallet.name"),
        hotkey=getattr(args, "wallet.hotkey"),
        path=getattr(args, "wallet.path"),
    )

    wallet_uid = metagraph.hotkeys.index(wallet.hotkey.ss58_address)
    top_64_stake = sorted(neuron.total_stake for neuron in metagraph.neurons)[-64:]

    print(
        f"Current requirement for validator permits based on the top 64 stake stands at {min(top_64_stake)}"
    )

    print(f"Wallet hotkey stake {metagraph.neuron(wallet_uid).total_stake}")

    print(f"Validator permit: {metagraph.neuron(wallet_uid).validator_permit}")


if __name__ == "__main__":
    # Example run command
    # python neurons/validator/scripts/validator.py \
    #     --subtensor.network test \
    #     --wallet.name validator \
    #     --wallet.hotkey default

    main(netuid=155)
