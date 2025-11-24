# Bittensor Wallet Setup

This guide covers creating and managing Bittensor wallets to participate in the Numinous subnet.

---

## Creating a New Wallet

Create a coldkey and hotkey pair:

```bash
# Create coldkey
btcli wallet new_coldkey --wallet.name miner

# Create hotkey
btcli wallet new_hotkey --wallet.name miner --wallet.hotkey default
```

**Important:**
- Save your mnemonic phrase securely
- Do NOT share your mnemonic with anyone
- Both coldkey and hotkey can be publicly shared (addresses, not mnemonics)
- You'll need your mnemonic to regenerate the wallet

## Regenerating an Existing Wallet

If you have an existing wallet mnemonic:

```bash
# Regenerate coldkey
btcli w regen_coldkey --wallet-path ~/.bittensor/wallets/ \
    --wallet-name miner --mnemonic "${MNEMONIC_COLDKEY}"

# Regenerate hotkey
btcli w regen_hotkey --wallet-name miner --wallet.hotkey default \
    --mnemonic "${MNEMONIC_HOTKEY}"
```

## Getting TAO Tokens

### For Testnet

Request faucet TAO at: https://app.minersunion.ai/testnet-faucet

### For Mainnet

Buy TAO on exchanges and withdraw to your wallet address.

**Note:** Registration cost fluctuates based on network demand.

## Registering on Subnet

### Testnet (netuid 155)

```bash
btcli subnet register --wallet.name miner --wallet.hotkey default --subtensor.network test
```

### Mainnet (netuid 6)

```bash
btcli subnet register --wallet.name miner --wallet.hotkey default --netuid 6
```

Follow the prompts:

```bash
>> Enter netuid (0): 155  # or 6 for mainnet
Your balance is: τ5.000000000
The cost to register by recycle is τ0.000000001
>> Do you want to continue? [y/n]: y
>> Enter password to unlock key: ********
✅ Registered
```

## Verifying Registration

Check your wallet status:

```bash
# Testnet
btcli wallet overview --wallet.name miner --subtensor.network test

# Mainnet
btcli wallet overview --wallet.name miner
```

You should see:
- UID assigned
- ACTIVE status
- Your balance

## Wallet Security Best Practices

- Store mnemonic offline in a secure location
- Use different wallets for testnet and mainnet
- Keep backups of your mnemonic
- Never share your mnemonic
- Never commit mnemonic to version control
- Avoid storing mnemonic in plain text files

## Wallet File Locations

Default wallet directory: `~/.bittensor/wallets/`

```
~/.bittensor/wallets/
  ├── miner/
  │   ├── coldkey
  │   ├── coldkeypub.txt
  │   └── hotkeys/
  │       └── default
  └── validator/
      ├── coldkey
      ├── coldkeypub.txt
      └── hotkeys/
          └── default
```

## Additional Resources

- **Bittensor Docs:** https://docs.bittensor.com/getting-started/wallets
- **Miner Guide:** https://docs.bittensor.com/miners/
- **Coldkey Guide:** https://docs.bittensor.com/getting-started/wallets
