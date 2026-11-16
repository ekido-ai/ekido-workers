# ekido-workers

The software providers run on their own devices: the device **agent** (`agent/`), the training **runtime** (`runtime/`), the quality **verifier** (`verifier/`) and the **provider kit** (`provider-kit/`). Public so anyone lending a device can read exactly what runs on it.

Part of [Ekido](https://github.com/ekido-ai): your own small AI model, trained on spare devices and paid for only when it works. The shared library, the API contract and the evidence behind every number live in [ekido-core](https://github.com/ekido-ai/ekido-core).

## Rules

- Every change is a pull request with the full report (the PR template); the **Gate** check refuses it otherwise.
- Testnet and test money only. Never commit keys, `.env` contents, wallet keys or customer data.
- Every number we quote is measured and has its raw file.

Licence: all rights reserved (see LICENSE).
