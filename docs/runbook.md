# Runbook

## Normal Startup

1. Verify broker mode, account, and `strategy_id`.
2. Confirm the kill switch is not engaged.
3. Run `python main.py --config <config> health` and inspect the last reconciliation result.
4. Start the runtime with `python main.py --config <config>`.

## If The Kill Switch Engages

1. Stop sending new market events or stop the process.
2. Read the latest alert and dead-letter journals in `state/`.
3. Run `python main.py --config <config> health`.
4. Reconcile broker positions and open orders manually.
5. Release only after the root cause is understood:

```bash
python main.py --config <config> kill-switch release --reason "operator verified recovery"
```

## Startup Reconciliation Mismatch

Paper mode:
- inspect `health` and `reconciliation`
- verify whether the persisted state belongs to the same strategy/account
- clear stale paper state by removing the environment-specific state store only if the mismatch is known to be non-economic

Live mode:
- do not bypass
- reconcile broker positions, executions, and open orders first
- restart only after the mismatch is resolved

## Dead Letters

Dead letters are written to:

- `state/dead_letters.jsonl`

Investigate:

- `stage`
- `error_type`
- serialized event payload

If dead letters are recurring, treat that as an incident even if the process remains alive.

## Replay For Incident Review

```bash
python main.py --config <config> replay --audit-path state/events.jsonl
```

Use replay to validate that a fix changes behavior intentionally.

## High Latency

1. Run `python main.py --config <config> metrics`.
2. Inspect the top bottleneck in the latency section.
3. Confirm the market hot path is still using incremental features and background news.
4. If latency breaches persist, engage the kill switch and investigate before continuing.
