# Live Trading Checklist

- `runtime.mode = "live"`
- `execution.broker = "ibkr"`
- `execution.paper_trading = false`
- `execution.allow_live_trading = true`
- `execution.confirm_live_account` exactly matches `execution.account`
- `runtime.allow_demo_components = false`
- no demo news backend or static JSON news source
- dedicated live `[persistence]` paths configured
- startup reconciliation clean before opening the session
- kill-switch commands tested on the live environment before enabling order flow
- health, metrics, alerts, and dead-letter paths monitored externally
- broker reconnect and stale-order cleanup procedures rehearsed
- secret distribution handled outside plain repo files
- human approval recorded for live promotion
