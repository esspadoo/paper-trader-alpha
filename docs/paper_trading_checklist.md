# Paper Trading Checklist

- `runtime.mode = "paper"`
- `execution.broker = "paper"` or IBKR paper account only
- `execution.allow_live_trading = false`
- unique `strategy_id` for the paper deployment
- environment-specific `[persistence]` paths configured
- `python main.py --config <config> health` shows no active kill switch
- startup reconciliation status reviewed
- dead-letter, alert, and audit journals writable
- smoke run completed successfully
- operator knows the kill-switch commands
- replay command validated against the latest audit journal
