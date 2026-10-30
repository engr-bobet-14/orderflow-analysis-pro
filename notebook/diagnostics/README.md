# Historical diagnostic outputs

`post_fix_verification_results.json` and `low_trade_count_four_arm_results.json`
are historical outputs of the retired tick-cache experiment. They are **not
validation evidence for the information-driven-bar pipeline** and remain here
only as records of that earlier experiment.

For current validation, run from the checkout root:

```sh
python -m orderflow_system.validate_infobars --bar-type all --start 2023-11-19 --end 2023-11-24
```

The current validator replays bars through `InfoBarPipeline.process_bar` and
checks captured executions through the shared `TradeLedger`. The retired
`low_trade_count_four_arm.py` command documents this replacement via `--help` or
`--dry-run`; it does not regenerate the historical JSON files.
