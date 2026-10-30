"""Retired tick-cache diagnostic; retained as a documented command-line stub.

The four original arms compared already-fixed state-machine/clock defects using
removed raw-trade caches and obsolete detector settings. Those counterfactuals
cannot be preserved on information-driven bars. Historical JSON results in this
directory describe that retired experiment and are not current validation.

Use `python -m orderflow_system.validate_infobars --bar-type all --start
2023-11-19 --end 2023-11-24` for deterministic shared-pipeline/TradeLedger audits.
No study, parameter tuning, dataset write, or replay is performed by this stub.
"""

import argparse


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Describe the replacement validation")
    parser.parse_args(argv)
    print("Retired four-arm tick diagnostic. Run orderflow_system.validate_infobars "
          "for shared InfoBarPipeline and TradeLedger validation; see --help.")


if __name__ == "__main__":
    main()
