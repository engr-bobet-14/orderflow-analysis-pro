CLAUDE.md

Implementation of order flow analysis — footprint charts, delta analytics, volume profile, pattern detection, and a Dash dashboard with WebSocket.

Forked from mahmoud20138/OrderFlow-Analysis-Pro.

IMPORTANT:

* Be brief and concise.
* Default branch is master.
* Prefer minimal, targeted edits over large refactors.
* Do not modify unrelated files.
* Use Graphify to understand code relationships before making broad changes.
* Activate the project environment before running Python commands: source .venv_orderflow/bin/activate.
* Run the smallest relevant test set after changes.
* Before the final response, check git diff and review all changes for correctness, regressions, incomplete work, and unintended edits.
* Do not stage, commit, amend, rebase, merge, or push Git changes unless explicitly asked.
* Use /codex:review when available, followed by a /codex:adversarial-review to challenge assumptions and catch missed issues.
* Fix issues found during review before responding.
* End completed implementation tasks with FINAL VERDICT: APPROVED FOR PRODUCTION or FINAL VERDICT: NOT APPROVED FOR PRODUCTION.
* Approve only when relevant validation passes and no known blocking issues remain.

Commands

* Environment: source .venv_orderflow/bin/activate
* Install: pip install -e ".[dev]"
* Run system: python -m orderflow_system.main
* Run dashboard: python -m orderflow_system.dashboard
* Tests: pytest

Architecture

* Authoritative architecture: orderflow_system/main.py
* Core package: orderflow_system/
* Analytics: orderflow_system/analytics/
* Patterns: orderflow_system/patterns/
* Signals: orderflow_system/signals/
* Dashboard: orderflow_system/dashboard/
* Configuration: orderflow_system/config/

Caveats

* Treat this repository as canonical, not the upstream fork.
* Do not preserve upstream implementation choices solely for compatibility.
* .claude/worktrees/ contains working copies and is not the canonical source tree.