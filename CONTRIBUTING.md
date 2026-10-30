# Contributing to OrderFlow Analysis Pro

Thank you for your interest in contributing!

## Getting Started

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Make your changes
4. Run tests: `pytest` (and `python -m orderflow_system.validate_infobars` if analytics changed)
5. Submit a pull request

## Development Setup

```bash
pip install -e ".[dev]"
```

## Areas for Contribution

- **New pattern detectors** — add to `patterns/` directory
- **Calibration** — tune detector thresholds against the info-bar dataset
- **Live Binance feed** — emit `InfoBar`s into `InfoBarPipeline`
- **Dashboard improvements** — frontend or API enhancements
- **Documentation** — guides, tutorials, API docs
- **Bug fixes** — check GitHub Issues

## Code Style

- Python 3.10+ with type hints
- Dataclasses for data models
- Async/await for I/O; analytics stay synchronous and deterministic (`pipeline.py`)
- Never fabricate price-level / trade-level data the bars do not contain
- Follow existing patterns in each module

## Reporting Issues

Use GitHub Issues to report:
- Bugs or unexpected behavior
- Feature requests
- Documentation errors
- Instrument configuration issues

## Pull Request Process

1. Ensure tests pass
2. Add tests for new features
3. Update documentation if needed
4. Keep PRs focused — one feature or fix per PR
