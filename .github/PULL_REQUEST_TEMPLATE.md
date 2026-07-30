## Summary

Briefly describe the change.

## Type

- [ ] Bug fix
- [ ] New feature
- [ ] Refactor (no behavior change)
- [ ] Documentation

## Pre-Submission Checklist

All items must be checked. Unchecked items will result in rejection.

### Architecture

- [ ] No new module violates the layer dependency rules in [ARCHITECTURE.md](ARCHITECTURE.md)
- [ ] Engine code does not import from `interception/`
- [ ] OS calls use `subprocess.run` with `check=True` (no silent failures)

### No Simulation Code

- [ ] Searched new code for `mock`, `simulate`, `fake`, `demo_data`, `generate_packet` — zero hits in `networksecurity/`
- [ ] No `np.random.randn()` or `random.randint()` in feature extraction or detection logic
- [ ] All test mocks live in `tests/`, not in `networksecurity/`

### Real Blocking

- [ ] Blocking logic calls `nf_packet.drop()` or `iptables -j DROP` via `IptablesManager`
- [ ] No in-memory flag (`is_blocked = True`) used as the sole blocking mechanism
- [ ] System call errors are logged at ERROR level, not silently ignored

### No Dead Code

- [ ] Every new function has at least one caller in the production path
- [ ] No `while True: sleep(N)` loops without real packet processing
- [ ] No `if False:` or permanently unreachable branches

### Platform Compatibility

- [ ] Import succeeds on macOS (Linux-only modules use lazy imports)
- [ ] `python -c "import networksecurity"` completes without error

## Testing

- [ ] Verified on macOS (import + pipeline + API endpoints)
- [ ] If modifying interception/: verified on Linux with real traffic

## Related Issues

Closes #
