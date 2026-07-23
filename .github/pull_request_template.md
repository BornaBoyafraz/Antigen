## Summary

Describe the problem and the approach taken.

## Verification

List automated checks and any manual testing performed.

- [ ] `.venv/bin/ruff check .`
- [ ] `.venv/bin/mypy .`
- [ ] `.venv/bin/pytest -q`

## Change checklist

- [ ] New behavior and bug fixes have focused tests.
- [ ] Python changes remain compatible with Python 3.10 and 3.12.
- [ ] New top-level runtime modules are wired into packaging, the container,
      and packaging tests.
- [ ] Model, dataset, security, and performance claims reflect measured results
      and state meaningful limits.
- [ ] Visible web-demo changes include screenshots or a short recording.
- [ ] Generated model binaries, local caches, and environment files are not
      included.

## Related issue

Link the issue this change addresses, if any.
