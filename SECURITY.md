# Security policy

## Reporting a vulnerability

Please **do not file a public issue.** Instead, use [GitHub's private
vulnerability reporting][private-report] for this repository. That route
keeps the disclosure channel off the public timeline until a fix is
ready.

[private-report]: https://github.com/RP2/grainbase/security/advisories/new

If the GitHub UI is unavailable, contact a maintainer through the
channels listed in the repository's social preview / About panel.

We aim to acknowledge reports within 72 hours and to ship or publicly
plan a fix within 30 days of triage, in line with industry norms for
open-source disclosures.

## Scope

- The grainbase engine (`crates/grainbase`).
- The grainbase CLI (`crates/grainbase-cli`).
- The CI workflows, supply-chain config, and build scripts shipped in
  this repository.

The supported targets for now are desktop CPU builds. Issues specific
to downstream forks are out of scope.

## Status

This project is **pre-alpha**. There is no published release, no
formal security patch line, and no bug-bounty. Treat the disclosure
process above as best-effort.
