# Security policy

## Project status

Antigen is a portfolio prototype, not a production security product. Its
classifier can miss prompt-injection attempts and can flag benign text. Do not
use it as the only control protecting sensitive data, tools, or systems.

Security reports are still welcome. They help keep the demonstration, API, and
packaging safe for people who run the project locally.

## Supported versions

| Version | Security updates |
|---|---|
| `main` | Best-effort fixes |
| `0.1.x` | Best-effort fixes |
| Earlier versions | Not supported |

There is no guaranteed response or patch schedule. When a fix is practical, it
will normally be applied to `main`; the project does not currently publish
backported maintenance releases.

## Reporting a vulnerability

Please do not publish exploit details in a public issue.

1. Use the repository's **Security** tab to submit a private vulnerability
   report.
2. Include the affected revision, impact, reproduction steps, and any suggested
   mitigation.
3. Remove secrets, personal data, and unrelated system information from the
   report.

If private reporting is unavailable, open a public issue containing only a
request for a private reporting channel. Do not include vulnerability details
in that issue.

You can expect a best-effort acknowledgement and an assessment of whether the
report is in scope. Please allow time for a fix before public disclosure.

## Scope

In-scope reports include vulnerabilities in the API, command-line interface,
web demo, container configuration, dependencies, and project packaging.

Model-quality limitations such as false positives, false negatives, adversarial
examples, and dataset gaps are useful findings, but they are not software
security vulnerabilities by themselves. Please report those as regular issues
unless they also demonstrate a concrete security flaw in the application.
