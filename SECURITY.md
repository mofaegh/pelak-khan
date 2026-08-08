# Security Policy

## Supported versions

Pelak-Khan is currently in beta. Security fixes are primarily applied to the latest development branch and the latest published release where practical.

## Reporting a vulnerability

If you discover a security issue, avoid posting sensitive exploit details, credentials, private vehicle data, or other confidential information in a public issue.

If GitHub Private Vulnerability Reporting is enabled for this repository, use it. Otherwise, open a minimal public issue asking the maintainer for a private contact channel without including exploit details.

Useful information in a report includes:

- affected version or commit
- operating system and environment
- concise reproduction steps
- expected and observed behavior
- security impact
- relevant logs with secrets and personal data removed

## Deployment notes

The standard desktop configuration binds the FastAPI service to `127.0.0.1`. Do not expose the backend directly to an untrusted network without adding appropriate authentication, TLS termination, rate limiting, upload limits, logging controls, and deployment hardening.

Pelak-Khan stores recognition history locally. Operators are responsible for protecting local databases, captured media, backups, and any personal or vehicle-related data processed by their deployment.
