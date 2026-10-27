# Security reporting

## Scope

Report reproducible security concerns against the current default branch, including the commit SHA and affected component. Historical experiments and old snapshots are retained for reference; this document does not promise security support for any release or response time.

## Report privately when a channel is available

Open the repository's [Security tab](https://github.com/dev-belly/alphaforge/security) and check whether **Report a vulnerability** is available. If it is available, use that private reporting channel. This policy does not claim that private vulnerability reporting is enabled.

If no private reporting option is available, do not publish exploit steps, credentials, confidential data or a working exploit in a public issue or pull request. You may open a general [contact request](https://github.com/dev-belly/alphaforge/issues/new) asking for a private reporting channel, without including sensitive technical details. Wait for an agreed private channel before sharing them.

A useful private report contains the affected commit and environment, minimal reproduction, impact and any proposed mitigation. Use synthetic or public fixtures wherever possible; redact tokens and identifying records. Do not test against third-party services or accounts without authorization.

## Project-specific data handling

Reproduce first with the bundled synthetic/sample data. Real-market experiments must retain provider and availability limitations; do not infer investable performance from the sample report. Remove API keys, private datasets, broker details and identifying records before sharing.

Include the provider, actual data source, configuration, date range, model, seed, fold coverage, execution convention and cost assumptions. A frozen input sample and redacted run manifest are more useful than a performance screenshot alone.

Security handling is best effort. Do not send passwords, access tokens, customer records or unredacted private datasets through any repository form.
