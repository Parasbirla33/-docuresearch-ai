# Security Policy

## Supported Versions

This project doesn't cut versioned releases yet - only the latest commit on
`master` is supported. Security fixes land there.

## Reporting a Vulnerability

Please **do not** open a public GitHub Issue for security vulnerabilities.

Instead, report it privately:

- Email **parasbirla421@gmail.com** with details and, if possible, steps to
  reproduce, or
- Use GitHub's [private vulnerability reporting](https://github.com/Parasbirla33/-docuresearch-ai/security/advisories/new)
  if enabled for this repository.

You should get an acknowledgement within a few days. Once a fix is
available, it'll be released and you'll be credited in the fix (unless you'd
rather stay anonymous) before any public disclosure.

## Scope notes

- **Never commit real API keys.** `.env` is gitignored; only
  `.env.example` (with blank values) is tracked. If you accidentally commit
  a real key, rotate it immediately - assume it's compromised as soon as
  it's pushed.
- Outbound webpage fetches (`src/docuresearch/tools/webpage.py`) are already
  scheme-restricted and DNS-checked against private/loopback/link-local
  ranges to guard against SSRF; if you find a bypass, that's a valid report.
- This project calls third-party LLM/search APIs with content it fetches
  from the web - prompt-injection-style findings in that pipeline are in
  scope.
