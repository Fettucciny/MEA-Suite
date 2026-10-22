# Moving MEA Pattern Creator to Claude Code: one-time setup (Windows)

Do steps 1–3 once on **Fett**. Do step 1 again on FettLaptop only if you want
Claude Code there too.

## 1. Install

You need Claude Code, plus Git for Windows for the version history in step 2.
Open **PowerShell** and run:

```
irm https://claude.ai/install.ps1 | iex
winget install Git.Git
```

Close PowerShell, open a new one, and check that both installed:

```
claude --version
git --version
```

(Claude Code is also in the Claude desktop app as the **Code** tab, if you'd
rather not use a terminal.)

## 2. Turn the folder into a git repository (version history)

```
cd "$env:USERPROFILE\Nextcloud2\MEA\MEA Analyzer\MEA Pattern Creator"
git init
git add .
git commit -m "Baseline: MEA Pattern Creator at Cowork handoff (2026-10-22)"
```

`.gitignore` already leaves out the camera SDK, per-machine settings, `.exe`
files and the large `.seq` files. `.gitattributes` keeps `.bat` files in CRLF.

> Caution: Nextcloud also syncs the hidden `.git` folder. Commit from one machine
> at a time, and let Nextcloud finish syncing before using git on the other
> machine.

## 3. Start Claude Code in the folder

```
cd "$env:USERPROFILE\Nextcloud2\MEA\MEA Analyzer\MEA Pattern Creator"
claude
```

Claude Code reads `CLAUDE.md` automatically. A good first message:

> Read CLAUDE.md, docs/HANDOFF.md and docs/POLYGON_AND_STIMULATION.md, run the
> tests, and summarise the open items.

Do **not** run `/init`. `CLAUDE.md` is already written; `/init` would only
suggest edits to it.
