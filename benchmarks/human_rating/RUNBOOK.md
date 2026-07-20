# Human Rating Runbook

Operational guide for executing the human-rating workflow safely.

## Before Rating

1. Run all tests: `python benchmarks/human_rating/tests/run_all.py`
2. Confirm the queue contains exactly 30 items
3. Confirm the answer key is in `private/` (not under `rater/`)
4. **Do not inspect the answer key** — this would unblind the study
5. Close any tools or previews that expose method identity
6. Use one consistent `--rater-id` throughout the session

## During Rating

- Rate items in queue order (the CLI enforces this)
- Use the documented 1–5 integration rubric (`h` for help)
- Use notes (`n`) only for exceptional observations
- Flag (`f`) malformed or ambiguous items
- Do not manually edit `ratings.jsonl`
- To stop: type `q`, or Ctrl+C, or Ctrl+D — all preserve progress
- Resume with the same queue path, ratings path, and rater ID

## After Rating

1. Confirm `ratings.jsonl` contains exactly 30 records
2. Run compilation privately (requires answer key access)
3. Preserve raw artifacts: queue, ratings, session, answer key
4. **Do not regenerate the run after rating begins**
5. Record the Git commit and protocol version used

## Recovery Procedures

| Situation | Resolution |
|-----------|------------|
| Interrupted session | Resume with same command — the CLI picks up at the next unrated item |
| "Another rating session" error | Check for stale `.lock` file; remove only if no other process is active |
| Malformed ratings file | Do not edit manually. If corruption is minor (e.g., truncated last line from crash), the CLI will report the exact line number. Consult the research team. |
| Queue fingerprint mismatch | The queue file was modified after the session started. Restore the original queue from version control or the generation command. |
| Accidental answer-key exposure | Document the exposure. Consider whether re-blinding with a new seed is necessary. |
| Need to correct a prior rating | The ratings file is append-only. Do not manually edit it. Use a fresh run with a new `run-id` if a systematic error occurred, or document individual corrections in the notes of a subsequent research report. |
| Accidental regeneration attempt | The `--overwrite` flag is required. Without it, existing artifacts are protected. If you accidentally overwrote, restore from Git. |
| Failed compilation | Check the error message. Common issues: incomplete ratings, wrong paths, fingerprint mismatch. Fix the root cause and retry. |
