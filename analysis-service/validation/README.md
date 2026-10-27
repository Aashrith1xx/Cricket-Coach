# Calibration harness

**Nothing in this system should be shown to a user as a score until it has been through
this.** The tier assignments in `metric_tiers.py` come from published RMSE figures measured
on *other* rigs — three synchronised iPads at 240 Hz, in a controlled setting. Our capture
is one phone in a net or on a boundary. Those numbers are the starting assumption, not a
measurement of this system.

## What it checks

1. **Agreement** — for each Tier A metric, how closely does our value track a coach's or a
   reference system's on the same clip? Reported as bias, limits of agreement, and RMSE.
2. **Are the stated intervals honest?** If ~95% of reference values fall inside our 95% CIs,
   the uncertainty model is calibrated. Materially fewer means we are overconfident, which
   is the dangerous direction — it turns noise into verdicts. Materially more means we are
   too vague to be useful.
3. **Event detection** — frame error per event, against hand-labelled frames. Expect the
   foot-contact events to be good (gait literature achieves ~24–25 ms) and `IMP`/`BR` to be
   noticeably worse, since both are proxied rather than observed.
4. **Failure rate** — what fraction of real clips the capture gate rejects, and why. The
   OpenCap cricket study lost 44% of trials. If our rejection rate is far *lower* than that,
   the gate is probably too permissive rather than our capture being better.
5. **Repeatability** — same player, same session, consecutive clips. Between-clip scatter
   sets the real minimal detectable change, which drives how many clips a session needs.

## What a pass looks like

- Tier A metrics: bias not significantly different from zero; RMSE no worse than the
  assumed SEM in `uncertainty.py`.
- CI coverage between roughly 90% and 98%.
- Foot-contact events within ±2 frames at 120 fps.
- If a Tier A metric fails, it moves to Tier B or C. **It does not get a wider error bar and
  stay where it is** — that would preserve the appearance of a graded metric while removing
  its meaning.

## Getting reference data

In rough order of practicality:

1. **Coach-labelled clips** — a coach marks events and rates form on the same clips. Cheap,
   and directly checks the thing users care about, but coach ratings have their own
   inter-rater variability, so use several coaches where possible.
2. **A second camera** at a known angle, for cross-checking view-dependent metrics.
3. **Marker-based mocap**, if a university lab is accessible. Gold standard, and the only
   way to genuinely validate absolute angle accuracy.

A modest set is enough to be useful: roughly 30–50 clips across several players, form types
and capture conditions. Deliberately include clips that *should* fail the gate — a harness
that only sees good footage will not catch a gate that is too permissive.

## Running it

```bash
python -m validation.harness --clips path/to/labelled --out report.json
```

## Interpreting the report against `docs/ACCURACY.md`

If measured error comes in materially worse than the assumed SEMs — which is the likely
outcome, since every published figure came from better equipment — update the constants in
`uncertainty.py` rather than adjusting the tiers by feel. The tiers then follow from the
measurements, which is the whole point.
