# PartoGuard Findings

## Clinical and operational problem

Partograph problems in LMIC settings are often implementation problems rather than lack of belief in the tool. Reported barriers include inadequate training, staff shortages, non-availability of forms, time burden, and incomplete or retrospective documentation.

Nigeria-specific evidence supports the premise that partograph use and completion are inconsistent:

- Niger Delta study: correct completion was reported at about one-third of reviewed charts in two tertiary facilities; barriers included non-availability, staff shortage, inadequate knowledge, and time burden.
- Ogun State peripheral units: only a small minority of providers routinely used the partograph, and knowledge was often fair/poor.
- Other Nigerian studies report gaps between general awareness and detailed component knowledge.

Supported replacement for overclaim:

- **Avoid:** “For every 30 minutes after Action Line crossed, risk of death goes up 5x.” This specific claim is currently unsupported.
- **Use instead:** prolonged/obstructed labour is associated with increased maternal risk, and lack of partograph use/completion is associated with worse labour monitoring and obstructed labour risk in several studies.

## Partograph rule context

Two related but different clinical frameworks matter.

### Modified WHO partograph still widely used

The modified WHO partograph starts active-phase plotting at **4 cm** cervical dilatation.

Classic line rules:

- Alert line: starts at 4 cm and rises at 1 cm/hour.
- Action line: parallel to the alert line, offset 4 hours to the right.
- Cervical dilatation is typically plotted with `X` marks.
- Fetal head descent is typically plotted with `O` marks.
- Fetal heart rate is recorded more frequently, often every 30 minutes.

For prototype logic on the modified partograph, detection can be implemented geometrically once the chart is registered to a template.

### WHO Labour Care Guide is current WHO direction

WHO’s newer Labour Care Guide (LCG) replaces fixed Alert/Action lines with time thresholds by centimetre of dilation and broader care documentation. WHO 2018 guidance also states that a fixed 1 cm/hour cervical dilation threshold alone is not recommended as a routine trigger for intervention.

Implication for PartoGuard:

- The prototype can support the modified WHO partograph because it remains common in facilities and training material.
- The docs should clearly say WHO’s current direction is the LCG and that future PartoGuard should support both the older modified partograph and LCG-style workflows.
- Alerts should be phrased as “review/escalate per protocol,” not “intervene because 1 cm/hour was not met.”

## Real image and data availability

No open, large dataset of real filled WHO partograph photographs has been identified. This is a major project risk and also a credible hackathon motivation.

Available substitutes:

- Public blank templates and official manuals.
- Public training examples with filled/illustrated partographs.
- Case-study data that can be plotted onto templates.
- Synthetic images generated from known curves and then degraded with rotation, blur, low light, stains, compression, and crumple effects.

Important limitation:

> Synthetic data is useful for bootstrapping and demo development, but it cannot justify clinical accuracy or safety claims. Any accuracy number must come from clinician-labeled real holdout images.

Corpus strategy:

- Use blank templates and educational filled examples as form/layout references.
- Use structured labour datasets only through their official access processes and terms.
- Generate synthetic filled forms from blank templates plus clinically plausible curves.
- Keep raw real filled partographs, journal figures, video frames, and screenshots in local review-only storage unless reuse rights and PHI review are complete.
- Do not build an open dataset from random real filled partographs found online.

## Visual complexity observed from public examples and audits

Filled partographs are visually structured but messy.

Common elements:

- Top fetal condition band: fetal heart rate dots/lines, liquor letters, moulding entries.
- Central labour progress graph: cervical dilation `X`, fetal descent `O`, pre-printed grid, Alert/Action lines where present.
- Contraction section: shaded boxes representing contraction count/duration.
- Maternal condition rows: pulse, blood pressure, temperature, drugs, urine.

Real-world failure modes to plan for:

- Skewed or cropped photos.
- Poor lighting and shadows.
- Photocopy degradation.
- Smudges, stains, folds, ink bleed.
- Missing Alert/Action lines on local forms.
- Missing descent `O` marks.
- Incomplete fetal heart rate rows.
- Retrospective entries plotted after delivery.
- Different local form variants.

## Product implication

The strongest design is not a generic “ask the model about the page” flow. The page is a known structured form. PartoGuard should exploit that structure:

- Register the image to a known template.
- Crop only the needed regions.
- Extract marks with confidence.
- Compute clinical triggers deterministically.
- Expose uncertainty instead of guessing.

## Training-video pattern: three canonical teaching cases

A downloaded YouTube tutorial, “HOW TO FILL PARTOGRAM - VERY EASY,” demonstrates a simple educational pattern that is useful for PartoGuard demos and synthetic data labels:

1. **Normal progress:** 4 cm at 8:00 a.m. and 10 cm at 12:00 p.m.; plotted line remains left of the Alert line.
2. **Alert-line crossing:** 4 cm at 8:00 a.m. and 6 cm at 12:00 p.m.; plotted line crosses/right of the Alert line, prompting reassessment and cause-finding rather than autonomous intervention.
3. **Action-line crossing:** 4 cm at 8:00 a.m., 6 cm at 12:00 p.m., and 8 cm at 4:00 p.m.; plotted line reaches/crosses the Action line, prompting immediate senior review/escalation.

PartoGuard can use this as a demo structure: normal → warning → urgent review, with the app showing extracted `X` marks and the deterministic line comparison behind each output.
