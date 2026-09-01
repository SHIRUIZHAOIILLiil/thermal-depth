# Generic four-backbone figure layout

Use this version for the main paper figure. The earlier two-pass Lotus diagram
is an implementation-detail reference, not the common architecture.

## Caption condition

Use one prompt variable and one text-encoder path:

`prompt p in {correct caption c, empty string} -> frozen CLIP -> C(p)`

Each experimental arm supplies one prompt per sample. Do not draw `C(c)` and
`C(empty)` as simultaneous inputs to one prediction.

## Common backbone interface

Draw one abstract, route-specific trainable block:

`(z_I, C(p), route-specific state) -> f_theta^(b) -> prediction`

with

`b in {Lotus-G, Lotus-D, Marigold, E2E-FT}`.

The four routes are trained separately; the block does not mean shared
weights. `route-specific input adapter A_b` hides differences in noisy-target
input, timestep, prediction target, decoding, and loss resolution.

## Target normalisation is route-specific too

⚠️ The list above omitted this and the exported figure inherited the omission:
it draws one shared `P_D -> z_0^D` chain in the offline panel and puts all
route dependence later, at the target parameterisation `T_b`.

That is wrong. `P_D` applies the normalisation of whichever backbone is being
trained (`hypersim_dataset.py:118-136`): Lotus normalises truncated
**disparity**, Marigold and E2E-FT normalise truncated **depth**. The four
routes therefore do not share a target tensor at all.

Draw it as `P_D^(b) -> z_0^(b)`, with the same fill as the trainable /
route-specific blocks rather than the shared grey.

This is not cosmetic. Reading a depth-normalised prediction through the
disparity-space affine inverts near and far, which cost a full evaluation round
on 2026-08-31 (AbsRel 0.265 against a true 0.125) and which the method section
now devotes a paragraph to. The figure must not contradict it.

## Auxiliary Lotus branch

Do not place the Lotus reconstruction branch in the main horizontal flow.
Add one note below the training panel:

> Lotus-G/D additionally retain the original empty-text thermal-reconstruction
> branch; Marigold and E2E-FT use only the depth branch.

This keeps the main figure faithful to all four routes without hiding the
Lotus-only auxiliary objective.

## Inference

Use one selected-route path:

`thermal frame + one prompt p -> selected trained route b -> depth output`.

Do not split inference into separate G/D rows in the main figure. Route-level
details belong in the method text or an ablation table.
