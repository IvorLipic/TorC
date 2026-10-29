# Library audit notes

Notes from tracing TorC's autograd before building the site (state of the library at the time; several defects listed in
`implementation-contract.md` were fixed on TorC `main` afterwards). Kept for reference and for deciding what to teach as a
"gotcha"; not part of the deployed site. `scene-notes.md` records which library behaviour each scene demonstrates. `v0.1.0-findings.md` lists what the later lessons found
in TorC v0.1.0 (for example the `MSELoss` shape problem, fixed on `main` by PR #5 and PR #6; the site pins that fix, `38785db`).
