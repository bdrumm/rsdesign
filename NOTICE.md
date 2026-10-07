# Third-party assets

- **Roboto** (fixtures/fonts/*.woff2 referenced by roboto.local.css) — Google, Apache License 2.0 / SIL OFL 1.1 (v3+).
- **Material Symbols Outlined** (fixtures/fonts, material-symbols.local.css; variable font in fixtures/icons) — Google, Apache License 2.0. The icon atlas (fixtures/icons/*.npz) is derived from it.
- **Google Sans** and **Google Sans Flex** (fixtures/fonts/gsans*.local.css) — Google, SIL Open Font License 1.1.
- **Google Sans Text** is served by Google Fonts but is not in its public catalogue; it is not distributed here. Font identification uses it only when installed locally.
- **@material/web** (fixtures/mwc/mwc.bundle.js, built from node_modules/@material/web) — Google LLC, Apache License 2.0. Used only to generate self-test corpora.
- **Material Design 3 token values** (dt/mapping/material3.py, fixtures/design_systems/material3.json) — derived from the public Material Web token files, Apache License 2.0.

Reference screenshots of public websites (`fixtures/screens/*.png`) are captured locally by
`dt/selftest/reference_screens.py` and are **not** committed; only the manifest of URLs is.
