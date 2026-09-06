# Package Release Guide

This document covers the checks that happen before versioning or tagging a release. It does not add a mandatory automated CI gate.

## Release readiness checklist

Before bumping the version or creating a tag, complete these checks:

1. Finalize `CHANGELOG.md`
   - close the current `## [Unreleased]` section
   - move shipped items into a dated release entry
   - recreate a fresh `## [Unreleased]` section at the top
   - make sure Added, Changed, Fixed, and Known Limitations reflect what is actually shipping
2. Review model and capability coverage
   - confirm the supported model list and model-specific notes still match reality
   - check that any new capability or constraint is documented in the most relevant guide
3. Verify install and quick starts
   - rerun any install path you touched
   - smoke test the README quick starts and examples that changed
4. Check links and contradictions
   - verify links across `README.md`, `docs/README.md`, `docs/model-compatibility.md`, `docs/benchmarking.md`, `ARCHITECTURE.md`, and `CHANGELOG.md`
   - make sure the README, docs, and release notes do not conflict
5. Record breaking changes and known limitations
   - call out migrations, unsupported paths, and user-visible limits
   - keep the summary in `CHANGELOG.md` and the matching guide
6. Capture performance context when you mention performance
   - model / checkpoint
   - hardware
   - software
   - workload
   - baseline
   - limitations
7. Confirm the actual release artifacts
   - verify whether this release publishes a wheel, an sdist, or both
   - build and inspect the exact artifacts before tagging so the uploaded files are the ones you intend to publish

This checklist is manual. It does not introduce a required automated CI gate.

## Automated Release Process

Stable releases are automated through GitHub Actions workflows in `.github/workflows/`:

- `.github/workflows/publish.yml`: builds and publishes tagged stable releases (`v*`) to PyPI and creates a GitHub release.
- `.github/workflows/publish-test.yml`: publishes nightly pre-release builds from `main` to PyPI.
- `.github/workflows/build-test.yml`: build validation for pull requests.

### One-time PyPI setup (Trusted Publishing)

Publishing authenticates via PyPI Trusted Publishing (OIDC); there are no PyPI
username/password/token secrets in the repository. The `pypa/gh-action-pypi-publish`
action reads credentials only from its `user`/`password` inputs (not from
`TWINE_*` env vars) and, with none supplied plus `id-token: write`, uses OIDC.

Before the first successful publish, register a trusted publisher on PyPI once
per workflow, at `https://pypi.org/manage/project/moe-infinity/settings/publishing/`:

- Owner: `EfficientMoE`
- Repository name: `MoE-Infinity`
- Workflow name: `publish-test.yml` (nightly) — then add a second, identical
  entry with Workflow name `publish.yml` (stable)
- Environment: leave blank

Until these are registered, the publish step fails with
`invalid-publisher: ... no corresponding publisher`.

### Steps to Release a New Version
To release a new version, such as version 1.0.0, follow this order:

0. Finalize release notes
   - close the current `## [Unreleased]` section in `CHANGELOG.md`
   - move shipped notes into the dated release entry
   - recreate a fresh `## [Unreleased]` section at the top
1. Versioning is automatic (setuptools-scm):
   - Versions are derived from the git tag at build time (see `pyproject.toml`
     `[tool.setuptools_scm]`) and written to `moe_infinity/_version.py`. There
     are no version strings to edit in `setup.py` or `moe_infinity/__init__.py`.
   - Tag `vX.Y.Z` publishes stable `X.Y.Z`. Every later commit on `main`
     publishes as the next-patch pre-release `X.Y.(Z+1).devN`, so nightlies
     always sort above the last stable and below the next one, with no manual
     `NIGHTLY_BASE_VERSION` bump.
   - First release only: the repository has no tags yet, so until the first tag
     exists nightlies version as `0.1.devN`. Create the initial tag (for example
     `git tag v0.0.1`) on the release commit to anchor the `0.0.x` series;
     afterwards nightlies become `0.0.2.devN` automatically.
2. Review the release checklist above
   - confirm model and capability coverage
   - verify install and quick starts
   - check links and contradictions
   - record breaking changes, known limitations, and performance context
   - confirm the release artifact choice, wheel, sdist, or both
3. Commit Changes: Commit these changes with an appropriate commit message that summarizes the update, such as "Update version for 1.0.0 release".
4. Create and Push Tag: Tag the latest commit with the new version number and push the tag to the repository. Use the following commands to accomplish this:
    ```bash
    git tag v1.0.0
    git push origin v1.0.0
    ```

Upon a successful tag push, the release workflow will create a release draft, build artifacts, and publish the package to PyPI.


## Manual Package Building and Publishing

For developers who prefer to manually build and publish their package to PyPI, the following steps provide a detailed guide to execute this process effectively.

1. Start by cloning the repository and navigating to the root directory of the package:
    ```bash
    git clone https://github.com/EfficientMoE/MoE-Infinity.git
    cd MoE-Infinity
    ```
2. Install the required dependencies to build the package:
    ```bash
    pip install -r requirements.txt
    pip install build
    ```
3. Build the source distribution and wheel for the package using:
    ```bash
    python -m build
    ```
    This command generates the package files in the `dist/` directory.
4. Upload the built package to the PyPI repository using `twine`:
    ```bash
    twine upload dist/*
    ```
    Ensure that you have the necessary credentials configured for `twine` to authenticate to PyPI.

Before uploading, inspect the wheel and sdist contents and confirm they match the release tag and changelog entries.


To build the package wheel for multiple Python versions, you should execute the build process individually for each version by specifying the corresponding Python interpreter.
