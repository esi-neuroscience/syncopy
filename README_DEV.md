# Syncopy Developer Information

These development instructions are only relevant for people on the Syncopy core team.

## Making a new Release

### PyPI

At github:

* Keep latest version/changes to be release on the `dev` branch for now
* Set the new the package version in `pyproject.toml` and make sure release notes in the `CHANGELOG.md` file are up-to-date.
* After last commit on github, log into the local ESI GitLab installation from within ESI and wait for the sync from Github to happen. The CI pipeline should start all runnners:
  - stage 1: single machine for all architectures like intelllinux, intelwin, macos
  - stage 2: slurm / HPC runners
* Check + fix all failures. Note that the pipeline on the interal Gitlab differs from the CI run on Github in several ways:
  - parallel tests are run
  - platforms other than linux x64 are used
  - the ESI filesystem/cluster is available, so tests that require local test data from cluster or the cluster itself are run
* Once tests are all green, in the gitlab "pipeline" tab, of the completed pipeline, there is a manual stage 3 'upload' entry named 'pypitest'. Click it to run the pypitest test deployment.
* If it succeeded: in the gitlab "pipeline" tab, of the completed pipeline, there is a manual stage 4 'deploy' entry named 'pypideploy'. Click it to run the final deployment to pypi.

This concludes the release to PyPI.

### Conda

Note: You need to release to PyPI first to start the conda release.

* Go to github.com/esi-neuoscience/esi-syncopy-feedstock and there in `recipe/meta.yaml`, check:
   - Update the version of the Syncopy package
   - Check versions of packages in `meta.yml` here versus versions in `pyproject.toml`/`syncopy.yml` in the root of the Syncopy GitHub repo (they need not be 100% identical, but having to old versions in there may lead to security risks or unexpected behaviour with older/buggy package versions).
   - Fill out the `PR` check list
   - If the conditions listed in section [When to Rerender](https://conda-forge.org/docs/maintainer/updating_pkgs.html#when-to-rerender) in the conda documentation apply to the current change/release: request `@conda-forge-admin please rerender` via comment on GitHub. Important: this may require you to fork the feedstock repo to your private github account first and do it from there, because of github permission errors with organizations.