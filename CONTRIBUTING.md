## Contributing to Syncopy

We are very happy to accept [pull requests](https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/creating-a-pull-request), provided you are fine with publishing your work under the [license of this project](./LICENSE).

If your contribution is not a bug fix but a new feature that changes or adds lots of code, please get in touch by [opening an issue](https://github.com/esi-neuroscience/syncopy/issues) *before* starting the project so we can discuss it first, coordinate and avoid wasted efforts.

Development happens on the *dev* branch. Please note that we do not accept PRs against other branches.


### Contribution workflow -- Overview

If you want to contribute something, the general workflow is:

- Fork the repo to your GitHub account.
- Clone your copy of the repo to your local computer.
- Create a new branch off the `dev` branch for your changes.
- Add tests relevant to your changes and run tests locally.
- When happy, create a PR on GitHub and wait for CI to run tests. Make sure to request merging into `dev`.
- When CI is green, we will review your code and get back to you to prepare merging into `dev`.
- On the next Syncopy release, `dev` will be merged into our stable branch, and your changes will be part of Syncopy.


### Contribution workflow -- Detailed instructions

Here are detailed instructions for the contribution workflow steps listed above:

- Log into your Github account, visit the [Syncopy repo page](https://github.com/esi-neuroscience/syncopy), and click [fork](https://github.com/esi-neuroscience/syncopy/fork) to fork the Syncopy repository to your account.
- Checkout **your forked** repository to your computer. You will be on the master branch. Make sure to **switch the branch** to the *dev* branch. E.g.:

```shell
git clone https://github.com/your_user/syncopy
cd syncopy
git checkout dev
```
- Now install the development version of Syncopy that you just checkout out, so you can actually test your changes to the code. We highly recommend to install into a new conda virtual environment, so that you do not
break your system-wide stable installation of Syncopy.

```shell
conda env create --file syncopy.yml --name syncopy-dev   # The file syncopy.yml comes with the repo.
conda activate syncopy-dev
pip install -e .
```

This allows you to run the Syncopy unit tests locally, and to run and test your changes. E.g., run a single test file:

```shell
python -m pytest syncopy/tests/test_preproc.py
```

We recommend running all unit tests once now to be sure that everything works. This also ensures that if you get errors on the tests later after you changed some code, you can be sure that these errors are actually related to your code, as opposed to issues with your Syncopy installation. To run all tests:

```shell
python -m pytest
```

This should take roughly 5 minutes and will open some plot windows. Please be patient.


- Now you have a verified installation and you are ready to make changes. Create a new branch off *develop* and name it after your feature, e.g., `add_cool_new_feature` or `fix_issue_17`:

```shell
git checkout dev  # Just to be sure we are still on the correct branch. This is important.
git checkout -b fix_issue_17
```

- Make changes to the Syncopy code and commit them into your branch. Repeat as needed. Add some tests.
- Make sure the unit tests run locally on your machine:

```shell
python -m pytest
```


- When you are happy with your changes, push your branch to your forked repo on Github.

```shell
git push --set-upstream origin fix_issue_17    # If your branch is named 'fix_isssue_17'.
```


- Then create a pull request on the GitHub website by visiting your copy of the repo. Make sure to request to merge your branch into the *dev* branch of the official Syncopy repo (the default is `master`, which is not what you want). You can verify that the branch is correct by clicking on the `Files changed` tab of the PR. It should list exactly your changes. If this is not the case, edit the PR and change the base to `dev`.

