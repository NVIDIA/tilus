# Tilus Contribution Guides

## Getting started

### 1. Install in editable mode

We recommend using [uv](https://docs.astral.sh/uv/) for fast dependency management:

```bash
uv pip install -e ".[dev]"
```

This installs tilus in editable mode along with all development dependencies (linters, test tools, etc.).

### 2. Run pre-commit checks

Before submitting a pull request, run pre-commit on all files to catch formatting and lint issues:

```bash
pre-commit run --all
```

### 3. Sign your commits

All commits must carry a DCO sign-off and a valid GPG signature. If you have unsigned commits on your branch, run:

```bash
python scripts/sign-commits.py --fix
```

This script finds all commits between your branch and `origin/main`, identifies any that are missing a `Signed-off-by` line, and rebases to add the sign-off automatically. It optimizes the rebase to only touch commits that actually need signing. Use `--check` (the default) to see a report without modifying anything.

See the [Signing your contribution](#signing-your-contribution) section below for details on how signing works.

## Signing your contribution
To help ensure the integrity and authenticity of contributions, all contributors are required to sign their commits. This is done by adding a 'Signed-off-by' line to each commit message, certifying compliance with the Developer Certificate of Origin (DCO).

### How to sign your commits

You can sign your commits using the `-s` or `--signoff` option with `git commit`:

```bash
git commit -s -m "Your commit message"
```

This will append a line like the following to your commit message:

    Signed-off-by: Your Name <your.email@example.com>

Make sure the name and email address match your Git configuration. You can check your current settings with:

```bash
git config user.name
git config user.email
```

If you need to update them:

```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

All commits must be signed to be accepted.

### Setting up GPG signing

In addition to DCO sign-off, commits must have a valid GPG signature. Here's how to set it up:

**1. Generate a GPG key** (if you don't have one):

```bash
gpg --full-generate-key
```

Choose RSA (4096 bits), set an expiration, and use the same email as your Git config.

**2. Find your key ID:**

```bash
gpg --list-secret-keys --keyid-format=long
```

Look for the line starting with `sec`, e.g., `sec rsa4096/ABCDEF1234567890`. The part after the `/` is your key ID.

**3. Back up your key pair** (store in a safe location):

```bash
gpg --armor --export ABCDEF1234567890 > my-gpg-public.asc
gpg --armor --export-secret-keys ABCDEF1234567890 > my-gpg-secret.asc
```

If you lose your secret key, you won't be able to sign commits and will need to generate a new one.

**4. Configure Git to use your key:**

```bash
git config --global user.signingkey ABCDEF1234567890
git config --global commit.gpgsign true
```

With `commit.gpgsign = true`, every commit will be GPG-signed automatically. Combined with the `-s` flag for DCO sign-off, a typical commit command is:

```bash
git commit -s -m "Your commit message"
```

**5. Add the key to GitHub:**

Export your public key and add it to your GitHub account under **Settings > SSH and GPG keys > New GPG key**:

```bash
gpg --armor --export ABCDEF1234567890
```

**Troubleshooting:** If you see `gpg failed to sign the data`, make sure your GPG agent is running and your terminal can prompt for the passphrase:

```bash
export GPG_TTY=$(tty)
```

Add this line to your `~/.bashrc` or `~/.zshrc` to make it permanent.

### Developer Certificate of Origin

```
Developer's Certificate of Origin 1.1

By making a contribution to this project, I certify that:

(a) The contribution was created in whole or in part by me and I
    have the right to submit it under the open source license
    indicated in the file; or

(b) The contribution is based upon previous work that, to the best
    of my knowledge, is covered under an appropriate open source
    license and I have the right under that license to submit that
    work with modifications, whether created in whole or in part
    by me, under the same open source license (unless I am
    permitted to submit under a different license), as indicated
    in the file; or

(c) The contribution was provided directly to me by some other
    person who certified (a), (b) or (c) and I have not modified
    it.

(d) I understand and agree that this project and the contribution
    are public and that a record of the contribution (including all
    personal information I submit with it, including my sign-off) is
    maintained indefinitely and may be redistributed consistent with
    this project or the open source license(s) involved.

For more information, see https://developercertificate.org/.
```
