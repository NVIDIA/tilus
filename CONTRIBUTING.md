# Tilus Contribution Guides

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

### Developer Certificate of Origin

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
