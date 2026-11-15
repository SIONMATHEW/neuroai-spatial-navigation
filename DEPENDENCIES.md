# Research dependencies

Trial generation is provided by two separate repositories:

- Python wrapper: <https://github.com/SIONMATHEW/RNNTrialStructures.py>
- Julia implementation: <https://github.com/SIONMATHEW/RNNTrialStructures.jl>

The final environmental experiment used these revisions:

| Repository | Commit |
|---|---|
| `RNNTrialStructures.py` | `a9d4b3b6f38dec6abf5f67f4a5cc6e4c0faf2b65` |
| `RNNTrialStructures.jl` | `0fb8c74710d8e129b3f6918daeb2bf16759282c2` |

The Python wrapper's `juliapkg.json` pins the Julia revision. Keeping the
generators outside this repository avoids maintaining a private fork inside
the experiment code.

Historical clean objective logs identify the installed Julia package snapshot
as `0Vqa9`. This is a Julia package-directory identifier, not a portable Git
commit. Exact figure reproduction uses the saved result tables and
checkpoints. A public release that promises exact retraining should first map
that identifier to an archived Julia tree and record its checksum.
