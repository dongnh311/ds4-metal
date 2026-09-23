# Deploying ds4 to the local AI-Gateway

The AI-Gateway serves the Qwen3.8-Flash-Next model through `ds4-server`,
launched on demand from the PROD checkout at
`~/.local/share/ai-gateway/ds4-metal` with the command stored in the gateway
registry (`~/.local/ai-gateway/runtime-registry.json`, runtime `ds4`, port
18086). This document is the only supported way to change what runs there.
`deploy-ai-gateway.sh` at the repository root implements the mechanical steps
and refuses to run when a rule below is broken.

## Branches

| Branch | Role |
| --- | --- |
| `main` | Upstream base (ivanfioravanti/ds4-metal). Not deployed. |
| `develop` | Integration branch. All work lands here and must build and pass the smoke test before a deploy. |
| `prod/<feature>-YYYYMMDD` | One per deploy, cut from `develop` on the deploy date. What PROD runs is exactly the tip of one of these branches. |

Rules:

1. PROD is built only from a `prod/<feature>-YYYYMMDD` branch that exists on
   `dongnh311/ds4-metal-qwen` and is contained in `develop`. Never build PROD
   from `develop`, a feature or experiment branch, or a worktree.
2. Never copy binaries or `metal/` sources from another checkout into the PROD
   checkout. Kernels are compiled at startup from the `metal/` directory next
   to the binary, so a binary and its `metal/` must come from the same commit.
3. Never edit files in the PROD checkout. A fix goes to `develop` first and
   ships as a new `prod/` branch.
4. `prod/` branches are never rewritten or force-pushed. They are the rollback
   points.
5. One model process at a time on this 64 GB machine: stop the gateway ds4
   slot before building, testing or smoke-running.

## Procedure

### 1. Get `develop` ready

- Merge the work into `develop` and push it.
- Build: `make -j ds4 ds4-server ds4-bench ds4-eval ds4-agent`.
- Record reference outputs from the `develop` build with the registry command
  (use the new registry command if this deploy changes it, see step 4):

  ```sh
  ./deploy-ai-gateway.sh smoke --bin . --out /tmp/ds4-ref-<feature>
  ```

  Greedy output depends on the commit, the model and the launch flags, so the
  reference must come from the same commit and command you are about to deploy.

### 2. Cut the prod branch

```sh
./deploy-ai-gateway.sh cut <feature>      # creates prod/<feature>-YYYYMMDD from the remote develop
```

The branch is created from `develop` as it is on GitHub, so unpushed local
commits never reach PROD. `<feature>` is short kebab-case naming what the deploy
brings (for example `scale3-unc31`, `moe-kernels`).

### 3. Install it in the PROD checkout

Stop the gateway ds4 slot (no `ds4-server` process may be running), then:

```sh
./deploy-ai-gateway.sh install prod/<feature>-YYYYMMDD
```

This refuses if the PROD checkout has local edits, if ds4 is running, or if the
branch is not on the remote or not contained in `develop`. It then checks the
branch out tracking the remote, does a clean build of the five binaries, and
appends `previous -> new` to `.deploy-history` in the PROD checkout.

### 4. Registry changes (only when the launch command or model changes)

- Back up first: `cp runtime-registry.json runtime-registry.json.bak-<prod-branch>`.
- Edit `process_command`, `model_path` and `quantization` together. Validate
  with `python3 -m json.tool runtime-registry.json >/dev/null`.
- When the model or the KV cache format changes, move the disk KV cache aside
  (`ds4-kv-cache` -> `ds4-kv-cache.pre-<prod-branch>`). A checkpoint header only
  records the architecture, so an old checkpoint would be restored into a
  different model.
- Model files live in `~/.local/share/ai-gateway/ds4-models`; model artifacts
  are published on Hugging Face (dongnhdev), never committed to git.

### 5. Smoke test before handing over

```sh
./deploy-ai-gateway.sh smoke --ref /tmp/ds4-ref-<feature>
```

This starts the exact registry command from the PROD checkout on a scratch
port (18297) with a scratch KV directory, sends a Vietnamese and a code prompt
at temperature 0, and compares both replies with the `develop` reference. The
deploy passes only if both replies are non-empty and identical to the
reference, and the tokens per second match the `develop` run. It never touches
port 18086 or the live KV cache.

### 6. Hand over

Start the gateway ds4 slot (or let the gateway launch it on the first request)
and check the first real request in the gateway log.

## Rollback

```sh
./deploy-ai-gateway.sh install <previous prod branch>   # from .deploy-history
```

Then restore the registry backup from step 4 and, if it was moved, the KV cache
directory. Rolling back is a normal install of an older `prod/` branch, so it
goes through the same checks and clean build.
