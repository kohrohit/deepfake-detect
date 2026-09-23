# The dfd service

An always-on deepfake-detection service: it watches a folder, scores whatever
lands in it, records an immutable audit record per decision, and serves the
results over HTTP.

**Read this first.** The service runs. It does not work, and it says so on
every page it serves. No detector on this deployment has been measured above
chance on a corpus it did not train on.

**Sharpened 2026-09-23, and it is worse than "unmeasured".** `blend_seam`
scores 0.289 on DF40 and the Apache-2.0 ViT scores 0.521, but neither number
is evidence of anything: a model fitted on SHUFFLED labels scores 0.229–0.780
on that corpus, so every result ever reported against it is inside the
no-signal null. The corpus has since failed two further checks — its real and
fake halves separate on colour alone at 0.843, and 5.41% of its declared
subject pairs are the same person (26x the unrelated-face rate). It can
neither support nor refute a detector. There is currently **no corpus on this
machine that can measure one**, which makes an evaluation corpus the entire
critical path (`docs/EULA-ACCESS.md` §4a).

The evidence gate below is what turns that fact into behaviour rather than a
footnote: every verdict is `insufficient_evidence`, and will stay that way
until something is measured above the floor.

---

## Install

```bash
cd /path/to/deepfake
python3 -m pip install -e .        # once
./ops/fetch-assets.sh              # model weights — gitignored, so a fresh
                                   # checkout has none and everything abstains
./ops/install.sh                   # systemd user unit, starts immediately
```

`ops/fetch-assets.sh` downloads the YuNet face detector and the SFace identity
embedder from OpenCV Zoo and verifies both against pinned sha256 values. Run
`./ops/fetch-assets.sh --check` any time to confirm what is on disk is what was
registered; it exits non-zero on a missing or drifted file. Pinning is not
housekeeping — the asset gate checks that a PATH is registered, which is a claim
about a logical id and not about bytes, so a file swapped at the same path passes
it unchanged.

`ops/install.sh` writes `~/.config/systemd/user/dfd.service`, enables it, and
starts it. No root. To keep it running after logout and across reboots:

```bash
sudo loginctl enable-linger $USER
```

| | |
|---|---|
| dashboard | http://127.0.0.1:8077 |
| inbox | `~/.local/share/dfd/inbox` — drop files here |
| database | `~/.local/share/dfd/dfd.sqlite3` |
| taken files | `~/.local/share/dfd/work/<date>/` |
| logs | `journalctl --user -u dfd -f` |
| stop / start | `systemctl --user stop dfd` / `start dfd` |
| remove | `./ops/uninstall.sh` (keeps the database) |

Without systemd, the same thing in the foreground:

```bash
python3 -m dfd.service -v
```

## Use

Three ways in, one way out.

```bash
# 1. Drop a file in the watched folder.
cp suspect.jpg ~/.local/share/dfd/inbox/

# 2. POST it.
curl -X POST --data-binary @suspect.jpg \
     'http://127.0.0.1:8077/api/scan?filename=suspect.jpg'

# 3. Drag it onto the dashboard.
```

```bash
curl -s http://127.0.0.1:8077/api/submissions | python3 -m json.tool
curl -s http://127.0.0.1:8077/api/submissions/<id> | python3 -m json.tool
```

| endpoint | what it gives |
|---|---|
| `GET /` | dashboard |
| `GET /health` | status, queue depth, uptime |
| `GET /api/stats` | counts by status and verdict |
| `GET /api/submissions?limit=&status=` | newest first, without the records |
| `GET /api/submissions/{id}` | one submission **with** its audit record |
| `POST /api/scan?filename=` | queue raw bytes, returns `202` and an id |
| `GET /api/evidence` | the measured report card the gate reads |

## The evidence gate

`bench/evidence_card.json` records, per detector, the AUC it was **measured**
at and on what corpus. At startup the service reads it and keeps a
calibration curve only for detectors at or above `--auc-floor` (0.75 by
default). A detector below the floor still runs and its raw score still
reaches the audit record — the data is worth accumulating — but with no
curve it contributes `llr 0.0` with `uncalibrated_for_band` and cannot move a
verdict.

Three properties make this a gate rather than a setting:

- **Unmeasured fails closed.** `auc: null` never decides, exactly as an
  unregistered asset is treated as non-commercial in `assets/manifest.yaml`.
- **The card cannot lower the floor.** It may only raise it. A gate whose bar
  is set by the thing being gated is not a gate.
- **A missing or malformed card is an error, not an empty gate.** Both end
  with nothing deciding; only the error tells you which one happened.

`tests/service/test_evidence.py::test_no_detector_currently_clears_the_floor`
asserts the current state. The day a detector is measured above 0.75, that
test fails, and whoever raised it has to come and delete it deliberately —
which is the moment this service starts issuing real verdicts, made by a
person, on the record.

## What a verdict means today

```
$ curl -s .../api/submissions/<id> | python3 -m json.tool
"verdict": "insufficient_evidence",
"evidence": [
  {"detector": "blend_seam", "llr": 0.0, "raw_score": 0.803,
   "reason": "uncalibrated_for_band", "version": "0.2.0-fairface10k"},
  {"detector": "effnet_b4",  "llr": 0.0, "raw_score": null,
   "reason": "weights_absent",       "version": "0.1.0"},
  {"detector": "npr",        "llr": 0.0, "raw_score": null,
   "reason": "weights_absent",       "version": "0.1.0"}
]
```

`raw_score: 0.803` is what the seam model said. `llr: 0.0` is what the
service is willing to conclude from it, which is nothing, because that model
was measured at 0.289 on an unseen corpus. Both numbers are kept: the first
is evidence for a future decision about the detector, the second is the
decision about this file.

Other reasons you will see, and what each actually means:

| reason | meaning |
|---|---|
| `uncalibrated_for_band` | scored, but the detector is not trusted at this quality band (today: not trusted at all) |
| `below_quality_floor` | the crop was too small or too poor for this detector's floor — a 224px thumbnail bands as `reject` |
| `weights_absent` | no weight file on this deployment |
| `detector_error` | that detector raised; the others still decided |

A failed row is a refusal, not a verdict: unsupported extension, undecodable
file, or over a decode limit. The message is recorded verbatim.

## Calibration

`training/fit_calibration.py` fits the per-band curves that turn a raw score
into nats, and writes them as plain JSON:

```bash
python3 -m training.fit_calibration \
    --corpus ~/Desktop/agents/datasets/df40_eval/extracted/val/val \
    --out assets/models/calibration.json
```

The corpus is split by **source** before anything is fitted, the curves see
only the fit side, and every number in the report comes from the holdout
side — spec §8.2 guard 5, made structural rather than remembered.

**No calibration file ships with this repository, deliberately.** Every
detector is currently gated out, so a curve on disk would imply a readiness
that does not exist. Run the fitter when a detector clears the gate. Fitting
a curve is not a licence to decide: that is the gate's job, and it reads
measured cross-corpus AUC, not this file.

## Operations

- **Crash recovery.** A submission is `running` only while a worker holds it.
  Anything still `running` at startup is requeued, and the file is still in
  the workdir, so the work is repeatable.
- **Partial files.** A file must be the same size on two consecutive scans
  before it is taken. Half-copied files decode as truncated, and a refusal
  about a file that was fine is worse than a second of latency.
- **One bad file.** A decision that raises fails that submission and nothing
  else; the worker loop survives filesystem and database errors too.
- **Restart.** `Restart=always` with a 5s delay. A detection service that has
  stopped looks exactly like one that is up and scoring nothing.
- **Retention.** Scored files are deleted after `--retain-days` (30 by
  default; `0` keeps everything), hourly, from the worker loop. **Audit
  records are never deleted** — the file is the input, the record is the
  decision. A submission still queued or running is never pruned whatever its
  age. `GET /api/submissions/{id}` reports `file_retained`, so an old
  decision shows "the file is gone" rather than a path that silently does not
  resolve.

## Security

There is **no authentication**, deliberately and only because the service
binds `127.0.0.1`. Binding anything else is **refused** unless you pass
`--allow-remote-access`, which exists so that exposing the port is a decision
somebody made rather than four characters in a unit file nobody re-read. `POST /api/scan` writes caller-chosen bytes into the
workdir and spends CPU decoding them. Do not move it off the loopback
interface without an authenticating reverse proxy in front.

Uploads are refused from the `Content-Length` header, before the body is
read, at `Limits.max_file_bytes` (256 MB). Decode limits (pixels, frames,
duration) are enforced from the file header before allocation — see
`dfd/limits.py`.

## What the HTTP surface refuses, and why

Loopback-only and unauthenticated, so the limits below are about keeping one
caller from taking the service down rather than about untrusted traffic. All
four were added 2026-09-23 after reviewing the service as something that runs
unattended rather than as something that passes its tests.

| refusal | status | why |
|---|---|---|
| `?limit=abc` | 400 | it used to raise inside the handler, so the caller got a closed connection rather than an error |
| `?limit=-1` or `0` | 400 | SQLite reads `LIMIT -1` as NO limit, so a bounded endpoint returned the entire table |
| `Transfer-Encoding: chunked` | 411 | this handler reads exactly `Content-Length` bytes; a chunked body arrived as length 0 and was stored as an empty submission, then failed at decode — recording a refusal about the FILE when the fault was in the request |
| body shorter than `Content-Length` | 400 | the prefix would be stored as a truncated file |
| 33rd concurrent request | 503 | `ThreadingHTTPServer` spawns one thread per connection with no ceiling; the failure mode was the process rather than one refused request |

A connection that stalls is closed after 30s. Without that, a caller sending
`Content-Length: 268435456` and then nothing held a worker thread until the
process died.

## Runbook

| symptom | check | likely cause |
|---|---|---|
| every verdict `insufficient_evidence` | `GET /api/evidence` | expected — no detector is above the gate's floor. This is not a fault |
| every detector `weights_absent` | `./ops/fetch-assets.sh --check` | weight files missing or drifted |
| `faces=weights_absent` | same | the YuNet `.onnx` is absent; nothing can be cropped |
| queue depth climbing | `journalctl --user -u dfd -f` | a worker is stuck or the inbox is receiving faster than one CPU can score |
| 503s from the API | `GET /health` | concurrency ceiling reached; the service is up and shedding load |
| submissions stuck `running` after a restart | none needed | they are requeued at startup and the file is still in the workdir |
| database growing | `du -sh ~/.local/share/dfd` | audit records are never deleted, by design. Scored FILES are pruned after `--retain-days` |

**Back up `~/.local/share/dfd/dfd.sqlite3`.** It holds every audit record, and
those are the decisions; the files beside it are only the inputs and are pruned
on a schedule. `sqlite3 dfd.sqlite3 ".backup /path/to/backup.sqlite3"` is safe
against a running service (the database is in WAL mode).

## Dependencies

None beyond what the library already needs. The HTTP layer is
`http.server`, the queue is `sqlite3`, and the dashboard is one HTML string
with no CDN and no build step. This repo pins every dependency exactly and
runs its gates at both ends of every declared range; a web framework would
add that maintenance to a service whose entire API is six JSON endpoints and
one page.
