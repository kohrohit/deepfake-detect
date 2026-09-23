# Dataset access: the academic-signatory route

**Created 2026-09-21.** The owner ruled research/internal-only (`docs/HANDOFF.md` §0) and then, once
§4 showed only DFDC was reachable alone, ruled **find an academic signatory**. This file is the
working packet for that: what is true about each agreement, what a prospective PI is actually being
asked to take on, and text ready to send. Every quoted clause below was read at source on
2026-09-21 — §4 records where.

---

## 1. The constraint that decides what these datasets are *for*

Read before drafting anything, because it changes the ask.

FF++ clause 1: *"Researcher shall use the Database only for non-commercial research and educational
purposes."* Celeb-DF: *"for non-commercial research purposes only"*, and separately — this is the
wider phrase — *"You and your affiliated institution must agree not to reproduce, duplicate, copy,
sell, trade, resell or **exploit any portion of the videos or derived data**."*

**A model trained on these is derived data.** So a detector whose weights were fitted on FF++ or
Celeb-DF cannot ship in a commercial product later, whatever `assets/manifest.yaml` says about the
weight file — the gate (`assert_all_assets_registered`) checks registration, and registration is a
claim someone has to make truthfully. §1's correction block already marks the three equivalent
weight entries `"research-only — VERIFY before any commercial release"`. This extends the same rule
to anything we fit ourselves on that data.

**Therefore: use the research datasets for evaluation and publication only. The shipping detector
must be trained on licence-clean data — self-generated swaps over real faces we may use.** That is
the same corpus §0 says is required anyway, since 7 positives cannot train anything. The academic
route buys a *benchmark*, and specifically the several generators that
leave-one-generator-out needs. It does not buy training data for a product.

Say this plainly to any PI who asks what the data will be used for, because it is both true and the
answer that makes the request easy to agree to.

---

## 2. What each agreement requires

| | Signatory needs | Address | Stated restriction |
|---|---|---|---|
| **FF++** | Lab/Department/Affiliation, PI's name, PI's email — all required form fields | no academic-address rule stated | non-commercial research and education only |
| **Celeb-DF** | Affiliation/Organization, city, country, typed signature | **"your ACADEMIC email address"** — stated, not preference | non-commercial research only; no exploiting videos *or derived data* |
| **DFDC** | AWS account + IAM user + account ID | none | non-commercial per secondary sources only — **read the licence at download** |

### The clause to settle before anyone signs

FF++ clause 6, verbatim:

> If Researcher is employed by a for-profit, commercial entity, Researcher's employer shall also be
> bound by these terms and conditions, and Researcher hereby represents that he or she is fully
> authorized to enter into this agreement on behalf of such employer.

And clause 4:

> Researcher may provide research associates and colleagues with access to the Database provided
> that they first agree to be bound by these terms and conditions.

**The intended structure is: the academic is the "Researcher" and signatory; this project's owner is
a colleague under clause 4, bound by the same terms.** That keeps the affiliation and PI fields
truthful, which is the whole point of the route.

**Open question, and it is not one to answer by guessing:** once bound under clause 4, does clause 6
reach a colleague's for-profit employer? The text attaches clause 6 to "Researcher", but a colleague
bound by "these terms and conditions" is bound by clause 6 among them. The safe reading is that it
does reach them. Given §1's correction that this product is not for ScoreMe, do not submit anything
that could bind ScoreMe without ScoreMe knowing. Resolve this with someone qualified, or structure
the work so the owner is not the one accessing the data — the benchmark runs where the data is.

---

## 3. Outreach note to a prospective PI

Short on purpose. Replace the bracketed parts.

> Subject: Named PI for a deepfake-detection benchmark (FaceForensics++ / Celeb-DF)
>
> Dear [Name],
>
> I'm building an open deepfake-detection evaluation harness — a calibrated evidence pipeline that
> abstains rather than guessing when the signal isn't there, with a leave-one-generator-out protocol
> and per-source bootstrap confidence intervals. It's at [repo link]; it runs, it's fully tested,
> and it currently and correctly returns "insufficient evidence" on everything, because no
> benchmark has been run against it.
>
> To run that benchmark I need FaceForensics++ and Celeb-DF, and both require an academic signatory
> — FF++ asks for a lab and a named PI, Celeb-DF sends the download link only to an academic
> address. I'm writing to ask whether you'd be willing to be that PI.
>
> What I'd be asking, stated exactly: that you submit the access requests as the named Researcher,
> and that I work under you as a colleague bound by the same terms (FF++ clause 4 provides for
> this). The data would be used for evaluation and publication only. No model trained on it would
> ever ship in a commercial product — both agreements forbid exploiting derived data, and the
> production detector is being trained on separately generated, licence-clean material for exactly
> that reason.
>
> What I think is in it for you: a reproducible benchmark harness and a co-authored write-up of
> whichever results come out, including the negative ones — the existing evidence is that public
> detectors do not transfer across generators, and that is worth documenting properly.
>
> Happy to send the design document and the current test report, or to talk.
>
> [Name]

---

## 4. Form answers, ready to paste

**FF++ "Research purpose/project description"** and **Celeb-DF "Purpose Description"** — one text,
truthful under both agreements:

> Evaluating cross-generator generalisation of deepfake detectors. We are building an open-source
> calibrated evidence pipeline that reports per-source bootstrap confidence intervals and abstains
> where evidence is insufficient, and we need a multi-generator benchmark to measure it under a
> leave-one-generator-out protocol. Use is non-commercial research and education only. No model
> trained on this data will be distributed or used commercially; the production detector is trained
> separately on licence-clean material.

Leave the affiliation, PI and signature fields to the signatory. Do not fill them speculatively.

---

## 4a. The owner action list, ranked. Added 2026-09-23, after the permutation null.

**Why this section exists.** Until 2026-09-23 the EULA route was upside: DF40's ungated
repackaging could still *refute* a detector, so the benchmark could wait. The permutation control
(`docs/HANDOFF.md` §0, "The self-blend pair transfers at chance") closed that: every number this
project has produced on that corpus is inside the null a model fitted on shuffled labels produces.
**There is now no measurement route on this machine at all**, so one of the items below is the
critical path for the whole project, and nothing in the code can substitute for it.

What is needed of a replacement corpus is specific, and worth checking before spending effort on
any candidate:

1. **Its real and fake halves must share an imaging chain.** DF40-repackaged fails this — its
   halves come from different upstream corpora, which is why colour means alone separate them at
   AUC 0.843.
2. **Its sources must not be one family.** DF40-repackaged fails this too: 999 of 1,601 fakes are
   one filename family, a Kish effective n of 2.4.
3. **Per-technique labels**, without which leave-one-generator-out cannot run and a swap-only
   subset cannot be cut.

| # | Action | Who can do it | Cost | What it unblocks |
|---|---|---|---|---|
| 1 | Fix Kaggle auth | owner, alone | 2 min | any Kaggle-hosted corpus |
| 2 | Submit the DF40 request form | owner, alone | 10 min | per-technique labels — criteria 3, 8, and a swap-only subset |
| 3 | Find an FF++ academic signatory | needs a person | weeks | the benchmark, and Celeb-DF with it |
| 4 | Build our own eval corpus | owner + consenting people | days | **everything, including training** |
| 5 | Free disk | owner, alone | minutes | precondition for 2 and 4 |

### 1. Kaggle auth is broken on this machine

`~/.kaggle/` holds a 38-byte `access_token` and no `kaggle.json`, and `import kaggle` fails with
`OSError: Could not find kaggle.json`. The SFHQ download worked around it. Fix it properly:
kaggle.com → Settings → API → **Create New Token**, save the downloaded `kaggle.json` to
`~/.kaggle/kaggle.json`, then `chmod 600 ~/.kaggle/kaggle.json`.

### 2. DF40's own form — the cheapest ask, and the one to send first

Verified at source 2026-09-23, from the authors' README:

- **Request form:** <https://docs.google.com/forms/d/1ESAWoWusOEGEEVnXCH_emv-wJqCYMhCbD6-85RMIoDk/edit>
  (linked from <https://github.com/YZY-stack/DF40> as "Download DF40")
- **Licence: CC BY-NC 4.0.** So this is an EVALUATION benchmark and never training data for
  anything that ships — the same rule §1 states for FF++ and Celeb-DF, and it applies to any
  weights fitted on it.
- **Read before assuming it is independent of item 3.** DF40's own table lists FF++ and Celeb-DF as
  the source data for 30+ of its 40 techniques. Expect the form to ask you to confirm access to
  those, in which case this item collapses into item 3 rather than bypassing it. Send it anyway —
  the answer costs one form and settles the question.

Paste §4's purpose text; it is truthful under CC BY-NC 4.0 as written.

### 3. FF++ / Celeb-DF — unchanged, still the long pole

§3's outreach note is drafted and has not been sent. Two things to settle first, both in §2:
whether clause 6 reaches a colleague's for-profit employer, and therefore that nothing submitted
can bind ScoreMe without ScoreMe knowing.

### 4. The corpus that needs nobody's permission

Record genuine capture sessions with consenting people, and generate swaps over those faces with a
swapper whose licence permits it. This is the only route that unblocks **training** as well as
evaluation — §0 has said since 2026-09-21 that 7 positives cannot train anything — and the shipping
detector requires it on every route, because §1 rules out ever training on the research datasets.
It is slower than a form and it is the only item here that cannot fail for reasons outside this
project.

### 5. Disk

`/` is at 87% — 59 GB free, with 23 GB of it already SFHQ. DFDC is ruled out on this alone
(~470 GB), and item 4 will want room for recordings.

---

## 5. What to do in parallel

The outreach has weeks of lead time and may fail. It blocks nothing that matters most: the swap
corpus over licence-clean real faces is required on every route — for training, since the 442
sessions hold 7 positives and are the evaluation set — so start it now and treat the academic route
as upside on the benchmark, not a dependency.
