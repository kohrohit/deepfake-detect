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

## 5. What to do in parallel

The outreach has weeks of lead time and may fail. It blocks nothing that matters most: the swap
corpus over licence-clean real faces is required on every route — for training, since the 442
sessions hold 7 positives and are the evaluation set — so start it now and treat the academic route
as upside on the benchmark, not a dependency.
