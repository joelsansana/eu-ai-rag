# Golden Set Curation Guidelines

Checklist for manually vetting candidate questions in `vet_golden.py` before
accepting them into `lepanto.jsonl` or `demo.jsonl`.

## 1. Gold-chunk correctness (reject if this fails — not negotiable)

- **The chunk fully answers the question**, not just tangentially. If
  answering well actually requires a cross-referenced article (the AI Act
  frequently references other articles, e.g. "in accordance with Article
  72"), either pick a different candidate for that chunk or add the
  referenced chunk's hash to `all_acceptable_ids`.
- **The question isn't broader than the chunk.** "What are the obligations
  of the AI Act?" can't be answered by one Article 26 chunk — that's a
  corpus-level question, not a chunk-level one. Reject or narrow it.
- **`answerable: true` actually holds up.** The generator sometimes
  mislabels a stub chunk (a bare chapter heading, a cross-reference-only
  recital) as answerable. If the chunk doesn't substantively answer
  anything, reject rather than force it in.

## 2. Realism — does this sound like something a person would actually type?

- **Avoid parroting the chunk header verbatim.** A question lifted straight
  from the header text tests lexical overlap, not retrieval quality — the
  embedding model barely has to work. Prefer rephrasing into the
  question's own words.
- **Don't name the article number unless that's the point.** A real user
  rarely says "under Article 26" — they ask about the obligation itself.
  Explicit-citation questions are trivially easy for retrieval regardless
  of embedding quality; keep a few for calibration, but don't let them
  dominate.

## 3. Difficulty — make the label mean something

Recalibrate the generator's difficulty tag rather than trusting it blindly:

- **Easy** — close lexical overlap with the source text, maybe cites the
  article number.
- **Medium** — paraphrased, uses synonyms for legal terminology (e.g.
  "watchdog" instead of "market surveillance authority"), no explicit
  citation.
- **Hard** — requires inferring intent or combining a term from one domain
  with a concept from the regulation (e.g. "AI oversight committees" when
  the chunk says "human oversight") — tests real semantic retrieval, not
  keyword matching.

If everything accepted ends up "easy," Hit@5 will read artificially high
and won't reflect retrieval quality under realistic conditions. Keep a
meaningful share of medium/hard.

## 4. Diversity and coverage

- **Watch for one article dominating.** It's easy to over-sample a single
  well-structured article (e.g. Article 26) just because it's convenient
  to vet. Spread across articles, chapters, recitals, and annexes roughly
  in proportion to how they'd actually be queried.
- **Drop near-duplicate questions on the same chunk.** If multiple
  candidates for one chunk are essentially the same question reworded,
  keep the best one, reject the rest.
- **Don't let one regulation crowd out the other.** Keep rough balance
  between `ai_act` and `nis2` within each tier unless there's a deliberate
  reason not to.

## 5. Keep the tier distinction meaningful

- **`lepanto`** (customer-shaped) — phrased the way someone trying to
  comply would ask it: practical, situational ("What do I need to tell my
  employees before using this system at work?").
- **`demo`** (regulation-shaped) — phrased the way someone studying the
  regulation would ask it: structural, precise ("What does Article 26(7)
  require regarding worker notification?").

If a question could go in either bucket interchangeably, the tier split
isn't adding signal — push it toward whichever phrasing fits better, or
rephrase it to commit to one.

## 6. Quick reject list

Reject outright (no editing needed) if the question is:

- Yes/no when the real content is nuanced (hides partial-credit cases the
  generation eval would want to probe later)
- Vague enough that multiple unrelated chunks could plausibly "answer" it
- Grammatically broken after generation (use `e` to fix a quick typo;
  reject if the whole question is incoherent)
- Actually unanswerable from the corpus — not a golden-set reject, set it
  aside as a candidate for the hand-authored `unanswerable.jsonl` instead

## Rough mental default

When in doubt: **would answering this well require actually reading and
understanding the chunk, or could a decent keyword search already nail
it?** Keyword-nailable questions are easy/low-value — fine to include a
few, but don't let them fill the whole budget.
