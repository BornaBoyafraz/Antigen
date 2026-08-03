# Antigen — threat model

## Security objective

Antigen scores text before an application places that text into a model's
context. Its security objective is to identify content that attempts to steer
the model away from the legitimate user's intent or the application's trusted
instructions, and to expose the signals behind that decision for review.

The central boundary is not simply “user text versus system text.” It is the
point where **untrusted content can become trusted instruction**. A webpage,
email, tool result, source file, or serialized argument may be legitimate data
for the user's task while still containing attacker-authored language addressed
to the model. If the surrounding application inserts that language into the
same context as trusted instructions without preserving or enforcing the trust
distinction, the content can acquire authority it was never meant to have.

Antigen operates at that boundary as a text classifier. It does not create the
boundary, track provenance, or enforce downstream tool permissions; the caller
must do those things.

## Attack classes

### Direct prompt injection

The attacker supplies behavior-steering text through the normal user-input
channel: instruction overrides, role spoofing, requests to reveal hidden
instructions, or jailbreak-style persona changes. The content is overtly an
instruction to the model, even if it is framed as a game, test, or hypothetical.

### Indirect prompt injection

The attacker plants an instruction in content the application retrieves or
processes on a user's behalf. The user may have asked for a summary, search,
code review, or data lookup rather than the hostile action. The failure occurs
when attacker-controlled data crosses the trust boundary and is interpreted as
an instruction.

Direct and indirect injection can contain identical words. The difference is
provenance and delivery path, which Antigen does not observe. Its narrower task
is to identify instruction-like, behavior-steering content in either channel.

### Obfuscated prompt injection

The attacker preserves the meaning of an instruction while disrupting literal
surface matching. Covered examples include encoded spans, zero-width
characters, character substitutions, changed casing, inserted spacing, and
leetspeak. Obfuscation can be applied to either a direct or an indirect attack.

This category matters because clean-data accuracy alone assumes the attacker
cooperates with the detector. The measured evasion results below make the
current character-level robustness boundary explicit.

## Attack carriers

The detector is intended to score text from carriers such as:

- **Web pages and retrieved documents:** visible prose, hidden or low-salience
  content, page metadata, and browser-retrieved excerpts.
- **Emails and messages:** attacker-controlled bodies or quoted content passed
  to summarization, triage, or action-taking workflows.
- **Tool and function results:** search results, database records, file reads,
  external service responses, and other output returned as data to an agent.
- **Code comments:** directives embedded in `#`, `//`, or HTML comment syntax
  within a file a coding workflow reads.
- **JSON and tool arguments:** behavior-steering strings inside serialized
  values or named argument fields.
- **URL parameters:** directives carried in query values, including percent-
  encoded values that become readable after parsing.

The last three carrier classes—code comments, JSON/tool-argument strings, and
URL query parameters—are the newly added structured-context feature families.
They are not treated as suspicious merely because the carrier syntax is
present; the extracted content must also contain behavior-steering language,
and quoted discussion context is suppressed.

## In scope

- Binary classification of a text block as `benign` or `injection`, with an
  injection score and interpretable matched signals.
- Direct, indirect, jailbreak/roleplay, encoded, and character-obfuscated
  prompt-injection content represented in the checked-in datasets and feature
  families.
- Third-party text about to enter a model context, including the structured
  carrier families listed above.
- Threshold selection based on the application's tolerance for missed attacks
  and false alarms.
- Narrow multi-turn codeword smuggling: an earlier turn explicitly defines a
  trigger and a later short turn invokes it.
- Pre-screening and human-review prioritization as one layer in a defense-in-
  depth design.

## Out of scope

- A guarantee that text is safe, trustworthy, or aligned with the user's
  intent after it receives a benign label.
- Source authentication, provenance tracking, trust labeling, instruction/data
  separation, access control, sandboxing, or least-privilege tool policy.
- Sanitizing or rewriting hostile content, preventing a model from following
  it, or validating the safety of a downstream action.
- General malicious-content detection unrelated to prompt injection, including
  malware classification, phishing verdicts, factuality, toxicity, and policy
  compliance.
- General multi-turn manipulation, gradual trust escalation, or persona drift
  without an explicit codeword setup and invocation.
- Established performance on multilingual traffic, images, audio, or other
  non-text inputs.
- Comprehensive coverage of semantic paraphrase, novel encodings, long-context
  dilution, chained evasions, or an attacker iteratively probing the detector.
- A production security boundary or a claim that the checked-in synthetic
  benchmarks represent deployment traffic.

## Measured weaknesses

The standard evaluation identifies benign text that quotes or discusses attack
language as the most persistent false-positive class. `benign_hard` accuracy
was 0.692 on the held-out split and 0.750 on the separately authored
adversarial suite. The discussion-context feature and guarded score cap reduce
this failure mode without solving it.

The adversarial robustness evaluator measures **evasion retention**: among the
171 injection examples caught in clear text at threshold 0.5, the fraction
still caught after each character-level transform. The actual run reported:

| Evasion | Retained | Retention |
|---|---:|---:|
| Zero-width insertion | 171 | 100.0% |
| Case flipping | 171 | 100.0% |
| Intra-word spacing | 167 | 97.7% |
| Homoglyph substitution | 143 | 83.6% |
| Leetspeak | 138 | 80.7% |

On this run, zero-width insertion and case flipping caused no loss among the
baseline-caught attacks, and intra-word spacing caused a smaller loss.
Homoglyph substitution and leetspeak were weaker; leetspeak was the lowest-
retention transform at 80.7%. This is evidence about the exact deterministic
benchmark, not a universal ordering of attacker techniques.

The denominator is conditional: attacks missed before perturbation are absent.
The evaluator also applies each transform separately, does not measure false
positives, and does not model a probing attacker. The results therefore expose
specific evasion gaps without establishing an overall adversarial success rate.

## Security controls around the detector

A deployment should preserve provenance and trust labels as text moves through
the system, keep untrusted data structurally separate from authoritative
instructions, minimize the tools and data available to the model, validate
tool arguments before execution, and require confirmation or review for
consequential actions. Detector scores should be logged with enough context to
audit decisions, while sensitive source content and detailed explanations
should not be exposed more broadly than necessary.

The operating threshold should be selected against representative local
traffic. A blocking gateway, a review queue, and a low-stakes telemetry signal
have different error costs. The score is calibrated to the checked-in data and
should not be interpreted as a universal probability under a different traffic
distribution.

## Residual risk

An adaptive attacker can use novel language, confusable characters,
leetspeak, multilingual phrasing, semantic indirection, long benign padding, or
combined transforms to reduce detection. Legitimate security material can
produce the opposite failure by containing attack phrases in discussion. Even
when Antigen flags the text correctly, a surrounding application can still
fail if it ignores the result or grants excessive authority to the model.

Antigen is therefore one inspection layer at a trust boundary, not the trust
boundary itself.
