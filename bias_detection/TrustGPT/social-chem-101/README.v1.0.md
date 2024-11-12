# Social-Chem-101 Dataset
## v1.0

## Quick Info

The **Social-Chem-101 Dataset** centers around _rules-of-thumb (RoTs)_ as a conceptual
unit for understanding social and moral norms.

- Project page: https://maxwellforbes.com/social-chemistry/
- Paper: https://arxiv.org/abs/2011.00620
- Code: https://github.com/mbforbes/social-chemistry-101

Citation:

```
@conference{forbes2020social,
    title = {Social Chemistry 101: Learning to Reason about Social and Moral Norms,
    author = {Maxwell Forbes and Jena D. Hwang and Vered Shwartz and Maarten Sap and Yejin Choi},
    year = {2020},
    date = {2020-11-16},
    booktitle = {Conference on Empirical Methods in Natural Language Processing (EMNLP)},
}
```

## Dataset Columns

The dataset is tab-separated with the following columns:

_Note: the following is a markdown table. It's best viewed with a markdown renderer._

column | type | description
--- | ---  | ---
`area` | str | {confessions, dearabby, rocstories, amitheasshole}
`m` | int | {1, 3, 5, 50} How many workers did the RoT Breakdown for this RoT. Roughly corresponds to the split, but not exactly. Usually you'll want to use `split` instead.
`split` | str | {train, dev, test, dev-extra, test-extra, analysis, none} Which split this RoT belongs to. Much more information on splits are given below.
`rot-agree` | int\|null | {0, 1, 2, 3, 4, ""} Worker answer to question "What portion of people probably agree that ${rot}?" If question is unanswered, this value is written as "" to indicate null. The buckets in order are {&lt; 1%, 5% -- 25%, 50%, 75% -- 90%, &gt; 99%}. See the Mturk UI for descriptions of these buckets.
`rot-categorization` | str | Worker labeled "\|" separated list of 0 -- 4 RoT categorizations. Choices: {morality-ethics, social-norms, advice, description}. For example, "social-norms\|description". See Mturk UI for full descriptions of these values.
`rot-moral-foundations` | str | Worker labeled "\|" separated list of 0 -- 5 moral foundation _axes_. Choices: {care-harm, fairness-cheating, loyalty-betrayal, authority-subversion, sanctity-degradation}. For example: "care-harm\|fairness-cheating".
`rot-char-targeting` | str\|null | {char-none, char-N, ""} where N is in 0 -- 5 (inclusive). Worker answer to the question, "Who is the RoT most likely targeting in the following situation?" Value key: "" means null and the question was not answered; char-none means the worker picked "no one listed;" char-N means that the worker picked character N, a 0-index into the `characters` column (above).
`rot-bad` | int | {0, 1}  Whether the worker labeled the RoT as "confusing, extremely vague, very low quality, or can't be split into action and judgment."
`rot-judgment` | str\|null | Worker-written string representing the judgment portion of the RoT. We intended to throw this away; it was used for priming. "" means null; question not answered. For example, "it's bad".
`action` | str\|null | The action (conjugated / tweaked substring of RoT), written by the worker. "" means null; question not answered. For example, "taking candy from a baby"
`action-agency` | str\|null | {agency, experience, ""} Worker answer to the question "Is the action ${action} something you do or control, or is it something you experience?" where ${action} is the action (previous column) that the worker wrote. "" means null; question not answered.
`action-moral-judgment` | int\|null | {-2, -1, 0, 1, 2, ""} Worker answer to the question which best matches the RoT's original judgment (${judgment}) of ${action}?" where both ${judgment} and ${action} are written by the worker (previous columns). "" means null; question not answered. The buckets in order are {very bad, bad, expected/OK, good, very good}. See the Mturk UI for descriptions of these buckets.
`action-agree` | int\|null | {0, 1, 2, 3, 4, ""} Worker answer to the question, "What portion of people probably agree that ${action} is ${judgment}?", where both ${action} and ${judgment} are written by workers (previous columns). "" means null; question not answered. The buckets in order are {&lt; 1%, 5% -- 25%, 50%, 75% -- 90%, &gt; 99%}. See the Mturk UI for descriptions of these buckets.
`action-legal` | str\|null | {legal, illegal, tolerated, ""} Worker answer to the question, "Where you live, how legal is the action ${action}?" where ${action} is the action written by a Worker (previous column). See Mturk UI for descriptions of these buckets. "" means null; question not answered.
`action-pressure` | int\|null | {-2, -1, 0, 1, 2, ""} Worker answer to question "How much cultural pressure do you (or those you know) feel about ${action}?" where ${action} was written by the worker (previous column). "" means null; question not answered. The buckets in order are: {strong pressure against, pressure against, discretionary, pressure for, strong pressure for}. See the Mturk UI for descriptions of these buckets.
`action-char-involved` | str\|null | {char-none, char-N, ""} where N is in 0 -- 5 (inclusive). Worker answer to the question, "In this situation, who is most likely to do the action ${action} or its opposite?" where ${action} was written by the worker (previous column). Value key: "" means null and the question was not answered; char-none means the worker picked "no one listed;" char-N means that the worker picked character N, a 0-index into the `characters` column (above).
`action-hypothetical` | str\|null | {explicit-no, probable-no, hypothetical, probable, explicit, ""}. Worker answer to question "Is that character explicitly doing the action ${action}? Or is it that the action might happen (maybe the RoT was advice)?" "" means null; the question was not answered. Null is provided if they pick "char-none" to the previous question (`action-char-involved`), because this question is then skipped. See the Mturk UI for descriptions of these buckets.
`situation` | str | Text of the situation
`situation-short-id` | str | Unique ID for the situation, shorter and more convenient
`rot` | str | The rule of thumb written by the worker
`rot-id` | str | ID of the rule of thumb. Includes worker ID of RoT author and which RoT it was (1 -- 5).
`rot-worker-id` | str  | The worker who _wrote this rule of thumb_.  (No relation to worker did this RoT breakdown, though it could be the same by coincidence.)
`breakdown-worker-id` | str | The worker who _did this RoT breakdown_. (No relation to worker who wrote this RoT, though it could be the same by coincidence.)
`n-characters` | int | 1 -- 10 (10 max I've seen; no upper limit). How many characters were identified in the story during the NER mturk task. Minimum is 1, because 1 is the "narrator" who we assume said/wrote the situation. Maximum 6 characters are displayed during this HIT and available for selection (including "narrator").
`characters` | str | "\|" separated list of characters that appeared. 1 -- 6 characters will be shown. For example, "narrator\|a family member"

More information about `split` and `m` columns:

- `split`
    - the train/dev/test splits all have 1 worker / RoT breakdown. These are for
      training models (generative &  discriminative).
    - there are additionally dev-extra and test-extra “splits” with 5 (additional)
      workers / RoT breakdown. These are synchronized so that dev-extra come from a
      subset of the dev set, same with test-extra  from test. If we want, this lets us
      do a more nuanced scoring for these subsets (e.g., 5 correct answers or majority
      voting).
    - the analysis split comes from the 50-worker per RoT breakdown annotations

- `m` (how many worker annotated an RoT) is pretty straightforward, with a few twists:
    - m = 1 is the vast majority (like 99%) of the train/dev/test “main” dataset
    - m = 3 is a super small subset of our data. I did 3 workers early on for a couple
      batches just to get some agreement numbers. For that subset, we pick 1 breakdown
      to go towards the main dataset (i.e., then partitioned into train/val/test along
      with the m=1 annotations), and we mark the other 2 with none as their split.
    - m = 5 RoTs (in {dev,test}-extra) were sampled fully at random across all RoTs from
      each domain, with the condition that the RoT wasn’t marked as “bad” (unclear)
      during the 1-worker annotation. (The dev/test sets were then expanded from the
      situations in these initial subsets.)
    - m = 50 RoTs are for worker (not dataset) analysis. As such, the RoTs they use are
      not uniformly sampled from our data. (There’s also no constraint all RoTs for a
      situation make it in.) Instead, we take RoTs from the m = 5 annotation, find ones
      that <= 1 people marked as “experience” (so they will likely get the full RoT
      breakdown), and then sort by “most controversial,” i.e. lowest agreement in the m
      = 5 annotation. We annotate a small number of these maximally controversial RoTs
      from each domain with 50 workers.

- Other small notes:
    - The train/dev/test splits are are sampled by situation. So, all RoTs for a
      situation go to the same split. Also, each domain (AITA, rocstories, etc.) is
      split independently 80/10/10, so the domain proportions are the same across
      splits.
    - The {dev,test}-extra splits are also sampled by situation (all RoTs for a
      situation go to the same split). However, they are the same size for each domain.
    - If you want to get the “main” dataset for training models, don’t select by m=1!
      Instead, select by split = train (or dev or test). This is because a small portion
      of the dataset has m=3  — but with 1 annotation making it into the data splits,
      and the other 2 being assigned the none split.

## Dataset Collection

We provide a bit of information here about the three main stages of the dataset
collection. We do this in case it may help you better understand the meaning of the
dataset fields, their values, and when they are null.

### 0. Situations

See the _Social Chemistry 101_ paper Appendix A.1 for more details on the situations
used as sources for writing rules-of-thumb.

### 1. Rule-of-Thumb Writing

- **Overview:** A worker writes 1 -- 5 RoTs per situation.

- **Guidelines:** See the _Social Chemistry 101_ paper Appendix A.3 for more details on
  the constraints and guidelines for writing rules-of-thumb.

- **Skipping:** A worker can pass on a situation. If they write 0 RoTs, the situation is
  then omitted from the dataset.

### 2. Character Identification

- **Overview:** Workers pick spans corresponding to characters in situations.

- **Guidelines:** See the _Social Chemistry 101_ paper Appendix A.2 for more details on
  the constraints and guidelines for character identification.

- **Skipping:** Workers are not allowed to skip any situations.

We collect three character annotations per situation, and use the largest set of
characters provided by any of the annotators individually. (We don't perform a union to
avoid near-duplicates.)

### 3. Rule-of-Thumb Breakdowns

- **Overview:** Workers label attributes of an RoT, rewrite the RoT as "judgment" and
  "action" text boxes, and then label attributes of the "action" that they wrote. We
  have 1 annotation per RoT for most RoTs, but for a subset of the data, we have 3, 5,
  or 50 annotations.

- **Guidelines:** See the _Social Chemistry 101_ paper Appendix A.4 for more details on
  all of the fields annotated in the RoT breakdowns.

- **Skipping:** There are three ways workers can skip parts of the task.

    1. If a worker thinks the RoT is bad, they still must fill out the RoT labels
       (though they may have put dummy values in), but they can check the "rot-bad" box
       to skip the rest of the HIT.

    2. If a worker checks "experience" on the "agency or experience?" question, they
       skip all of the "action" labels.

    3. If a worker checks "no one listed" for the "who is most likely to do the action
       or its opposite" question, the `action-hypothetical` ("Is that character
       explicitly doing the action...") question is removed.

## Loading the Dataset

Since we allow workers to skip or partially breakdown unclear RoTs, many RoT breakdown
fields can be null. [Pandas](https://pandas.pydata.org/) only supports writing null
values to file as "". It also only loads integer null values ("") as floats, converting
the whole column to floats. For this reason, you need to convert these data types upon
loading to get back integer (ordinal) values, with `pd.NA` for missing entries. You can
do this with the `convert_dtypes()` function. (Make sure you are on a recent version of
Pandas; `pd.NA` is a new addition.) For example:

```python
import pandas as pd
df = pd.read_csv(
    "social-chem-101.v1.0.tsv", sep="\t"
).convert_dtypes()
```

## Contact

Maxwell Forbes - mbforbes@cs.uw.edu
