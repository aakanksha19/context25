# Context25: Evidence and Grounding Context Identification for Scientific Claims
This repository hosts the training/dev datasets and evaluation scripts for the [2025 Workshop on Scholarly Document Processing](https://sdproc.org/2024/sharedtasks.html#context25) Shared Task: **Context25: Evidence and Grounding Context Identification for Scientific Claims**

Submissions for the shared task will be evaluated on the `eval.ai` platform at this challenge URL: `{TODO}` (with a challenge period of `{TODO}`).

## Background and Problem

People read and use scientific claims both within the scientific process (e.g., in literature reviews, problem formulation, making sense of conflicting data) and outside of science (e.g., evidence-informed deliberation). When doing so, it is critical to **contextualize** claims with key supporting empirical evidence (e.g., figures of key results) and methodological details (e.g., measures, sample).

However, retrieving this contextual information when encountering and using claims in the moment (often far removed from the source materials and data) is difficult and time-consuming. Can we train AI models to help with this?

To assist with development of such models, this dataset contains `{TODO}` examples of scientific claims actually in use in lab notes and discussions for synthesis and research planning, across domains of biology, computer science, and the social sciences. For all of these claims, the dataset includes “gold” annotations for figures/tables that ground the key results behind each claim (this is “Task 1” for the workshop: see below). For a subset of these claims, we have “gold” examples of textual summaries --- and associated example text snippets from the paper --- of the key methodological details that ground each claim (this is “Task 2” for the workshop: see below).

## Dataset and directory structure

The claims and papers from this task come from four separate datasets, each of which comes from a different set of research domains.
1. `akamatsulab`: Cell biology
2. `BIOL403`: Cell biology
3. `dg-social-media-polarization`: Social sciences (political science, economics, HCI)
4. `megacoglab`: Various (HCI, psychology, economics, CS, public health)

The directory structure is as follows:
```
task1-train-dev.json
task2-train-dev.json
full_texts-train.json
silver-data/
eval/
```

The main training/dev datasets are in `task1-train-dev.json` and `task2-train-dev.json`.

PDFs for Task 1 are available at this gated HuggingFace repository: `{TODO}`

The current set of full-text parses for each paper are in `fulltexts-train.json`. The json is structured as a dictionary, where each key is a citekey (e.g., `nomura2004human`) and with a string containing the whole full-text parse for that paper as its associated value.

Evaluation scripts for each task are in `eval/`

As an additional possibly useful resource, `silver-data` contains full text parses for 17,007 papers from 1-2 hop in-bound and out-bound citations of the focal papers. 

Test set claims will be named `task1-test.json` and `task2-test.json`, respectively, and will have the same organizational structure as the training data. Full-texts will be in `full_texts-test.json`

## Task 1: Evidence Identification

### Task description

Given a scientific claim and a subset of `{TODO}` pages from the relevant research paper PDF that contains the claim, extract the figures and tables in those pages and then predict a ranked shortlist of key figures or tables from those pages that provide supporting evidence for the claim.

Here is an example claim with a Figure as its key supporting evidence.

```
{
	"id": "akamatsulab-WJvOy9Exn",
	"claim": "The density of free barbed ends increased as a function of growth stress",
	"citekey": "li2022molecular",
	"dataset": "akamatsulab",
	"findings": [
		"FIG 1D"
	]
}
```

And an example claim with a Figure and Table as its key supporting evidence.

```
{
	"id": "dg-social-media-polarization-tQy4sEF_R",	
	"claim": "Perceived polarization increased as a function of time spent reading a tweet, but only for Republican users",
	"citekey": "banksPolarizedFeedsThreeExperiments2021",
	"dataset": "dg-social-media-polarization",
	"findings": [
		"FIG 6",
		"TAB 2"
	]
}
```

Each claim corresponds to a paper via the `citekey` field. The PDFs for the papers are available in the following gated HuggingFace repository: `{TODO}`.

Scoring will be done using NDCG at 5 and 10. More details in `eval1.py` in `eval/`. 

`{TODO: verify we are covering the possible variations}`
> [!NOTE] Fig/table label format
> Since the starting point for the models is a PDF, rather than a set of known Figure/Table labels, for a claim, models need to extract figures/tables from the PDFs and then rank them. These extracted figure/table labels must conform to a specific format to allow our evaluation scripts to appropriately score predictions: 
> - **Figure** labels should be of the form `FIG {optionaldesignation}{fignum}{optionalsublabel}`, e.g., `FIG 1A`, or `FIG S3D` (for a supplemental figure), or `FIG 6`
> - **Table** labels should be of the form `TAB {optionaldesignation}{tabnum}{obtionalsublabel}`, e.g., `TAB 1`, or `TAB 3A`

> [!NOTE] Compound and subfigure scoring
> Note that many figures are compound figures, with labeled subfigures (e.g., FIG 1A, FIG 1B): sometimes the relevant grounding figure is a subfigure, and sometimes it is the whole (parent) figure. Thus, for NDCG scoring, predicted images that are parent/sub-figures of a gold figure label receive a relevance score of 0.5. 

### Training and dev data description

There are currently `{TODO}` total scientific claims across the four datasets, in the following breakdown:

| Dataset                      | N   |
| ---------------------------- | --- |
| akamatsulab                  | `{TODO}` |
| BIOL403                      | `{TODO}`  |
| dg-social-media-polarization | `{TODO}`  |
| megacoglab                   | `{TODO}` |

The full training dataset of `{TODO}` claims are in `task1-train-dev.json`.

## Task 2: Grounding Context Identification

### Task description

Given a scientific claim and a relevant research paper, identify all grounding context from the paper discussing methodological details of the experiment that resulted in this claim. For the purposes of this task, grounding context consists of a) one or more textual summary descriptions of key methodological details, and b) one or more grounding snippets from the paper that relate to at least one textual summary description is restricted to quotes from the paper. Grounding context snippets are typically dispersed throughout the full-text, often far from where the supporting evidence is presented. 

For maximal coverage for this task, generate textual summaries (and search for grounding snippets) that cover the following key aspects of the empirical methods of the claim:
1. **What** observable measures/data were collected
2. **How** (with what methods, analyses, etc.) from
3. **Who**(m) (which participants, what dataset, what population, etc.)

Descriptions should be provide as much local/temporal contextual detail as possible to enable appropriate reasoning about comparability and generalizability of empirical results. For instance, in a psychology study of creativity, the description of the measure (What) should not just be `novelty`, but the specific operationalization/measure (e.g., `Likert-style ratings from 3 independent experts in the domain of the problem`).

_NOTE_: we will not be scoring the descriptions and snippets separately by context "category" (e.g. who/how/what): we provide them here to clarify the requirements of the task.

Here is an example claim with descriptions and grounding quotes for its key methodological context:

```
{
    "id": "megacoglab-W3sdOb60i",
    "claim": "US patents filed by inventors who were new to the patent's field tended to be more novel",
    "citekey": "artsParadiseNoveltyLoss2018a",
    "dataset": "megacoglab",
    "context": [
		{
			"description": "Logarithmic transformation of the number of first-time pairwise subclass combinations, as an index of novelty of patents",
			"snippets": [
				 "To assess patent novelty, we calculate new combinations (ln) as the logarithmic transformation of one plus the number of pairwise subclass combinations of a patent that appear for the first time in the US. patent database (Fleming et al. 2007, Jung and Jeongsik 2016). To do so, each pairwise combination of subclasses is compared with all pairwise combinations of all prior U.S. patents. (p. 5)"
			]
		},
		{
			"description": "absence of any overlapping class between all prior patents and the focal patent, as indicator of exploring new fields",
			"snippets": [
				 "For each inventor-patent observation, we retrieve the three-digit technology classes of all prior patents of the focal inventor and identify whether there is any overlap between the three-digit technology classes of the focal patent and the three-digit technology classes linked o all prior patents of the same inventor. We rely on all classes assigned to a patent rather than just the primary class. Exploring new fields is a binary indicator that equals one in the absence of any overlapping class between all prior patents and the focal patent. (p. 6)"
			]
		},
		{
			"description": "2,705,431 patent–inventor observations from US firms assigned between 1975–2002",
			"snippets": [
				"we select the full population of inventors with U.S. patents assigned to \ufb01rms for 1975\u20132002 (p. 3)",
				"we begin with the full population of inventors and collect all patents assigned to \ufb01rms but, by design, must restrict the sample to inventors who have at least two patents assigned to the same \ufb01rm. The advantage of this panel setup is that we can use inventor\u2013firm fixed effect models to control for unobserved heterogeneity among inventors and firms, which arguably have a strong effect on the novelty and value of creative output. This approach basically uses repeated patents of the same inventor within the same firm to identify whether the inventor creates more or less novel\u2014and more or less valuable\u2014patents when any subsequent patent is categorized in a new \ufb01eld. The sample includes 2,705,431 patent\u2013inventor observations assigned to 396,336 unique inventors and 46,880 unique firms, accounting for 473,419 unique inventor\u2013firm pairs. (p. 5)",
			]
		},
		{
			"description": "inventor–firm fixed effect models",
			"snippets": [
				"we can use inventor\u2013\ufb01rm \ufb01xed effect models to control for unobserved heterogeneity among inventors and \ufb01rms, which arguably have a strong effect on the novelty and value of creative output (p. 5)",
			]
		},
    ]
  },
```

In this example, the descriptions and snippets fall into the following aspects of empirical methods:

**What**: \
Logarithmic transformation of the number of first-time pairwise subclass combinations, as an index of novelty of patents:
> "To assess patent novelty, we calculate new combinations (ln) as the logarithmic transformation of one plus the number of pairwise subclass combinations of a patent that appear for the first time in the US. patent database (Fleming et al. 2007, Jung and Jeongsik 2016). To do so, each pairwise combination of subclasses is compared with all pairwise combinations of all prior U.S. patents. (p. 5)"

absence of any overlapping class between all prior patents and the focal patent, as indicator of exploring new fields:
> "For each inventor-patent observation, we retrieve the three-digit technology classes of all prior patents of the focal inventor and identify whether there is any overlap between the three-digit technology classes of the focal patent and the three-digit technology classes linked o all prior patents of the same inventor. We rely on all classes assigned to a patent rather than just the primary class. Exploring new fields is a binary indicator that equals one in the absence of any overlapping class between all prior patents and the focal patent. (p. 6)"

**Who**:\
2,705,431 patent–inventor observations from US firms assigned between 1975–2002
> "we select the full population of inventors with U.S. patents assigned to \ufb01rms for 1975\u20132002 (p. 3)"
> "we begin with the full population of inventors and collect all patents assigned to \ufb01rms but, by design, must restrict the sample to inventors who have at least two patents assigned to the same \ufb01rm. The advantage of this panel setup is that we can use inventor\u2013firm fixed effect models to control for unobserved heterogeneity among inventors and firms, which arguably have a strong effect on the novelty and value of creative output. This approach basically uses repeated patents of the same inventor within the same firm to identify whether the inventor creates more or less novel\u2014and more or less valuable\u2014patents when any subsequent patent is categorized in a new \ufb01eld. The sample includes 2,705,431 patent\u2013inventor observations assigned to 396,336 unique inventors and 46,880 unique firms, accounting for 473,419 unique inventor\u2013firm pairs. (p. 5)"

**How**:\
inventor–firm fixed effect models
> "we can use inventor\u2013\ufb01rm \ufb01xed effect models to control for unobserved heterogeneity among inventors and \ufb01rms, which arguably have a strong effect on the novelty and value of creative output (p. 5)"

Scoring will be done using ROUGE and BERT score similarity to the gold standard descriptions and quotes. See `eval2.py` in `eval/` for more details.

### Example test data

Task 2 is a "test-only" task. In liueu of training data, we are releasing a small (N=`{TODO}`) set of examples, which can be used to get an idea for the task, with the following breakdown across the `akamatsulab` and `megacoglab` datasets:

| Dataset                      | N   |
| ---------------------------- | --- |
| akamatsulab                  | `{TODO}` |
| megacoglab                   | `{TODO}` |

## Evaluation and Submission

You can see how we will evaluate submissions --- both in terms of scoring, and prediction file format and structure --- for Task 1 and 2 by running the appropriate eval script for your predictions. 

Submissions will be evaluated on the `eval.ai` platform at this challenge URL: `{TODO}`

### Task 1

Predictions for this task should be in a `.csv` file with two columns:
1. The claim id (e.g., `megacoglab-W3sdOb60i`)
2. The predicted figure/table ranking, which will be comma-separated string of figure/table names, from highest to lowest ranking

Example:
```
claimid,predictions
megacoglab-W3sdOb60i,"FIG 1, TAB 1"
```

> [!WARNING]
> The script expects a header row, so make sure your csv has a header row, otherwise the first row of your predictions will be skipped. The names in the header row do not matter, because but we don't use the header names to parse the predictions data.


To get scores for your predictions, inside the `eval/` subdirectory, run `task1_eval.py` as follows:

`{TODO: update}`
```
python task1_eval.py --pred_file <path/to/predictionfilename>.csv --gold_file ../task1-train-dev.json --parse_folder ../figures-tables
```

You can optionally add `--debug True` if you want to dump scores for individual predicions for debugging/analysis.

### Task 2

Predictions for this task should be in a `.json` file (similar in structure to the training-dev file) where each entry has the following fields:
1. `id`: maps to the string id of the claim
2. `context`: maps to a list of dictionaries, each of which has the following fields:
   1. `label`: maps to a text string summary of the methodological detail
   2. `snippets`: maps to a list of grounding snippets

Order is not important fro the list of dictionaries and list of grounding snippets.

Before running the eval script for task 2, you will need to first install required dependencies of`bert-score` and `rouge-score`.

`bert-score`: https://github.com/Tiiiger/bert_score

```
pip install bert-score
```

`rouge-score`:

```
pip install rouge-score
```

Then run the `task2_eval.py` script in the following format:

```
TODO: update
```
