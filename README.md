# Enhancing Differential Testing With LLMs For Testing Deep Learning Libraries

This is the artifact of our submission, "Enhancing Differential Testing With LLMs For Testing Deep Learning Libraries".

## Repository Structure

```
./
├── codes  --  code of our tool
├── data  --  Associated experiment data
├── figures  -- Associated figures
├── pytorch_confirmed_bugs.csv  -- confirmed 22 pytorch bugs
├── README.md
├── requirements.txt  --  necessary environment for running the provided notebook
├── RQ1CounterpartSynthesisEvaluation.ipynb  --  notebook for analyzing experiment data of RQ1
├── RQ2ConstraintExtraction.ipynb  --  notebook for analyzing experiment data of RQ2
├── RQ3Coverage.ipynb  --  notebook for analyzing experiment data of RQ3
├── scripts  --  scripts for running counterpart synthesis/path constraint extraction/testing
├── tensorflow_confirmed_bugs.csv  -- confirmed 37 tensorflow bugs
├── toolbox.py  --  toolbox for loading/analyzing experiment data
└── utils  --  utility files
```

## Experiment Result Reproduction

### API selection

In total, we select 1,862 and 1,121 APIs from TensorFlow and PyTorch, respectively, as our experiment subjects.
- We release the signatures of our selected APIs in `./data/tf_filtered_apis.txt` and `./data/torch_filtered_apis.txt`.
- We provide packages that we exclude in `./data/tf_skip_pref.json` and `./data/torch_skip_pref.json`.


### RQ1

We provide all counterparts synthesized by DLLens in RQ1. These counterparts are stored under this directory: `./data/working_dir/rq1/dllens`.

In RQ1, we compare DLLens with TensorScope and we strictly followed TensorScope's methodology to collect counterparts.

We release our reproduction result of TensorScope in `./data/working_dir/rq1/tensorscope`.

You can reproduce the Table 5 and Figure 3, 4 by running the notebook `RQ1CounterpartSynthesisEvaluation.ipynb`.

### RQ2

We release all path constraints extraction in RQ2. These path constraints are stored in `./data/working_dir/rq2/with-icf`.

In RQ2, we compare with DocTer's constraints, these constraints are stored in `./data/working_dir/rq2/docter`.

We also conduct an ablation study to compare with path constraints extracted without using our proposed input constraint inference method. These path constraints are stored in `./data/working/rq2/non-icf`.

You can reproduce the Table 6, 7 and Figure 5 by running the notebook `RQ2ConstraintExtraction.ipynb`.

### RQ3

In RQ3, we compare DLLens with existing approaches on 200 randomly sampled APIs, the signatures of these APIs are stored in `./data/working_dir/rq3/tf_100.txt` and `./data/working_dir/rq3/torch_100.txt`.

We record the branch coverage among these 200 APIs collected by each tool under the directory `./data/working/rq3/coverage`.

We list unique bugs among these 200 APIs detected by each tool under the directory `./data/working/rq3/bugs`. (Note that false positives are manually filtered out)

You can reproduce the coverage result by running the notebook `RQ3Coverage.ipynb`.

### RQ4

In RQ4, we claimed that DLLens has detected **71** bugs in TensorFlow and PyTorch, with **59** confirmed, including **13** already known bugs and **46** new ones (i.e., bugs confirmed by developers as previously unknown bugs).

We list all confirmed bugs (including known bugs and new bugs) in `./tensorflow_confirmed_bugs.csv` and `./pytorch_confirmed_bugs.csv`

- **Note that some entries (\ie, rows) of our provided CSV discuss multiple bugs, thus the total number of entries in our CSV file are smaller than the total number of bugs detected.** Please refer to each entry's 'Bug Count' column for the total number of bugs detected.

- For each bug detected by DLLens, we first check if it is fixed in the latest/nightly version of DL libraries and if it is reported by other users.
If a bug is already fixed or already confirmed by developers in issues created by other users, we consider it as a known bug. Otherwise, we will create new issues to report bugs.

We use the following GitHub ids to report bugs: maybeLee, enor2017, dlibk, drewshark, and QuantumCoder4.


## Implementation

We synthesized the counterpart using the following command line:
```angular2html
python -u -m scripts.synthesize_counterpart
```

- Within `codes/counterpart/counterpart_agent.py`, we use the `get_sample_inputs_from_llm` function for LLM prompting, `mutate_sample_inputs` function for property mutation, and `find_counterpart_from_llm` function for iterative counterpart synthesis.
- The prompt used for LLM prompting and counterpart synthesis/feedback construction can be found in `codes/prompt_text/sample_input_prefix_1.py` and `codes/prompt_text/counterpart_collection_prefix.py`.

We extracted path constraints using the following command line:
```angular2html
python scripts/extract_constraint.py
```
- We manually craft nine rules to reduce nine tensor operations, which can be found in the line 145-153 of `./codes/constraints/constraint_parser.py`.
- We use the prompt in `codes/prompt_text/condition_solver_template.py` for input constraint inference during the path constraint extraction.

We generate test inputs using the following command line:
```angular2html
python scripts/gen_tests.py
```
