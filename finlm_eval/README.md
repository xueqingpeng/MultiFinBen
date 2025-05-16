# finben-refactor

**finben-refactor** is a specialized adaptation of the [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) framework, tailored specifically for evaluating language models on financial tasks. This tool focuses on assessing models available through commercial APIs and Hugging Face (HF) models, providing a streamlined approach for financial domain evaluations.

## How to Evaluate

To evaluate a language model using finben-refactor, follow these steps:

1. **Installation**:
   Clone the repository and install the necessary dependencies:
   ```bash
   git clone https://github.com/theFinAI/finben-refactor.git
   cd finben-refactor
   pip install -e .
   ```

2. **Model Selection**:
   finben-refactor supports models accessible via commercial APIs and Hugging Face. Specify the model type and parameters using the `--model` and `--model_args` flags. For example, to evaluate a Hugging Face model:
   ```bash
   lm_eval --model hf --model_args pretrained=your-model-name --tasks your_task --device cuda:0 --batch_size 8 --hf_hub_log_args "hub_results_org=your_org,results_repo_name=your_repo,push_results_to_hub=True,public_repo=True"
   ```

3. **Task Selection**:
   Choose the financial task(s) you wish to evaluate. A list of supported tasks can be viewed with:
   ```bash
   lm_eval --tasks list
   ```

4. **Running Evaluation**:
   Execute the evaluation by specifying the model and task(s):
   ```bash
   lm_eval --model hf --model_args pretrained=your-model-name --tasks your_task --device cuda:0 --batch_size 8 --hf_hub_log_args "hub_results_org=your_org,results_repo_name=your_repo,push_results_to_hub=True,public_repo=True"
   ```

For detailed information on additional parameters and advanced configurations, refer to the [lm-evaluation-harness documentation](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/interface.md).

## How to Add New Tasks

Adding new financial tasks to finben-refactor involves creating a YAML configuration file that defines the task's parameters and behavior. This configuration allows the framework to understand how to process the dataset and evaluate the model's performance.

### Steps to Add a New Task

1. **Define the Task Configuration**:
   Create a YAML file (e.g., `your_task.yaml`) with the necessary fields. Below is an example configuration for a multiple-choice task:
   ```yaml
   dataset_name: default
   dataset_path: your-dataset-path
   doc_to_target: gold
   doc_to_text: '{{query}}'
   output_type: multiple_choice
   doc_to_choice: choices
   fewshot_split: train
   should_decontaminate: true
   doc_to_decontamination_query: "{{support}} {{question}}"
   metric_list:
     - metric: acc
       aggregation: mean
       higher_is_better: true
     - metric: acc_norm
       aggregation: mean
       higher_is_better: true
   metadata:
     version: '1.0'
   num_fewshot: 0
   task: YourTaskName
   test_split: test
   training_split: train
   ```

2. **Implement Both Logits and Generation Tasks**:
   Ensure that both multiple-choice (logits-based) and generative evaluation tasks are defined. An example for generation-based tasks:
   ```yaml
   dataset_name: default
   dataset_path: your-dataset-path
   output_type: generate_until
   doc_to_target: '{{answer}}'
   doc_to_text: '{{query}}'
   fewshot_split: train
   should_decontaminate: true
   doc_to_decontamination_query: "{{query}}"
   generation_kwargs:
     until:
       - "."
       - ","
     do_sample: false
     temperature: 0.0
     max_gen_toks: 50
   filter_list:
     - name: "score-first"
       filter:
         - function: "regex"
           regex_pattern: "(Φορολογία & Λογιστική|Επιχειρήσεις & Διοίκηση|Οικονομικά|Βιομηχανία|Τεχνολογία|Κυβέρνηση & Έλεγχοι)"
         - function: "take_first"
   metric_list:
     - metric: exact_match
       aggregation: mean
       higher_is_better: true
   metadata:
     version: '1.0'
   num_fewshot: 0
   task: YourGenTaskName
   test_split: test
   training_split: train
   ```

3. **Add the Task to the Framework**:
   Place the task configuration file in `lm_eval/tasks/finben/` and ensure the task name is registered within the evaluation harness.

4. **Test Your Task**:
   Run the evaluation pipeline to verify that your task is properly configured:
   ```bash
   lm_eval --model hf --model_args pretrained=your-model-name --tasks YourTaskName --device cuda:0 --batch_size 8 --hf_hub_log_args "hub_results_org=your_org,results_repo_name=your_repo,push_results_to_hub=True,public_repo=True"
   ```

## How to Report Results to the Leaderboard

To report evaluation results to the leaderboard, use the `aggregate.py` script included in this repository. This script processes evaluation results and updates the leaderboard.

### Steps to Report Results:

1. **Add a New Model Configuration**:
   Update the `MODEL_DICT` in `aggregate.py` with your model details:
   ```python
   MODEL_DICT["your-model-name"] = {
       "Architecture": "Transformer",
       "Hub License": "your-license",
       "#Params (B)": 7,
       "Available on the hub": True,
   }
   ```

2. **Add a New Task Mapping**:
   Update the `METRIC_DICT` in `aggregate.py` to define task-specific parameters:
   ```python
   METRIC_DICT["YourTask"] = {"task_name": "YourTaskName", "random_baseline": 0.2}
   ```

3. **Run the Aggregation Script**:
   Use `aggregate.py` to collect and process evaluation results:
   ```bash
   python aggregate.py
   ```

By following these steps, you can ensure your evaluation results are properly processed and reflected in the leaderboard.
