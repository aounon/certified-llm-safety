# Certifying LLM Safety against Adversarial Prompting

Large language models (LLMs) released for public use are often fine-tuned to "align" their behavior to ensure they do not produce harmful or objectionable content. When prompted to produce inappropriate content, a well-aligned LLM should decline the user's request.
However, such safety measures have been shown to be vulnerable to **adversarial prompts**, which are maliciously designed sequences of tokens aimed to make an LLM produce harmful content despite being well-aligned.
Given a harmful prompt P which would be rejected by the LLM, it is possible to generate an adversarial prompt P + [adv] which can deceive the LLM into thinking that it is safe and make the LLM comply with the harmful prompt P.
Moreover, the generation of such prompts can be entirely automated, creating an endless supply of quick and easy attacks.

The goal of this project is to design a *certified* safety procedure to defend against adversarial prompts. Given a harmful prompt P, the objective is to guarantee that it is detected as harmful even if it is modified with an adversarial sequence [adv] as long as the length of [adv] is bounded.
We design a framework, erase-and-check, that evaluates a safety filter on all subsequences of an input prompt created by erasing tokens one by one.

The file `defenses.py` implements the safety filter and different versions of the erase-and-check procedure. The file `main.py` is the main evaluation script for the experiments in the paper. The `data` directory contains the safe and the harmful prompts used for the experiments.

To evaluate the performance of the safety filter on the harmful prompts, run:
```
python main.py --num_prompts 500 --eval_type harmful
```

To generate the results for the different attack modes, run the following scripts from the directory named `bash scripts`:

Adversarial Suffix: `jobs_suffix.sh`

Adversarial Insertion: `jobs_insert.sh`

Adversarial Infusion: `jobs_infuse.sh`

Smoothing-based Certificates: `jobs_smoothing.sh`

Each of these scripts will produce a JSON file in the `results` directory. Use the following plotting scripts from `plot scripts` to generate the plots in the paper:

For accuracy on safe prompts, run:
```
python plot\ scripts/plot_acc.py results/[result file].json
```

For plotting running time, run:
```
python plot\ scripts/plot_time.py results/[result file].json
```

For comparison plot for the smoothing-based certificates, run:
```
python plot\ scripts/plot_smoothing.py results/[result file].json
```

## Trained Safety Classifier
To reproduce the results for the DistilBERT safety classifier, first train the classifier using the script `bert_toxic_classifier.py` in the `safety_classifier` directory. Then, use the `main.py` script with the option `--use_classifier` to evaluate on safe and harmful prompts:

```
python main.py --num_prompts 120 --eval_type harmful --use_classifier --model_wt_path [path-to-model-weights] --harmful_prompts data/harmful_prompts_test.txt

python main.py --num_prompts 120 --max_erase 20 --use_classifier --model_wt_path [path-to-model-weights] --safe_prompts data/safe_prompts_test.txt
```

Note: Results in this section are only available for the suffix mode for now.

## Attacking the Safety Classifier
`gcg.py` implements the Greedy Coordinate Gradient attack in [Universal and Transferable Adversarial Attacks on Aligned Language Models](https://arxiv.org/abs/2307.15043) for the DistilBERT safety classifier. To generate attacks of a sequence length of 20, run:

```
python gcg.py --num_adv 20 --prompts_file data/harmful_prompts_test.txt
```

To evaluate the safety filter on the adversarial prompts, run:
```
python main.py --num_prompts 120 --eval_type harmful --use_classifier --model_wt_path [path-to-model-weights] --harmful_prompts [path-to-adversarial-prompts]
```

## Randomized erase-and-check
The radomized version of the erase-and-check procedure evaluates the safety filter on a randomly sampled subset of the erased subsequences. The fration of the subsequences sampled is controlled by the `--sampling_ratio` parameter.
To evaluate the empirical performance of this version, first generate adversarial prompts with sequence lengths 0, 2, 4, ..., 20. Then run the following command:

```
python main.py --num_prompts 120 --eval_type empirical --max_erase 20 --use_classifier --randomize --sampling_ratio 0.1
```

Note: The randomized erase-and-check does not have certified safety guarantees.

## Installation
Follow the instructions below to set up the environment for the experiments.

1. Install Anaconda:
    - Download .sh installer file from https://www.anaconda.com/products/distribution
    - Run: 
        ```
        bash Anaconda3-2023.03-Linux-x86_64.sh
        ```
2. Create Conda Environment with Python:
    ```
    conda create -n [env] python=3.10
    ```
3. Activate environment:
    ```
    conda activate [env]
    ```
4. Install PyTorch with CUDA from: https://pytorch.org/
	```
    conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
    ```
5. Install transformers from Huggingface:
    ```
    conda install -c huggingface transformers
    ```
6. Install accelerate:
    ```
    conda install -c conda-forge accelerate
    ```
7. Install `scikit-learn` (required for training safety classifiers):
    ```
    conda install -c anaconda scikit-learn
    ```