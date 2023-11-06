# Certifying Safety against Adversarial Prompts in LLMs

Large language models (LLMs) released for public use are often fine-tuned to "align" their behavior to ensure they do not produce harmful or objectionable content. When prompted to produce inappropriate content, a well-aligned LLM should decline the user's request.
However, such safety measures have been shown to be vulnerable to **adversarial prompts**, which are maliciously designed sequences of tokens aimed to make an LLM produce harmful content despite being well-aligned. Moreover, such prompts can be generated entirely automated, creating an endless supply of quick and easy attacks.

The goal of this project is to design a *provable* robustness procedure to defend against adversarial prompts. Given a non-adversarial prompt P, the objective is to guarantee that the LLM's behavior is safe for an adversarially appended prompt P + [adv] as long as the length of [adv] is bounded.

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

## Installation
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