# Certifying Safety against Adversarial Prompts in LLMs

Large language models (LLMs) released for public use are often fine-tuned to "align" their behavior to ensure they do not produce harmful or objectionable content. When prompted to produce inappropriate content, a well-aligned LLM should decline the user's request.
However, such safety measures have been shown to be vulnerable to **adversarial prompts**, which are maliciously designed sequences of tokens aimed to make an LLM produce harmful content despite being well-aligned. Moreover, such prompts can be generated entirely automated, creating an endless supply of quick and easy attacks.

The goal of this project is to design a *provable* robustness procedure to defend against adversarial prompts. Given a non-adversarial prompt P, the objective is to guarantee that the LLM's behavior is safe for an adversarially appended prompt P + [adv] as long as the length of [adv] is bounded.
