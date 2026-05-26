Submitted by:
Madhvan Bajaj (7072049)

Anish Chandrasekaran (7072812)

This repository contains the code that produced our best leaderboard result on the Stolen Model Detection task. The detector combines two complementary signal groups. The first computes weight-space, output-space, Centered Kernel Alignment, and Dataset-Inference-style loss-gap metrics on 5000 in-set and 5000 out-set CIFAR-100 training images. The second selects the 1024 CIFAR-100 test images with the highest predictive entropy under the target — boundary-proximal probes — and computes top-1 agreement, top-5 overlap, and the mean per-sample cosine similarity of full softmax probability vectors. The two groups are combined as `final = 0.70 · multi_signal_score + 0.30 · hard_score`.

The single command `python main_detector.py` reproduces the result.

Setup
```
git clone <https://github.com/madhvanb/TML_Assignment_2.git>
cd TML_Assignment_2

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Model weights are streamed from HuggingFace on first run and cached locally, no manual download needed. The target and all 360 suspect ResNet-18 checkpoints live at `SprintML/tml26_task2`.

Reproducing the leaderboard result
```
python main_detector.py
```
That's it. All hyperparameters are set as constants at the top of `main_detector.py` — edit them there if you want to change anything.

The script downloads the target and suspect models, computes the global similarity metrics (weight, output, CKA, Dataset-Inference-style loss gap), adds the hard-probe behavioral metrics on high-entropy CIFAR-100 test samples, saves raw per-suspect metrics to `outputs/all_metrics.csv`, and writes the final leaderboard file to `outputs/submission.csv`.

Runtime: roughly 45 minutes on a single NVIDIA A100 (Google Colab Pro). Cached HuggingFace weights live under `model_cache/`.


Submitting
```
# edit submission.py first -> set API_KEY and FILE_PATH = "outputs/submission.csv"
python submission.py
```

Files
- `main_detector.py` — Final pipeline combining multi-signal and hard-probe groups -> TPR@5%FPR
- `submission.py` - Leaderboard uploader
- `requirements.txt` - All the required libs
- Final report PDF - not included in this workspace
- `submission.csv` - Final submission file

References
1. Florian Tramèr, Fan Zhang, Ari Juels, Michael K. Reiter, and Thomas Ristenpart. *Stealing Machine Learning Models via Prediction APIs*. USENIX Security Symposium, 2016. https://arxiv.org/abs/1609.02943
2. Simon Kornblith, Mohammad Norouzi, Honglak Lee, and Geoffrey Hinton. *Similarity of Neural Network Representations Revisited*. International Conference on Machine Learning (ICML), 2019. https://arxiv.org/abs/1905.00414
3. Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. *Deep Residual Learning for Image Recognition*. IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016. https://arxiv.org/abs/1512.03385
4. Yuanchun Nguyen et al. *ModelDiff: Testing-Based DNN Similarity Comparison for Model Reuse Detection*. ACM SIGSOFT International Symposium on Software Testing and Analysis (ISSTA), 2022.
5. Adam Dziedzic, Mislav Dhawan, Nicolas Papernot, and Siddharth Garg. *Dataset Inference for Self-Supervised Models*. Advances in Neural Information Processing Systems (NeurIPS), 2022.
