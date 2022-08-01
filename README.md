# Generate + Fairness
A Survey on Bias and Fairness in Machine Learning(CSUR2021)<br>
A Review on Fairness in Machine Learning(CSUR2022)<br>

Mitigating Unwanted Biases with Adversarial Learning(AIES 2018)<br>
在含(X,Y,Z)标签的数据中训练，使用X训练生成Y，同时利用对抗训练对Z保持无偏，解决demographic parity问题<br><br>
FairGAN: Fairness-aware Generative Adversarial Networks(Big Data2018)<br>
GAN生成文本的(X,Y,Z)，同时增加判别器使得X,Y与Z无关，Z则是独立分布的随机生成，解决demographic parity问题<br><br>
Fair Generative Modeling via Weak Supervision(ICML2020)<br>
构造图像的参考数据集Dref，使用贝叶斯分类器计算数据权重，重加权生成无偏数据集D~Dref，解决属性的imbalance问题<br><br>
Constructing a Fair Classifier with the Generated Fair Data(AAAI2021)<br>
使用VAE-GAN，实际上是生成均匀的联合分布(Y,Z)<br><br>
Fair Attribute Classification through Latent Space De-biasing(CVPR2021)<br>
在GAN的基础上直接训练从噪声向量到二分类值的线性分类器，依靠线性条件解方程组计算其互补向量的闭式解，以此生成成对数据，解决demographic parity问题<br><br>
AI recognition of patient race in medical imaging: a modelling study(The Lancet Digital Health2022)<br>
深度学习模型可以在严重损坏的图像数据中识别出患者的种族等信息<br><br>

# Generate

## GAN
Continual Learning with Deep Generative Replay(NeurIPS2017)<br>
Low-shot Learning via Covariance-Preserving Adversarial Augmentation Networks(NeurIPS2018)
Robustness of conditional GANs to noisy labels(NeurIPS2018)<br>
Meta-transfer learning for few-shot learning*(CVPR2019)<br>
Learning to remember: A synaptic plasticity driven framework for continual learning(CVPR2019)<br>
Task-GAN: Improving Generative Adversarial Network for Image Reconstruction(Machine Learning for Medical Image Reconstruction2019)<br>

## VAE$AE
Delta-encoder: an effective sample synthesis method for few-shot object recognition(NeurIPS2018)<br>
EEC: Learning to encode and regenerate images for continual learning(CVPR2021)<br>

## Flow
Masked Autoregressive Flow for Density Estimation(NeurIPS2017)<br>
Featurized Density Ratio Estimation(UAI2021)<br>

## Other Generate
Training Data Generating Networks: Shape Reconstruction via Bi-level Optimization(ICLR2021)<br>
Non-generative Generalized Zero-shot Learning via Task-correlated Disentanglement and Controllable Samples Synthesis(CVPR2022)<br>



# Target Task

## Continual Learning
iCaRL: Incremental Classifier and Representation Learning(CVPR2017)<br>
Continual Learning with Deep Generative Replay(NeurIPS2017)<br>
Learning to remember: A synaptic plasticity driven framework for continual learning(CVPR2019)<br>
Mnemonics Training: Multi-Class Incremental Learning without Forgetting(CVPR2020)<br>
Semantic Drift Compensation for Class-Incremental Learning*(CVPR2020)<br>
SS-IL: Separated Softmax for Incremental Learning(CVPR2021)<br>
EEC: Learning to encode and regenerate images for continual learning(CVPR2021)<br>
Instance-Conditioned GAN(NeurIPS2021)<br>
DualNet: Continual Learning, Fast and Slow*(NeurIPS2021)<br>
Learning Fast, Learning Slow: A General Continual Learning Method based on Complementary Learning System(arxiv2022)<br>


## Meta Learning
Model-agnostic meta-learning for fast adaptation of deep networks(ICML2017)<br>
Meta-transfer learning for few-shot learning*(CVPR2019)<br>
Meta variance transfer: Learning to augment from the others(ICML2020)<br>
Training Data Generating Networks: Shape Reconstruction via Bi-level Optimization(ICLR2021)<br>
Curriculum-Based Meta-learning(ICMR2021)<br>
Meta Learning Low Rank Covariance Factors for Energy-Based Deterministic Uncertainty(ML2021)<br>

## Few Shot Learning
Matching networks for one shot learning(NeurIPS2016)<br>
Prototypical networks for few-shot learning(NeurIPS2017)<br>
Delta-encoder: an effective sample synthesis method for few-shot object recognition(NeurIPS2018)<br>
Low-shot Learning via Covariance-Preserving Adversarial Augmentation Networks(NeurIPS2018)<br>
Meta-transfer learning for few-shot learning*(CVPR2019)<br>
