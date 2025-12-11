
# **Simulating Clinical Trial Counterfactuals: A Conditional GAN Approach to Domain Translation of Tabular Data**

## Project Motivation and Summary

This repository presents a methodological framework for addressing a fundamental challenge in pharmaceutical development: the regional heterogeneity of clinical trial data. The primary contribution is a **Conditional CycleGAN** designed for unpaired tabular domain translation. This model learns to translate the statistical distributions of clinical trial outcomes between geographies—specifically the US and EU—without requiring paired data. The objective is to enable more robust data synthesis, augment sparse datasets, and generate high-fidelity counterfactual simulations to support quantitative due diligence and strategic decision-making in drug development.

The work further investigates the application of a **Mixed-Type Cycle-Consistent Diffusion model** to the same problem, providing a comparative analysis and highlighting key areas for future research in generative modeling for complex tabular data.

-----

## The Challenge: Regional Heterogeneity in Clinical Data

The interpretation of clinical trial results is often confounded by systematic, region-specific variations in patient populations and clinical practices. These differences can significantly impact trial outcomes, posing a substantial risk to multinational drug development programs. A key example, which this project models directly, is the well-documented placebo response differential in schizophrenia trials. The mean placebo effect in the U.S. can be nearly four times greater than in the EU (e.g., a -15.0 point vs. -4.0 point change on the PANSS scale). This discrepancy can obscure a drug's true treatment effect, reduce statistical power, and lead to erroneous conclusions about efficacy, thereby jeopardizing regulatory and commercial success.

Harmonizing such disparate datasets is a non-trivial task, particularly when patient-level data is unpaired across domains. This project addresses the need for a quantitative method to bridge this gap.

## A Methodological Solution: The Conditional CycleGAN

To address this challenge, I developed a Conditional CycleGAN architecture specifically for unpaired tabular domain translation. The model learns bidirectional mappings ($G\_{EU \\to US}$ and $G\_{US \\to EU}$) constrained by a cycle-consistency objective. The methodological innovations that enable it to handle complex clinical data include:

1.  **Explicit Conditioning on Covariates:** The generator and discriminator networks are conditioned on critical trial parameters, including drug type, dosage, and study protocol information. Categorical variables are handled via learned embeddings, while continuous covariates are injected directly. This ensures the translation is context-aware and specific to the trial's parameters.

2.  **Stochastic Translation for One-to-Many Mappings:** Recognizing that identical patient profiles can yield a range of outcomes, the model incorporates a latent noise vector ($z \\sim \\mathcal{N}(0, I)$) into the generator's input. This allows the model to produce a distribution of plausible translations for a single input record, capturing the inherent stochasticity of biological systems, a critical feature for realistic simulation.

3.  **Architectural Stability:** The generators employ a deep architecture with residual blocks to facilitate gradient flow and stable training. The adversarial objective utilizes the Least Squares GAN (LSGAN) loss function, which mitigates the vanishing gradient problem and improves convergence properties over the traditional sigmoid cross-entropy loss.

## Applications and Strategic Implications

This framework has direct applications for sophisticated stakeholders in the life sciences ecosystem. The ability to generate robust counterfactuals provides a new layer of analytical depth.

  * **For Biotech Venture Capital and Life Sciences Hedge Funds:** The methodology serves as a tool for quantitative due diligence. By translating data from early, ex-U.S. trials, one can generate a simulated U.S. trial dataset. This allows for a more rigorous forecast of a drug's probability of success in the larger U.S. market, thereby informing valuation models (e.g., rNPV) and identifying risks that are not apparent from the source data alone.

  * **For Corporate Strategy in Pharmaceuticals:** This approach enables the robust pooling and harmonization of data from multinational studies, leading to more statistically powerful analyses that can inform go/no-go decisions and capital allocation. It can also be used in silico to optimize future trial designs by modeling the impact of different regional enrollment strategies on the overall study outcome.

  * **For Statistics and Biostatistics Academia:** This project contributes a novel architecture for a challenging unpaired, tabular translation task. The results, including the partial success of the GAN and the documented failure of the more complex diffusion model, provide a rigorous case study on the practical limitations and opportunities in applying state-of-the-art generative models to heterogeneous, real-world data structures.

-----

## Conditional CycleGAN

Further implementation details (full code, logs, and ablation studies) are provided in the accompanying notebook and in the PDF write-up.  
This section summarises the formal specification and empirical validation of the model.

### Core Methodology

The network is a **Conditional CycleGAN** that learns two inverse mappings between the outcome distributions of EU and US clinical trials:

* $G_{EU\rightarrow US}$ – translate EU outcomes into their US counterfactual.
* $G_{US\rightarrow EU}$ – translate US outcomes into their EU counterfactual.

Each *generator* is a 6-block residual MLP; each *discriminator* is a 4-layer MLP trained with the Least-Squares GAN (LSGAN) objective.  
Five sources of information are concatenated at the input of every generator:

1. $\mathbf{x}$ – 8-dimensional vector of outcome deltas (​$\Delta$PANSS, weight gain, etc.).
2. $\mathbf{c}$ – 7 continuous covariates (age, baseline severity, dose, …) standard-scaled.
3. $s$ – study identifier (learned embedding, 16 d).
4. $d$ – drug identifier / arm (learned embedding, 8 d).
5. $\mathbf{z}\sim\mathcal N(\mathbf 0, I_{16})$ – latent noise enabling one-to-many mappings.

The total generator loss is a weighted sum of three terms
The total generator loss is a weighted sum of three terms:

$$
\mathcal{L}_{G} = \mathcal{L}_{\text{adv}}(G) + \lambda_{\text{cyc}}\mathcal{L}_{\text{cyc}}(G,F) + \lambda_{\text{id}}\mathcal{L}_{\text{id}}(G,F), \quad \lambda_{\text{cyc}} = 10, \quad \lambda_{\text{id}} = 0.1
$$

* **Adversarial loss (LSGAN)**

$$
\mathcal{L}_{\text{adv}}(G) = \frac{1}{2} \mathbb{E}_{x\sim p_X, c\sim p_C, z\sim p_Z} \left[(D_Y(G(x,c,z), c)-1)^2\right]
$$

* **Cycle consistency (L1)**

$$
\mathcal{L}_{\text{cyc}}(G,F) = \mathbb{E}_{x,c}\left[\lVert F(G(x,c,z),c,\mathbf{0})-x\rVert_{1}\right] + \mathbb{E}_{y,c}\left[\lVert G(F(y,c,z),c,\mathbf{0})-y\rVert_{1}\right]
$$

* **Identity preservation (L1)**

$$
\mathcal{L}_{\text{id}}(G,F) = \mathbb{E}_{y,c}\left[\lVert G(y,c,\mathbf{0})-y\rVert_{1}\right] + \mathbb{E}_{x,c}\left[\lVert F(x,c,\mathbf{0})-x\rVert_{1}\right]
$$

The latent vector is set to **zero** in the cycle/identity passes to ensure that the inverse mapping is deterministic and centred.

Training hyper-parameters  
• Adam (lr = 2 × 10⁻⁴, β₁ = 0.5) • batch = 32 • 50 epochs • $\approx$ 40 k gradient steps.

### Key Assumptions & Justifications

1. **Feature invariance** – Baseline patient descriptors (age, baseline PANSS, etc.) are taken as domain-invariant; the model only translates *outcome deltas*.  
   *Justification:* Counterfactual reasoning asks, *what would have happened to this exact patient in a different trial environment?*

2. **L1 cycle loss** – Mean absolute error is used for cycle/identity terms.  
   *Justification:* Empirically produces sharper reconstructions in GANs and was sufficient to preserve the joint structure of this tabular data.

### Validation and Results

Performance was evaluated on EU → US translation using 1000 held-out EU records.

| Category | Metric (US domain) | Result | Interpretation |
|----------|--------------------|--------|----------------|
| **Structure** | Cycle MSE (EU → US → EU) | **0.0032** | Generators are near-inverse |
| | Correlation Δ‖∙‖<sub>Fro</sub> | **0.259** | Correlation structure preserved |
| **Global fit** | MMD (σ = 1) | **0.0010** | Multivariate distributions match closely |
| **Marginals** | Median KS *p*-value | **0.021** | Several endpoints still differ |
| **Detectability** | Domain classifier accuracy | **0.779** | Fake-US is partially recognisable |
| **Utility** | TSTR accuracy (Responder) | **0.83** | Synthetic data useful for downstream modelling |

Overall, the Conditional CycleGAN faithfully reproduces the joint outcome distribution while retaining a useful level of stochastic diversity.  
Remaining deficiencies at the univariate level (KS tests) highlight avenues for future work, e.g. feature-wise weighting or hybrid GAN–diffusion objectives.


### Key Assumptions & Justifications

1.  **Feature Invariance:** The model assumes that baseline patient characteristics (e.g., age at onset, baseline disease severity) are domain-invariant. The goal is to translate the *outcomes* conditional on these fixed baselines.

      * *Justification:* This is a necessary and logical assumption for simulating a counterfactual: what would have happened to *this patient* in a *different region's* trial environment.

2.  **L1 Norm for Cycle Loss:** The cycle-consistency loss uses the L1 norm (Mean Absolute Error), which treats all features with equal importance and does not explicitly model their covariance structure.

      * *Justification:* This is a standard and pragmatic choice in CycleGAN literature, known to produce less blurry results than an L2 norm. While it has limitations, the results demonstrate it was sufficient for preserving the primary structural relationships in the data.

### Validation and Results

The Conditional CycleGAN's performance was evaluated on its ability to translate EU data to the US context.

  * **Structural and Global Fidelity:**

      * The model achieved a low cycle reconstruction error (**MSE: 0.00319** for the EU→US→EU cycle), indicating the learned mappings are effective inverses.
      * Global distribution matching was strong, with a Maximum Mean Discrepancy (MMD) of **0.001**, suggesting the overall multivariate distribution was captured well.
      * The correlation structure was well-preserved, with a Frobenius norm difference of **0.259** between the real and generated correlation matrices.

  * **Limitations and Incomplete Translation:**

      * **Univariate Fidelity:** Kolmogorov-Smirnov (KS) tests on marginal distributions revealed statistically significant deviations ($p \< 0.05$) for several key clinical endpoints, indicating imperfect fidelity at the individual feature level.
      * **Domain Distinguishability:** A logistic regression classifier distinguished the generated "fake-US" data from real US data with **77.9% accuracy**, indicating that subtle artifacts of the translation remain.

-----

## Code & Usage

The repository contains a Jupyter notebook for data simulation, model implementation, and evaluation in TensorFlow.

