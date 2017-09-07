---
title: Variable Selection with I-priors
layout: single
permalink: /var-select/
---

<script type="text/x-mathjax-config">
MathJax.Hub.Config({
  TeX: { equationNumbers: { autoNumber: "AMS" } }
});
</script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.0/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

{% include toc %}

I-prior model selection can easily be done by comparing (marginal) likelihoods. 
We also explore a fully Bayes alternative to model selection for linear models. 
A simulation study gives positive results for the I-prior for linear models in situations with multicollinearity, beating other types of priors such as $$g$$-priors, independent priors (ridge regression), and also Laplace/double exponential priors (Lasso).

## Tests of Significance

For the regression model given in [(1)](/intro/#introduction) and the function $$f$$ of the form

$$
f(x_i) = f_1(x_{i1}) + \cdots + f_p(x_{ip})
$$

where each $$f_j$$ lies in a RKHS $$\mathcal F_{\lambda_j}$$, one may be interested in which of these functions contribute significantly in explaining the response. 
A valid way of doing this is by performing inference on the scale parameters $$\lambda_1, \dots, \lambda_p$$.

Where these parameters are estimated using maximum marginal likelihood, the distribution of the scale parameters are asymptotically normal. 
A simple test of significance can be done, and this would also correspond to the significance of the particular regression function since each $$f_j$$ is of the form

$$
f_j(x_{ij}) = \lambda_j \sum_{k=1}^n h(x_{ij}, x_{ik}) w_k,
$$

where the $$w_k$$ have a normal posterior distribution. 

## Model Comparison via Empirical Bayes Factors

Alternatively, model comparison can be done by comparing likelihoods. 
Again, as these models are fitted using maximum marginal likelihood, the log-likelihood ratio comparing two different models will follow an asymptotic $$\chi^2$$ distribution with degrees of freedom equal to the difference in the number of scale parameters in the two models.
If the number of parameters are the same, then the model with the higher likelihood can be chosen.

Such a method of comparing likelihoods is also known as comparing empirical Bayes factors, where the Bayes factor of comparing model $$M_1$$ to model $$M_2$$ is given by

$$
\text{BF}(M_1,M_2) = \frac{\text{marginal likelihood for model } M_1}{\text{marginal likelihood for model } M_2}.
$$

The word 'empirical' stems from the fact that the parameters are estimated via maximum likelihood, which is seen as an empirical Bayes approach.

## Bayesian Model Selection

When the number of predictors $$p$$ is large, then enumerating all possible model likelihoods for comparison becomes infeasible. Greedy selection methods such as forward or backward selection do exist, but these do not fully explore the entire model space.

We employ a fully Bayesian treatment to explore the space of models and obtain estimates for (posterior) model probabilities given by

$$
p(M|\mathbf y) = \int p(\mathbf y|M,\theta) p(\theta|M) p(M) \, \text{d}\theta,
$$

where $$M$$ is a model index and $$\theta$$ are the parameters of the model. The model with the highest probability can be chosen, or a quantity of interest $$\Delta$$ estimated using model averaging techniques over a model set $$\mathcal M$$:

$$
\sum_{M \in \mathcal M} p(M|\mathbf y) \cdot \text{E}[\Delta|M, \mathbf y].
$$

<div>
Posterior model probabilities may also be calculated using Bayes factors and prior model probabilities as 
follows:

$$
p(M|\mathbf y) = \frac{\text{BF}(M,M_k) p(M)}{\sum_{M'}\text{BF}(M',M_k) p(M')}
$$

based on an anchoring model <script type="math/tex">M_k</script>, and typically the null model (model with intercept only) or the full model is chosen.
</div>
{: .notice--info}

## Variable Selection for Linear Models

For linear models of the form 

$$
(y_1,\dots,y_n) \sim \text{N}_n \Big(\beta_0 \mathbf 1_n + \sum_{j=1}^p \beta_j X_j, \Psi^{-1} \Big),
$$

the prior 

$$
(\beta_1, \dots, \beta_p) = \text{N}_p(0, \Lambda X^\top \Psi X \Lambda)
$$

is an equivalent I-prior representation of model [(1)](/intro/#introduction) subject to [(3)](/intro/#eq-iprior) in the feature space of $$\beta$$ under the linear kernel. 

### Stochastic Search Methods

While posterior model probabilities may be enumerated one by one, variable selection with large $$p$$ would fail on listing all $$2^p$$ of these probabilities. Various methods exist in the literature for stochastic search methods via Gibbs sampling, such that only models that have considerable probability of being selected are visited in the posterior MCMC chain.

One such method is by Kuo and Mallick (1998). Each model is indexed by $$\gamma \in \{0,1\}^p$$, with $$j$$th value of zero indicating an exclusion and one an inclusion of the variable $$X_j$$. The following model is estimated via Gibbs sampling:

$$
y_i = \beta_0 + \gamma_1\beta_1 X_1 + \dots + \gamma_p\beta_p X_p + \epsilon_i,
$$

along with normally distributed errors as in [(1)](/intro/#introduction), an I-prior on $$\beta$$, and some suitable priors on $$\gamma$$, the intercept $$\beta_0$$ and error precision $$\Psi$$.

Posterior inclusion probabilities for a particular variable $$X_j$$ can be estimated as $$\frac{1}{T}\sum_{t=1}^T \gamma_j^{(t)}$$, where $$\gamma_j^{(t)}$$ is the $$t$$th MCMC sample. This gives an indication of how often the variable was chosen in all possible models. 

More importantly, posterior model probabilities may be estimated by calculating the proportion of a particular sequence $$\gamma$$, corresponding to a particular model $$M_\gamma$$, appearing in the MCMC samples.

### Simulation Study

A study is conducted to assess the performance of each method (prior) in choosing the correct variables across five different scenarios quantified by the signal to noise ratio (SNR). 
For each scenario, out of 100 variables $$X_1,\dots,X_{100}$$ with pairwise correlation of about 0.5., only $$s$$ were selected to form the "true" model and generate the responses according to the linear model above.
The SNR as a percentage is defined as $$s \%$$, and the five scenarios are made up of varying SNR from high to low: 90%, 75%, 50%, 25%, and 10%.

The experiment is conducted as follows:

1. For each scenario, generate data $$(y, X)$$.
2. Obtain the highest probability model and count the number of false choices made by each method.
3. Repeat 1-2 100 times to obtain averages.
4. Repeat 1-3 for each of the five scenarios.

The results are tabulated below:

*Table 1: Results for the I-prior method*
{: .text-center }

| False choices | 90%  | 75%  | 50%  | 25%  | 10%  |
|---------------|------|------|------|------|------|
| 0-2           | **0.92** | **0.92** | **0.92** | **0.78** | **0.52** |
| 3-5           | 0.08 | 0.08 | 0.08 | 0.20 | 0.28 |
| >5            | 0.00 | 0.00 | 0.00 | 0.02 | 0.20 |

<a></a>

*Table 2: Results for the $$g$$-prior method*
{: .text-center}

| False choices | 90%  | 75%  | 50%  | 25%  | 10%  |
|---------------|------|------|------|------|------|
| 0-2           | 0.00 | 0.00 | 0.00 | **0.67** | **0.88** |
| 3-5           | 0.00 | 0.00 | 0.00 | 0.18 | 0.12 |
| >5            | **1.00** | **1.00** | **1.00** | 0.15 | 0.00 |

<a></a>

*Table 3: Results for the ridge method*
{: .text-center}

| False choices | 90%  | 75%  | 50%  | 25%  | 10%  |
|---------------|------|------|------|------|------|
| 0-2           | 0.00 | 0.00 | 0.00 | **0.42** | **1.00** |
| 3-5           | 0.00 | 0.00 | 0.00 | 0.27 | 0.00 |
| >5            | **1.00** | **1.00** | **1.00** | 0.31 | 0.00 |

<a></a>

*Table 4: Results for the Lasso method*
{: .text-center}

| False choices | 90%  | 75%  | 50%  | 25%  | 10%  |
|---------------|------|------|------|------|------|
| 0-2           | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 3-5           | 0.28 | 0.08 | 0.00 | 0.00 | 0.00 |
| >5            | **0.72** | **0.92** | **1.00** | **1.00** | **1.00** |
