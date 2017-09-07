---
title: Examples
layout: single
permalink: /examples/
gallery_gcse:
  - url: /images/eg1-1.png
    image_path: /images/eg1-1.png
    title: "Comparison of intercepts."
  - url: /images/eg1-2.png
    image_path: /images/eg1-2.png
    title: "Comparison of slopes."    
gallery_vowel:
  - url: /images/eg6-1.png
    image_path: /images/eg6-1.png
    title: "Canonical kernel vowel results."
  - url: /images/eg6-2.png
    image_path: /images/eg6-2.png
    title: "fBm-0.5 kernel vowel results."    
gallery_cornwall:
  - url: /images/eg7-1.png
    image_path: /images/eg7-1.png
    title: "Spatial distribution of BTB types in Cornwall."
  - url: /images/eg7-2.png
    image_path: /images/eg7-2.png
    title: "Predicted spatial distribution of BTB."    
  - url: /images/eg7-3.png
    image_path: /images/eg7-3.png
    title: "Predicted spatial distribution of BTB across the years."        
---

<script type="text/x-mathjax-config">
MathJax.Hub.Config({
  TeX: { equationNumbers: { autoNumber: "AMS" } }
});
</script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.0/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

{% include toc %}

Some brief examples of regression and classification using I-priors.

## Multilevel Analysis of Pupils' GCSE Scores

***Aim***: Obtain estimates for random intercepts and slopes for a multilevel data set, and compare with standard random effects model estimates.

***Data***: GCSE scores ($$y_{ij}$$) for 4,059 pupils at 65 inner London schools ($$j$$), together with their London reading test results ($$x_{ij}$$).

***Model***:

$$
y_{ij} = f_1(x_{ij}) + f_2(j) + f_{12}(x_{ij}, j) + \epsilon_{ij}
$$

with $$f_1$$ in the canonical RKHS, $$f_2$$ in the Pearson RKHS, and $$f_{12}$$ in the tensor product space.

***Results***: Good agreement between I-prior estimates and standard random effects model estimates.

{% include gallery id="gallery_gcse" caption="Estimated intercepts and slopes for school achievement data under varying intercepts (left) and varying slopes model (right). The numbers plotted are the school indices with the identity line for reference." %}

## Longitudinal Analysis of Cow Growth

***Aim***: Discern whether there is a difference in the two treatments given to the cows, and whether this effect varies among individual cows.

***Data***: A balanced longitudinal data set of weight measurements $$y_{it}$$ for 60 cows ($$x_{1it}$$) at different time points $$t = $$ 1, ..., 11. 30 cows were randomly assigned to treatment group A, and the other 30 to treatment group B.

***Model***: Assume a smooth effect of time and nominal effect of cow i.d. and treatment group:

$$
y_{it} = f_1(x_{1it}) + f_2(x_{2it})+ f_3(t) + f_{13}(x_{1it}, t) + f_{23}(x_{2it}, t) + f_{123}(x_{1it}, x_{2it}, t) + \epsilon_{it}
$$

with $$f_1$$ and $$f_2$$ in the Pearson RKHS, $$f_3$$ in the fBm-0.5 RKHS, and the interaction functions in the appropriate tensor product space. This model can be succintly represented as

$$
y_{i} = f_{1t}(x_{1it}) + f_{2t}(x_{2it})+ f_{12t}(x_{1it}, x_{2it}) + \epsilon_{i}.
$$

***Results***: Four models were fitted, and results tabulated below.

|   | Explanation                           | Model               | Log-lik. |   No. of param. |
|---|---------------------------------------|---------------------|---------:|----------------:|
| 1 | Growth due to cows only               | $$f_{1t}$$          |  -2792.2 |               3 | 
| 2 | Growth due to treatment only          | $$f_{2t}$$          |  -2295.2 |               3 | 
| 3 | Growth due to both cows and treatment | $$f_{1t} + f_{2t}$$ |  -2270.9 |               4 |
| 4 | Growth due to both cows and treatment, with treatment varying among cows | $$f_{1t} + f_{2t} + f_{12t}$$ | -2250.9 | 4 |  

To test for a treatment effect, we test the signifiance of the scale parameter for the treatment variable in Model 2 (p-value < 10<sup>-6</sup>). To test whether treatment effect differs among cows, we compare likelihoods for Models 3 and 4.

## Predicting Fat Content from Spectrometric Analysis

***Aim***: Predict fat content of meat samples from its spectrometric curves (Tecator data set).

***Data***: For each meat sample, 100 channel spectrum of absorbances ($$x_i \in \mathbb R^{100}$$) together with the contents of moisture, fat ($$y_i$$) and protein measured in percent. 160 samples were used to train the model, and 55 were used for testing.

<figure>
  <a href="/images/eg3-1.png"><img src="/images/eg3-1.png"></a>
  <figcaption>Sample of spectrometric curves (the functional covariates) used to predict fat content (numbers shown in boxes) of meat.</figcaption>
</figure>

***Model***: Take first differences of $$x$$ (see [here](/intro/#the-sobolev-hilbert-inner-product) for explanation) and obtain $$z_i \in \mathbb R^{99}$$. We then assume various effects of $$z_i$$ on $$y_i$$ using the linear, polynomial (quadratric and cubic) and fBm (smooth).

***Results***: Training and test errors are tabulated below. The smooth I-prior model outperforms methods such as Gaussian process regression (13.3), kernel smoothing (1.85), single index models (1.18) and sliced inverse regression (0.90) in test error rates.

|   | Model            | Training error | Test error |
|---|------------------|---------------:|-----------:|
| 1 | Linear           |           2.85 |       3.24 |
| 2 | Quadratic        |           0.72 |       1.23 |
| 3 | Cubic            |           0.99 |       1.65 |
| 4 | Smooth (fBm-0.5) |           0.00 |       0.67 |

## Predicting Cardiac Arrhythmia

***Aim***: Predict whether patients suffers from a cardiac disease based on various patient profiles such as age, height, weight and a myriad of ECG data.

***Data***: There are 451 observations of 279 predictors and binary variables indicating whether the patient had arrhythmia or not. All 279 predictors are continuous, and they were standardised beforehand. 

***Model***: A binary I-probit model using the linear and fBm-0.5 kernel to predict the probability of having cardiac arrhythmia:

$$
p_i = \Phi\big(f(x_i)\big)
$$

where $$f$$ lies in either the canonical or fBm-0.5 RKHS. Since the variables $$x_i \in \mathbb R^{279}$$ were standardised, it is sufficient to use a single scale parameter.

***Results***: In order to test predictive performance, the data was randomly split into training sets of sizes 50, 100, and 200, with the remaining forming the test set. 
The model was fitted and out-of-sample misclassification rates noted. 
This was then repeated 100 times to obtain averages and standard errors. 
The I-probit models outperformed some of the more popular classifiers, including Gaussian process classification, nearest shrunken centroids, support vector machines and $$k$$-nearest neighbours.

<figure>
  <a href="/images/eg4-1.png"><img src="/images/eg4-1.png"></a>
  <figcaption>Plot of mean test error rates together with the 95% confidence intervals for the I-probit models and six popular classifiers.</figcaption>
</figure>

## Meta-analysis of Smoking Cessation

***Aim***: Inference on the effect size of nicotine gum treatment on smoking cessation based on data from multiple, independent studies.

***Data***: Records of whether each of the 5,908 patients in 27 separate studies ($$j$$) successfully quit smoking ($$y_{ij}$$) and also whether they were subjected to the actual treatment or a placebo ($$x_{ij}$$).

***Model***: The Bernoulli probabilities $$p_{ij}$$ for each patient $$y_{ij}$$ are regressed against the treatment group indicators $$x_{ij}$$ and each patients' study group $$j$$ via the probit link:

$$
p_i = \Phi\big(f_1(x_{ij}) + f_2(j) + f_{12}(x_{ij}, j)\big)
$$

where $$f_1$$ and $$f_2$$ lie in the Pearson RKHS, and $$f_{12}$$ in their tensor product space.

***Results***: Fitted model probabilities were obtained, and log odds ratios calculated and compared against the standard logistic random effects model.

<figure>
  <a href="/images/eg5-1.png"><img src="/images/eg5-1.png"></a>
  <figcaption>Forest plot of effect sizes (log odds ratios) in each study group as well as the overall effect size together with their 95% confidence bands. Sizes of the points indicate relative sample sizes.</figcaption>
</figure>

## Vowel Recognition

***Aim***: Multiclass classification of Deterding's vowel data set, in which the task is to correctly associate digitised speech recordings with the vowel used for that speech.

***Data***: Multiple speakers, male and female, uttered 11 vowels each. These were then processed using speech processing techniques. The result is 990 data points consiting of a ten-dimensional numerical predictor $$x_i$$ and also the class of vowel for each data point $$y_i$$. These were split roughly equally into a training and test set.

***Model***: A multinomial I-probit model to predict the class probabilities for each data point:

$$
p_{ij} = g^{-1} \big( f_j(x_i) \big)
$$

where $$f_j$$ lies in either the canonical or fBm-0.5 RKHS for each class $$j \in \{1,\dots,11\}$$ and $$g^{-1}$$ is the function as described [here](/classification/#multinomial-responses). The predicted class is given by $$\arg\max_j p_{ij}$$.

***Results***: Out-of-sample misclassification rates for the fBm I-prior model was the best among several methods which include logistic regression, linear and quadratic discriminant analysis, decision trees, neural networks, nearest neighbours, and flexible disciminant analysis. While the canonical I-probit did not fare as well, it did however gave improvement over linear regression.

{% include gallery id="gallery_vowel" caption="Confusion matrices for the predicted classes under a linear kernel (left) and an fBm-0.5 kernel (right). Note that the maximum value for any one cell is 42, and blank cells indicate nil values." %}

## Spatio-temporal Analysis of Bovine Tubercolosis in Cornwall

***Aim***: To determine the existence of spatial segregation multiple types of bovine tubercolosis (BTB) in Cornwall, and whether the spatial distribution had changed over time.

***Data***: 919 cases of BTB had been recorded over a period of 14 years all over cornwall. Data pertaining to the location $$x_{1i}$$ (Northings and Eastings) and year of the occurrence $$x_{2i}$$ is available. There are four types of BTB which are most commonly occurring, though another class of "others" was considered totalling five classes.

***Model***: A multinomial I-probit model regressing the class probabilities on $$x_{1i}$$ and $$x_{2i}$$:

$$
p_{ij} = g^{-1} \big( f_{1j}(x_{1i}) + f_{2j}(x_{2i}) + f_{12j}(x_{1i},x_{2i}) \big).
$$

We assume a smooth effect of $$x_{1i}$$ - so $$f_{1j}$$ lies in the fBm-0.5 RKHS. We have two choices for $$x_{2i}$$: 1) Assume a similar smooth effect; or 2) Aggregate the data into distinct time periods so that $$x_{2i}$$ emits a nominal effect (the Pearson RKHS of functions).

***Results***: The plots indicate that there is in fact spatial segregation between the various types of BTB in Cornwall. These can be tested formally by performing tests of significance on the scale parameters associated with location and time.

{% include gallery id="gallery_cornwall" caption="Plots of the spatio-temporal distribution of BTB types in Corwall, together with model predictions." %}

