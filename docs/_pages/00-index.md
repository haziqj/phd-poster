---
layout: splash
title: My PhD Project
permalink: /
header:
  overlay_image: /assets/images/main_image_filter.jpg
  overlay_filter: "0.25"
excerpt: 'Regression modelling using priors depending on Fisher information covariance kernels (I-priors) 
<br>
[Thesis (site)](https://haziqj.ml/publication/phd-thesis/){: .btn .btn--light-outline}&nbsp;
<!-- [Thesis (PDF)](http://etheses.lse.ac.uk/3828/1/Jamil__regression-modelling.pdf){: .btn .btn--light-outline}&nbsp; -->
[Poster (PDF)](/my-phd-poster.pdf){: .btn .btn--light-outline}
<br style="line-height: 80px" />
<small>🏆 [2020 Zellner Thesis Award (honourable mention)](https://community.amstat.org/businessandeconomicstatisticssection/new-item/new-item2)</small>'


intro: 
  - excerpt: 'A thesis submitted to the London School of Economics and Political Science for the degree of Doctor of Philosophy in Statistics.'
top_stuff:
  - title: "Introduction"
    excerpt: 'I-priors [(Bergsma, 2019)](https://doi.org/10.1016/j.ecosta.2019.10.002) are a class of objective priors which make use of the Fisher information. Estimation is simple, inference straightforward, and often gives better predictions for new data.'
    url: "/intro/"
    btn_label: "Read More"
    #btn_class: "btn--inverse"
feature_row:
  - image_path: /assets/images/logo_reg_filter_3.jpg
    alt: "regression"
    title: "Regression"
    excerpt: "Simple fitting of various regression models for prediction and inference."
    url: "/regression/"
    btn_label: "Read More"
    btn_class: "btn--inverse"
  - image_path: /assets/images/logo_class_filter.jpg
    alt: "classification"
    title: "Classification"
    excerpt: "Extension to categorical responses for binary and multiclass classification."
    url: "/classification/"
    btn_label: "Read More"
    btn_class: "btn--inverse"
  - image_path: /assets/images/logo_bvs.png
    alt: "var-select"
    title: "Variable Selection"
    excerpt: "A fully Bayesian approach to variable selection for linear models."
    url: "/var-select/"
    btn_label: "Read More"
    btn_class: "btn--inverse"
feature_additional:
  - title: "Abstract"
    excerpt: "Hello"
    url: "/intro/"
  - btn_label: "Read More"
---

{% include feature_row id="top_stuff" type = "center" %}

{% include feature_row %}

<!-- {% include feature_row id="feature_additional" type = "center" %} -->
