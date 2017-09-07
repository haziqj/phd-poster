---
title: My PhD Project
layout: splash
header:
  overlay_image: /images/main_image.png
  overlay_filter: "0.45"
excerpt: 'Regression modelling using priors with Fisher information covariance kernels (I-priors) <br /><br /> [Poster (PDF)](#link){: .btn .btn--light-outline}&nbsp;[View Source](https://github.com/haziqj/phd-poster/){: .btn .btn--light-outline}'
intro: 
  - excerpt: 'A project submitted to the London School of Economics and Political Science for the degree of Doctor of Philosophy in Statistics.'
top_stuff:
  - title: "Introduction"
    excerpt: 'I-priors (Bergsma, 2017) are a class of objective priors which make use of the Fisher information. Estimation is simple, inference straightforward, and often gives better predictions for new data.'
    url: "/intro/"
    btn_label: "Read More"
    #btn_class: "btn--inverse"
feature_row:
  - image_path: /images/logo_reg.png
    alt: "regression"
    title: "Regression"
    excerpt: "Simple fitting of various regression models for prediction and inference."
    url: "/regression/"
    btn_label: "Read More"
    btn_class: "btn--inverse"
  - image_path: /images/logo_class.png
    alt: "classification"
    title: "Classification"
    excerpt: "Extension to categorical responses for binary and multiclass classification."
    url: "/classification/"
    btn_label: "Read More"
    btn_class: "btn--inverse"
  - image_path: /images/logo_bvs.png
    alt: "var-select"
    title: "Variable Selection"
    excerpt: "A fully Bayesian approach to variable selection for linear models."
    url: "/var-select/"
    btn_label: "Read More"
    btn_class: "btn--inverse"
---
<!--
I-priors are a class of objective priors on regression functions which make use of its Fisher information in a vector space framework. We present firstly some methodology and computational work on estimating regression functions by working in the appropriate reproducing kernel Hilbert space of functions and assuming an I-prior on the function of interest. Secondly, work on extending the I-prior methodology to categorical responses for classification is presented, in which estimation is performed using a variational approximation to the likelihood. Finally, a fully Bayes approach is considered where I-priors are used for variable selection.
-->

{% include feature_row id="top_stuff" type = "center" %}

{% include feature_row %}

<!-- ```r

      R> (mod <- ipriorBVS(y ~ ., data))
      ##             PIP     1     2     3        
      ## X.1       0.979     x     x     x         
      ## X.2       0.973     x     x     x          
      ## X.3       0.425           x                
      ## X.4       0.991     x     x     x         
      ## X.5       0.194                 x          
      ## PMP             0.439 0.321 0.103  
      ## BF              1.000 0.730 0.235  

``` -->
<!-- {% include feature_row id="intro" type="center" %} -->

