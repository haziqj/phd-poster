---
title: Classification with I-priors
layout: single
permalink: /classification/
---

<script type="text/x-mathjax-config">
MathJax.Hub.Config({
  TeX: { equationNumbers: { autoNumber: "AMS" } }
});
</script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.0/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

{% include toc %}

The I-prior methodology is extended from the continuous-response case to the categorical-response case - we call this the *I-probit* model. 
Estimation involves some form of approximation as the marginal density cannot be found in closed form.

# Categorical Responses

Suppose that each of the response variables $$y_i$$ takes on one of the values from $$\{1,\dots,m\}$$, and that

$$
y_i \sim \text{Cat}(p_{i1}, \dots, p_{im})
$$

with probability mass function

$$
p(y_i) = \prod_{j=1}^m p_{ij}^{y_{ij}}, \hspace{1cm} y_{ij} = [y_i = j]
$$

satisfying $$p_{ij} \geq 0$$, and $$\sum_{j=1}^m p_{ij} = 1$$.

The categorical distribution is a special case of the multinomial distribution, and can be seen as a generalisation of the binomial distribution. Here, we have used the notation $$[\cdot]$$ to denote the Iverson bracket - $$[A]$$ equals one if the proposition $$A$$ is true, and zero otherwise.
{: .notice--info}

The assumption of normality on $$y_i$$ is now highly inappropriate. In the spirit of generalised linear models, we model 

$$
\text{E}[y_{ij}] = p_{ij} = g^{-1}\big( f_j(x_{ij})\big)
$$

with some link function $$g:[0,1] \to \mathbb R$$. We can now place an I-prior on the functions $$f_j$$. 
As we will see [later](#variational-approximation), the *probit* link $$g = \Phi^{-1}$$ is preferred, where $$\Phi$$ is the cumulative distribution function for a standard normal distribution.

### Binary Responses

In the simplest case where $$m=2$$, $$y_i$$ follows a binomial distribution with success probability $$p_i$$. The probit link can be motivated through the use of continuous, underlying latent variables $$u_i,\dots,u_n^*$$ such that

$$
y_i =
\begin{cases}
1 & \text{if } u_i \geq 0 \\
0 & \text{if } u_i < 0. \\
\end{cases}
$$

We can then model these auxiliary random variables $$u_i$$ using an I-prior as usual with fixed error precision $$\Psi = I_n$$. Thus,

\begin{align}
p_i = \text{P}[y_i = 1] &= \text{P}[u_i \geq 0] \nonumber \\\
&= \text{P}[f(x_i) + \epsilon_i \geq 0] \nonumber \\\
&= \Phi \big(f(x_i) \big). \nonumber
\end{align}

There is no loss of generality compared with using an arbitrary threshold $$\tau$$ (other than zero) or error precision $$\psi$$ in $$\Psi = \psi I_n$$.
{: .notice--info}

### Multinomial Responses

<!-- The latent variable approach can also be used here:

$$
y_i =
\begin{cases}
1 & \text{if } y_{i1}^* \geq y_{i2}^*,y_{i3}^*, \dots, y_{im}^* \\
2 & \text{if } y_{i2}^* \geq y_{i1}^*,y_{i3}^*, \dots, y_{im}^* \\
\vdots \\
m & \text{if } y_{im}^* \geq y_{i1}^*,y_{i2}^*, \dots, y_{im-1}^* \\
\end{cases}
$$ -->

The approach we take is to model each probability class $$p_{ij}$$
<!-- , and therefore its corresponding latent variable,  -->
using separate regression functions $$f_j$$ and separate I-priors (thus the index $$j$$ on the functions). 
In the most general setting, there would be $$m$$ sets of hyperparameters to estimate (one for each class), though it is possible to assume some common values among classes.

Using the latent variable motivation as in the binomial case, we find that 

$$
p_{ij} = \text{E}_Z\Bigg[\mathop{\prod_{k=1}^m}_{k \neq j} \Phi\big(Z + f_j(x_i) - f_k(x_i)\big) \Bigg].
$$

For $$m > 3$$ this is known not to have a closed-form expression, but nonetheless is easily evaluated using quadrature methods.

It is also possible to reparameterise the model (anchoring on one latent variable as the reference class and working with the latent di erences) so that only $$m − 1$$ I-priors are required. It is easily seen that using this method with $$m=2$$ reduces the model to the same binomial model described above.
{: .notice--info}

# Estimation

The marginal likelihood

$$
p(\mathbf y) = \int \prod_{i=1}^n \prod_{j=1}^m \left[ \big\{ g^{-1}\big(f_j(x_i)\big) \big\}^{[y_i=j]} \cdot \text{N}_n (\mathbf{f}_{0j}, \mathcal I[f_j]) \, \text{d}\mathbf f_j \right],
$$

on which the posterior depends, can no longer be found in closed form, unlike in the normal regression case. Several methods can be used and are described below in an order analogous to the [methods described](/intro/#estimation) in the normal regression model.

### Laplace's Method

Suppose that we are interested in 

$$ 
p(\mathbf f \vert \mathbf y) \propto p(\mathbf y \vert \mathbf f) p(\mathbf f) =: e^{Q(f)},
$$

with normalising constant $$p(\mathbf y) = \int e^{Q(f)} \, \text{d}\mathbf f$$. The Taylor expansion of $$Q$$ about its mode $$\mathbf f^*$$ 

$$
Q(\mathbf f) \approx Q(\mathbf f^*) - \frac{1}{2} (\mathbf f - \mathbf f^*)^\top A (\mathbf f - \mathbf f^*)
$$

is recognised as the logarithm of an unnormalised Gaussian density, with $$A = -\text{D}^2 Q(\mathbf f^*)$$ being the negative Hessian of $$Q$$ evaluated at $$\mathbf f^*$$.

The posterior density $$p(\mathbf f \vert \mathbf y)$$ is approximated by $$\text{N}_n(\mathbf f^*, A^{-1})$$, and the marginal by

$$
p(\mathbf y) \approx (2\pi)^{n/2} \vert A \vert^{-1/2} p(\mathbf y \vert \mathbf f^*) p(\mathbf f^*).
$$

The marginal density can then be maximised with respect to the hyperparameters using Newton-based methods. However, each Newton step requires finding the posterior modes $$\mathbf f^*$$, which is difficult for very large $$n$$.

### Variational Approximation

An approximation $$q(\mathbf f)$$ to the true posterior density $$p(\mathbf f \vert \mathbf y)$$ is considered, with $$q$$ chosen to minimise the Kullback-Leibler divergence (under certain restrictions):

$$
\text{KL}(q || p) = - \int \log \frac{p(\mathbf f \vert \mathbf y)}{q(\mathbf f)} q(\mathbf f) \, \text{d}\mathbf f.
$$

The name "variational" stems from the fact that we are seeking to minimise a functional (the Kullback-Leibler divergence) which uses calculus of variations techniques. Of course it would be impossible to minimise the KL over all possible functions $$q$$, so some restrictions are required. We use the mean-field factorisation assumption, which considers only densities which factorises completely over its components, i.e. densities of the form $$q(z_1, \dots, z_N) = \prod_{i=1}^N q(z_i).$$
{: .notice--info}

By assuming priors on the hyperparameters, we work in a fully Bayes setting and append these model hyperparameters $$\mathbf \theta$$ to $$\mathbf f$$ to form $$\mathbf z = (\mathbf f, \mathbf \theta)$$ and employ a variational approximation to the posterior density $$p(\mathbf z \vert \mathbf y)$$. The result is a sequential updating scheme similar to the EM algorithm.

This variational-EM algorithm works harmoniously with exponential family distributions, and as such the probit link provides an advantage over other link functions such as the more popular logit. In fact, all of the required posterior densities, with the exception of the $$y_i$$, involve the normal distribution. The $$y_i$$ of course follow a categorical distribution.

The marginal likelihood is approximated by a quantity known as the *variational lower bound*, and is given by $$\mathcal L = \text{E}_z[\log p(\mathbf y, \mathbf z)] \, / \, \text{E}_z[\log q(\mathbf z)]$$, where expectation is taken over the approximate posterior distribution.
{: .notice--info}

<figure>
  <a href="/images/compare_lap_var.png"><img src="/images/compare_lap_var.png"></a>
  <figcaption>Comparison of the approximations for Laplace's method and the variational method.</figcaption>
</figure>

### Markov Chain Monte Carlo

In keeping with the fully Bayes theme, MCMC samplers such as Gibbs or Hamiltonian Monte Carlo can also be used to estimate these I-probit models. In our experience, these methods can be computationally slow, and sampling difficulty often arises which result in unreliable posterior samples.

# Modelling and Prediction

The advantages of I-priors in the normal model extend even to the I-probit model. This includes being able to simply model various types of regression models by choosing appropriate kernel functions for the covariates. 

For prediction purposes, we can derive the posterior predictive distribution given a new data point $$x_\text{new}$$ as follows:

$$
p(y_{\text{new},j} \vert \mathbf y) \approx \int \prod_{j=1}^m \Big[ p(y_{\text{new},j} \, \vert \, f_{\text{new},j}) q(f_{\text{new},j}) \, \text{d} f_{\text{new},j} \Big],
$$

where $$f_{\text{new},j} = f_j(x_\text{new})$$ using the posterior estimate of $$f_j$$. This complex integral reduces to the expectation of products of standard normal CDFs (similar to equation [1](#multinomial-responses)).

For examples of I-probit models used for binary and multiclass classification, meta-analysis, and spatio-temporal modelling, see the [Examples](/examples/) section.


