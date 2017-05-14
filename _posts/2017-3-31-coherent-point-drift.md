---
layout: post
title: "PyCPD: Tutorial on the Coherent Point Drift Algorithm"
comments: true
published: true
---
*If you are only looking for code for the coherent point drift algorithm in Python, look at this [Pypi](https://pypi.python.org/pypi/pycpd/0.1) package. Or if you prefer to build from source, you can look at the following [Github](https://github.com/siavashk/pycpd).*

## Introduction
During my PhD, I was working on the specific problem of MR-US fusion for prostate biopsies. MR-US fusion simply means Magnetic-Resonance-UltraSound fusion. This typically involves solving a registration problem which aims to find the optimal transformation between a source (MR) and a target (US) prostate image. Both MR and US images comprise of multiple slices that span the prostate to create a volumetric view of the anatomy.

A popular approach for MR-US fusion prostate biopsy is to use a surface-based registration method. In this approach the prostate is first segmented, meaning that a trained radiologist contours and extracts the surface of the prostate in both US and MR volumes. Then, surface-based registration is performed to align the two volumes.

A related problem to surface-based registration is point cloud registration. They are some times used interchangeably in the literature. Strictly speaking, surface-based registration deals with surfaces that have connectivity information (think faces). Point cloud registration, on the other hand deals with, well, clouds of points without connectivity information (think vertices).

A point cloud registration, method that I found particularly useful was the [Coherent Point Drift](https://arxiv.org/abs/0905.2635) (CPD) algorithm by Myronenko and Song. They formulate the registration as a probability density estimation problem, where one point cloud is represented using a Gaussian Mixture Model (GMM) and the other point cloud is observations from said GMM.

## Point Cloud Registration
Let's start off with a simple toy example. Assume that we have two point clouds \\(X = \left\\{ X1, X2, X3 \right\\} \\) and \\(Y = \left\\{ Y1, Y2, Y3 \right\\} \\). These point clouds are shown in Figure 1 with red and blue circles, respectively. Our goal is to find the transformation that best aligns the two point clouds.

In this toy example, the unknown transformation is a rotation around the origin (parameterized by \\(\theta\\)) followed by a translation (parameterized by \\(t\\)).Assume, the actual value of the unknown parameters is \\( \left\\{ \theta=30^\circ, t=(0.2, 0.2) \right\\} \\). We can use numpy to define the two point clouds as seen in the following code snippet:

 ```python
import numpy as np

# transformation parameters
theta = np.pi/6.0
t = np.array([[0.2], [0.2]])

# rotation matrix
R = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta), np.cos(theta)]])
X = np.array([[0, 0, 1], [0, 1, 0]])
Y = np.dot(R, X) + t

xLabels = ["X1", "X2", "X3"]
yLabels = ["Y1", "Y2", "Y3"]
```

Plotting the two point clouds results in Figure 1. Now, since this is a toy example, we already know the correspondences between points in the two point clouds. The corresponding points are linked using the black dashed line. If the correspondences are known, the solution to the rigid registration is known as the orthogonal Procrustes problem:

$$\mathrm{argmin}_{R,t}\left\Vert{X - RY - t}\right\Vert^2, \mathrm{s.t} R^TR=I$$

![Point Cloud Registration](../notebooks/coherent-point-drift/registration1_files/registration1_1_0.png)<br/>

## Missing Correspondences
When correspondence is not explicitly known, point cloud registration algorithms implicitly assume that correspondence can be inferred through point proximity. In other words, points that are spatially close to each other correspond to one another.

We can assign an arbitrary correspondence probability to point clouds based on proximity. Figure 2 shows an example probability distribution based on proximity.

Points that are closer than a radius of \\(r=0.2\\) would confident matches, and we would assign a correspondence confidence of \\(p=1.0\\) to them. Pairs such as \\(\left\(X1, Y1\right\)\\) and \\(\left\(X2, Y2\right\)\\) pairs have a distance between \\(r=0.2\\) and \\(r=0.4\\) units are probable but not confident matches, so we could assign a probability of \\(p=0.5\\) to them. Beyond this, there is probably no correspondence, so our probability would drop to zero.

Even though this approach is quite simple, it provides two distinct advantages. First, it allows us to assign correspondences so that we can solve the registration as a Procrustes problem. Furthermore, it also allows us to weigh the loss functional according to the correspondence probability.
<br>
![Point Cloud Correspondences](../notebooks/coherent-point-drift/registration1_files/registration1_2_0.png)<br/>

## Gaussian Mixture Models
We will now side step from the point cloud registration problem briefly. Instead of dealing with \\(X, Y\\) point clouds directly, we construct a GMM from the moving point cloud, \\(Y\\), and treat \\(X\\) as observations from that GMM. In Figure 3, we have constructed a GMM where the three Gaussians have a variance of 0.75 units. Blue points, i.e. Gaussian centroids, are the transformed moving points (\\(Y\\)). Red points, i.e. the fixed point cloud \\(X\\), are observations from this GMM. Isocontours represent the log-likelihood that red points are sampled from this GMM.

<br>
![Constructed GMM](../notebooks/coherent-point-drift/registration1_files/registration1_3_0.png)<br/>

## GMM-based Registration
In order to perform registration, we have to solve correspondence and moving point cloud transformation problems simultaneously. This is done through expectation-maximization (EM) optimization. To solve the correspondence problem, we need to find which Gaussian the observed point cloud was sampled from (E-step). This provides us with correspondence probability, similar to Figure 2. Once correspondences probabilities are known, we maximize the negative log-likelihood that the observed points were sampled from the GMM with respect to transformation parameters (M-step).

## E-step
In Figure 3, if there was only one Gaussian component in the mixture, then the probability that a point \\(x\\) is sampled from this Gaussian is given using probability density distribution of the [multivairate normal distribution](https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Density_function). For the 2D case, with isotropic Gaussians, this simplifies to:

$$p(x) = \frac{1}{\sqrt{2\pi\sigma^2}}\exp{-\frac{\left\Vert{X - RY - t}\right\Vert^2}{2\sigma^2}}$$

However, since we are dealing with multiple Gaussians, we need to normalize this probability by the contribution of all Gaussian centroids. In the pycpd package, this is achieved (minor tweaks to simplify the explanation) using the following snippet:

```python
import numpy as np

def EStep(X, Y, sigma2):
  M = Y.shape[0] # number of moving points
  D = Y.shape[1] # dimensionality of moving points
  N = X.shape[0] # number of fixed points
  # Probability matrix: p_{ij} is the probability
  # that moving point i corresponds to fixed point j
  P = np.zeros((M, N))

  for i in range(0, M):
      diff     = X - np.tile(Y[i, :], (N, 1))
      diff    = np.multiply(diff, diff)
      P[i, :] = P[i, :] + np.sum(diff, axis=1)

  P = np.exp(-P / (2 * sigma2))
  den = np.sum(P, axis=0)
  den = np.tile(den, (M, 1))
  den[den==0] = np.finfo(float).eps

  P = np.divide(P, den)
  Pt1 = np.sum(P, axis=0)
  P1  = np.sum(P, axis=1)
  Np  = np.sum(P1)
  return P, Pt1, P1, Np
```

## M-step
Once correspondence probabilities are known, i.e. \\(P\\), we can solve for the transformation parameters. In the case of rigid registration, these transform parameters are the rotation matrix and the translation vector. In the pycpd package, this is achieved using the following snippet:

```python
import numpy as np

def MStep(X, Y, P):
  updateTransform(X, Y, P, P1, Np)

def updateTransform(X, Y, P):
  muX = np.divide(np.sum(np.dot(P, X), axis=0), Np)
  muY = np.divide(np.sum(np.dot(np.transpose(P), Y), axis=0), Np)

  XX = X - np.tile(muX, (N, 1))
  YY = Y - np.tile(muY, (M, 1))

  A = np.dot(np.transpose(XX), np.transpose(P))
  A = np.dot(A, YY)

  U, _, V = np.linalg.svd(A, full_matrices=True)
  C = np.ones((D, ))
  C[D-1] = np.linalg.det(np.dot(U, V))

  R = np.dot(np.dot(U, np.diag(C)), V)

  YPY = np.dot(np.transpose(P1), np.sum(np.multiply(YY, YY), axis=1))

  s = np.trace(np.dot(np.transpose(A), R)) / YPY

  t = np.transpose(muX) - s * np.dot(R, np.transpose(muY))
  return s, R, t, A, XX

def updateVariance(R, A, XX, Np, D):
  trAR = np.trace(np.dot(A, np.transpose(R)))
  xPx = np.dot(np.transpose(Pt1), np.sum(np.multiply(XX, XX), axis =1))
  sigma2 = (xPx - s * trAR) / (Np * D)
```
