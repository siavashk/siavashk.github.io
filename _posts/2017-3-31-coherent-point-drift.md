---
layout: post
title: "PyCPD: Tutorial on the Coherent Point Drift Algorithm"
comments: true
---
*If you are only looking for code for the coherent point drift algorithm in Python, look at this [Pypi](https://pypi.python.org/pypi/pycpd/0.1) package. Or if you prefere to build from source, you can look at the following [Github](https://github.com/siavashk/pycpd).

## Introduction
During my PhD, I was working on the specific problem of MR-US fusion for prostate biopsies. MR-US fusion simply means Magnetic-Resonance-UltraSound fusion. This typically involves solving a registration problem which aims to find the optimal transformation between a source (MR) and a target (US) prostate image. Both MR and US images comprise of multiple slices that span the prostate to create a volumetric view of the anatomy.

A popular approach for MR-US fusion prostate biopsy is to use a surface-based registration method. In this approach the prostate is first segmented, meaning that a trained radiologist contours and extracts the surface of the prostate in both US and MR volumes. Then, surface-based registration is performed to align the two volumes.

A related problem to surface-based registraion is point cloud registration. They are some times used interchangeably in the literature. Strictly speaking, surface-based registraion deals with surfaces that have connectivity information (think faces). Point cloud registration, on the other hand deals with, well, clouds of points without connectivity information (think vertices).

 A point cloud registration, method that I found particularly useful was the [Coherent Point Drift](https://arxiv.org/abs/0905.2635) (CPD) algorithm by Myronenko and Song. They formulate the registration as a probability density estimation problem, where one point cloud is represented using a Gaussian Mixture Model (GMM) and the other point cloud is observations from said GMM.

 
