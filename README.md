# DCTM

Repository for paper **A flexible deep learning framework for survival analysis with medical data**
[arxiv](https://arxiv.org/abs/2210.11366)

**Abstract.** Medical imaging data and electronic health records are an
integral part of clinical routine and research for prognostication of
patient survival and thus directly inform patient management. However,
standard regression models used to derive patient prognoses are
ill-equipped to handle such non-tabular data directly. Several neural
network architectures based on classification or the Cox model have
been proposed. Here, we present deep conditional transformation models
(DCTMs) for survival applications with medical imaging data. DCTMs
include the Cox model as a special case, but parameterize the log
cumulative baseline hazards via Bernstein polynomials and allow the
specification of non-linear and non-proportional hazards for both
tabular and non-tabular data and extend to all types of uninformative
censoring. DCTMs yield moderate to large performance gains over
state-of-the-art deep learning approaches to survival analysis on a
multitude of publicly available datasets featuring tabular or imaging
data from radiology and pathology.