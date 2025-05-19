---

title: "xr_fresh: Automated Gridded Time Series Feature Extraction for Remote Sensing Data"
author:
- name: Michael L. Mann
  affiliation: The George Washington University, Washington DC 20052
  thanks: Corresponding author. Email mmann1123@gmail.com
 
header-includes:
  - \usepackage{geometry}
  - \usepackage{pdflscape}
  - \usepackage{longtable}
  - \usepackage{fancyhdr}
  - \usepackage{float}
  - \usepackage{graphicx}
  - \usepackage{amsmath}
  - \usepackage{amsfonts}
  - \usepackage{lineno}
  - \usepackage{array}
  - \usepackage{booktabs}
  - \usepackage{caption}
  - |
   ```{=latex}
   \pagestyle{fancy}
   \fancyhf{}
   \rfoot{\thepage}
   \renewcommand{\headrulewidth}{0pt}
   \renewcommand{\footrulewidth}{0pt}
   \fancypagestyle{plain}{
     \fancyhf{}
     \rfoot{\thepage}
     \renewcommand{\headrulewidth}{0pt}
     \renewcommand{\footrulewidth}{0pt}
   }
   ```

abstract: |
  The extraction of meaningful features from gridded time series data in remote sensing is critical for environmental monitoring, agriculture, and resource management. xr_fresh extends the methodology of the Python package tsfresh by applying efficient, automated feature extraction techniques to pixel-based time series derived from satellite imagery datasets. By computing a comprehensive set of statistical and temporal features, xr_fresh allows for scalable feature engineering suitable for classical machine learning models. Utilizing parallelized computation via the xarray, jax and Dask libraries, xr_fresh significantly reduces computational time, making it ideal for large-scale remote sensing applications. The package integrates smoothly with established Python geospatial libraries, facilitating immediate usability in exploratory analyses and operational systems. This paper demonstrates xr_fresh's capabilities through case studies involving crop classification tasks using Sentinel-2 imagery, showcasing its potential for enhancing the accuracy and interpretability of remote sensing models.
--- 

Keywords: Remote sensing, feature extraction, time series, machine learning, crop classification, xarray, Dask.


\linenumbers  
\modulolinenumbers[1]  
\pagewiselinenumbers  
\newgeometry{margin=1in}
\captionsetup{justification=raggedright, singlelinecheck=false}
  
<!-- compile working with:
pandoc writeup.md --template=mytemplate.tex -o output.pdf --bibliography=refs.bib --pdf-engine=xelatex --citeproc 



# convert to word doc (2 steps)
pandoc writeup.md --template=mytemplate.tex \
  --from markdown+raw_tex \
  --to latex \
  --bibliography=refs.bib \
  --citeproc \
  -o output.tex

pandoc output.tex --from latex --to docx -o output.docx
-->
<!-- 
Look at https://mpastell.com/pweave/docs.html -->

## Introduction

Recent advancements in satellite technology and open-access remote sensing data have significantly enhanced our capability to monitor environmental phenomena through gridded time series analysis. Efficiently extracting relevant time series features at scale remains challenging, necessitating automated and robust methods. Inspired by tsfresh (Time Series FeatuRe Extraction on basis of Scalable Hypothesis tests), we introduce `xr_fresh`, tailored specifically for remote sensing datasets, to address this need by automating the extraction of time series features on a pixel-by-pixel basis.


Gridded time series data from satellites, such as Sentinel-2, contain rich temporal information essential for applications like crop type classification, environmental monitoring, and natural resource management. Traditional methods rely heavily on manually engineered features, which are time-consuming, require domain expertise, and often lack scalability.

To address these issues, `xr_fresh` automates the extraction of salient temporal and statistical features from each pixel’s time series. By leveraging automated feature extraction, `xr_fresh` reduces manual intervention and enhances reproducibility in remote sensing workflows.

## Problems and Background 

A remote sensing image time series can be represented as a three-dimensional array with spatial dimensions $x$ and $y$, and temporal dimension $z$. Each pixel at location $(x_i, y_j)$ holds a time series:

$$
\mathcal{D} = \{ X_{i,j} \in \mathbb{R}^T \mid i = 1, \ldots, H; j = 1, \ldots, W \}
$$

where $H$ and $W$ are the height and width of the image, and $T$ is the number of temporal observations (e.g. monthly composites or daily acquisitions).

To prepare this data for use in supervised or unsupervised machine learning, each pixel time series $X_{i,j} = (x_{i,j,1}, x_{i,j,2}, \ldots, x_{i,j,T})$ is transformed into a feature vector:

$$
\vec{x}_{i,j} = \left(f_1(X_{i,j}), f_2(X_{i,j}), \ldots, f_M(X_{i,j})\right)
$$

where each $f_m$ is a time series feature extraction function (e.g. mean, variance, trend, autocorrelation), and $M$ is the total number of extracted features.

This results in a 2D design matrix of features for the entire image:

$$
\mathbf{X}_{\text{features}} \in \mathbb{R}^{H \times W \times M}
$$

This transformation effectively reduces the temporal complexity while preserving informative temporal patterns, enabling efficient training of spatial models or further aggregation to coarser spatial units (e.g., fields or regions).

If additional static features are available per pixel (e.g., soil type, elevation), these can be concatenated:

$$
\vec{x}_{i,j}^\text{final} = \left[ \vec{x}_{i,j} \,|\, \vec{a}_{i,j} \right] \in \mathbb{R}^{M + U}
$$

where $\vec{a}_{i,j} \in \mathbb{R}^U$ represents the $U$ univariate attributes at pixel $(i, j)$.


### Feature Set

The table below summarizes the suite of time series features extracted by the `xr_fresh` module from satellite imagery. These features are designed to characterize the temporal behavior of each pixel $(x_i, y_j)$, capturing key aspects of crop phenology and seasonal dynamics. By including a diverse set of statistical, trend, and distribution-based metrics, `xr_fresh` enables detailed and scalable analysis of temporal patterns relevant to remote sensing applications such as crop classification and environmental monitoring.

\renewcommand{\arraystretch}{1.5}  
\begin{longtable}{|p{4cm}|p{5cm}|p{6cm}|}
\hline
\textbf{Statistic} & \textbf{Description} & \textbf{Equation} \\
\hline
\endhead
Absolute energy &  sum over the squared values & $E = \sum_{i=1}^n x_i^2$ \\
Absolute Sum of Changes  & sum over the absolute value of consecutive changes in the series  & $ \sum_{i=1}^{n-1} \mid x_{i+1}- x_i \mid $ \\
Autocorrelation (1 \& 2 month lag) & Correlation between the time series and its lagged values & $\frac{1}{(n-l)\sigma^{2}} \sum_{t=1}^{n-l}(X_{t}-\mu )(X_{t+l}-\mu)$\\
Count Above Mean & Number of values above the mean & $N_{\text{above}} = \sum_{i=1}^n (x_i > \bar{x})$ \\
Count Below Mean & Number of values below the mean & $N_{\text{below}} = \sum_{i=1}^n (x_i < \bar{x})$ \\Day of Year of Maximum Value & Day of the year when the maximum value occurs in series & --- \\
Day of Year of Minimum Value & Day of the year when the minimum value occurs in series & --- \\
Kurtosis & Measure of the tailedness of the time series distribution & $G_2 = \frac{\mu_4}{\sigma^4} - 3$ \\
Linear Time Trend & Linear trend coefficient estimated over the entire time series & $b = \frac{\sum_{i=1}^n (x_i - \bar{x})(t_i - \bar{t})}{\sum_{i=1}^n (x_i - \bar{x})^2}$ \\
Longest Strike Above Mean & Longest consecutive sequence of values above the mean & --- \\
Longest Strike Below Mean & Longest consecutive sequence of values below the mean & --- \\
Maximum & Maximum value of the time series & $x_{\text{max}}$ \\
Mean & Mean value of the time series & $\bar{x} = \frac{1}{n}\sum_{i=1}^n x_i$ \\
Mean Absolute Change & Mean of absolute differences between consecutive values & $\frac{1}{n-1} \sum_{i=1}^{n-1} | x_{i+1} - x_{i}|$ \\
Mean Change & Mean of the differences between consecutive values & $ \frac{1}{n-1} \sum_{i=1}^{n-1}  x_{i+1} - x_{i} $ \\
Mean Second Derivative Central & measure of acceleration of changes in a time series data & $\frac{1}{2(n-2)} \sum_{i=1}^{n-1}  \frac{1}{2} (x_{i+2} - 2 \cdot x_{i+1} + x_i)
$ \\
Median & Median value of the time series & $\tilde{x}$ \\
Minimum & Minimum value of the time series & $x_{\text{min}}$ \\
Quantile (q = 0.05, 0.95) & Values representing the specified quantiles (5th and 95th percentiles) & $Q_{0.05}, Q_{0.95}$ \\
Ratio Beyond r Sigma (r=1,2,3) & Proportion of values beyond r standard deviations from the mean & $P_r = \frac{1}{n}\sum_{i=1}^{n} (|x_i - \bar{x}| > r\sigma_{x})$ \\
Skewness & Measure of the asymmetry of the time series distribution & $\frac{n}{(n-1)(n-2)} \sum \left(\frac{X_i - \overline{X}}{s}\right)^3$ \\
Standard Deviation & Standard deviation of the time series & $  \sqrt{\frac{1}{N}\sum_{i=1}^{n} (x_i - \bar{x})^2}$ \\
Sum Values & Sum of all values in the time series & $S = \sum_{i=1}^{n} x_i$ \\
Symmetry Looking & Measures the similarity of the time series when flipped horizontally & $| x_{\text{mean}}-x_{\text{median}} | < r * (x_{\text{max}} - x_{\text{min}} ) $ \\
Time Series Complexity (CID CE) & measure of number of peaks and valleys & $\sqrt{ \sum_{i=1}^{n-1} ( x_{i} - x_{i-1})^2 }$\\
Variance & Variance of the time series & $\sigma^2 = \frac{1}{N}\sum_{i=1}^{n} (x_i - \bar{x})^2$ \\
Variance Larger than Standard Deviation & check if variance is larger than standard deviation & $\sigma^2 > 1$ \\
\hline
\end{longtable}
 
### Addtional Features  

Two common challenges in remote sensing time series data are the presence of missing values and the high dimensionality of the data. The `xr_fresh` library addresses these issues through advanced interpolation techniques and dimensionality reduction methods.

#### Interpolation

The `xr_fresh` library also includes functionality for interpolating missing values in the time series data. This is crucial for ensuring that the feature extraction process is not hindered by gaps in the data, which are common in remote sensing applications due to cloud cover or sensor errors. The interpolation methods implemented in `xr_fresh` are designed to be computationally efficient and can handle large datasets effectively.

Time series from remote sensing data often contain missing observations due to cloud cover or sensor gaps. The module supports advanced interpolation techniques including linear, nearest-neighbor, cubic spline, and univariate spline interpolation. These methods can utilize either regular intervals or explicitly provided date vectors to guide the interpolation along the temporal (z) dimension. Interpolation is applied pixel-wise to reconstruct continuous temporal profiles before feature extraction.

Formally, for a fixed pixel $(i, j)$, let the time series be:

$$
X_{i,j} = (x_{i,j,1}, x_{i,j,2}, \ldots, x_{i,j,T})
$$

where some $x_{i,j,t}$ may be missing due to clouds or sensor gaps. Interpolation estimates these missing values by fitting a function $f(t)$ to the observed time steps $\{t_k \in [1, T] \mid x_{i,j,t_k} \text{ is observed} \}$. The interpolated value at time $t$ is:

$$
\hat{x}_{i,j,t} = f(t), \quad \text{for } x_{i,j,t} \text{ missing}
$$

The function $f(t)$ may take the form of:

* **Linear interpolation:** $f(t) = a t + b$
* **Nearest neighbor:** $f(t) = x_{i,j,t_k}$ where $t_k = \arg\min_{t_k} |t - t_k|$
* **Cubic spline interpolation:** smooth piecewise cubic polynomials with continuity up to the second derivative
* **Univariate spline interpolation:** minimizes the penalized error

$$
\sum_k (x_{i,j,t_k} - f(t_k))^2 + \lambda \int (f''(t))^2 dt
$$

If acquisition times are irregular, time $t$ is replaced by a continuous index (e.g., days since first observation).


#### Dimensionality Reduction

For high-dimensional inputs or when the number of bands/time steps is large, dimensionality reduction can improve model interpretability and performance. xr_fresh integrates a GPU/CPU-parallelized Kernel Principal Component Analysis (KPCA) module using a radial basis function (RBF) kernel. The KPCA implementation samples valid observations for training, fits the kernel model, and projects each pixel’s time series into a lower-dimensional space. The transformation is parallelized across spatial blocks using Ray and compiled with Numba for fast evaluation.



## Software Framework

### Software Architecture

xr_fresh utilizes xarray data structures for efficient multidimensional array processing and integrates Dask for parallel computation across large datasets. The feature extraction algorithms are parallelized, significantly reducing computational overhead. The library seamlessly integrates with existing Python geospatial and machine learning libraries such as scikit-learn, enabling straightforward adoption into diverse workflows.

### Software Functionalities

The following table provides a comprehensive list of the time series features extracted from the satellite imagery using the xr_fresh module. These features capture the temporal dynamics of crop growth and development, providing valuable information on the phenological patterns of different crops. The computed metrics encompass a wide range of statistical measures, changes over time, and distribution-based metrics, offering a detailed analysis of the temporal patterns in the study area.