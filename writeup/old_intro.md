
Recent advancements in satellite technology and sensor data have significantly enhanced our capability to monitor environmental phenomena through time series analysis of images. Efficiently extracting relevant time series features at scale remains challenging, necessitating automated and robust methods. Inspired by `tsfresh` (Time Series FeatuRe Extraction on basis of Scalable Hypothesis tests), we introduce `xr_fresh`, tailored specifically for image time series,by automating the extraction of time series features on a pixel-by-pixel basis [@CHRIST201872].

Time series data, characterized by sequences of measurements indexed over time, are ubiquitous in diverse fields, including finance, healthcare, industrial monitoring, and environmental sciences [@faouzi2022time]. Analyzing and modeling these temporal sequences for tasks such as classification (assigning a time series to a predefined category) or regression (predicting a continuous value associated with a time series) is a fundamental problem in machine learning [@faouzi2022time]. However, applying standard machine learning algorithms directly to raw time series data is often challenging due to the inherent temporal structure and potentially high dimensionality of the sequences [@faouzi2022time]. An effective representation of the time series characteristics is crucial for building accurate and robust models [@mumuni2024automated;@faouzi2022time].

Traditional approaches to time series analysis and machine learning often rely on manual feature engineering, a process in which domain experts handcraft relevant features from raw data [@li2020forecasting]. Although effective in specific contexts, this process is labor intensive, requires significant expertise, and may not generalize well across different datasets or tasks [@li2020forecasting;@faouzi2022time]. The increasing volume and complexity of modern time series data require more automated and scalable approaches to data processing and feature extraction  [@mumuni2024automated;@li2020forecasting].

A prominent tool in this domain is `tsfresh` [@CHRIST201872], a Python package specifically designed for time series feature extraction based on scalable hypothesis tests   [jin2022automated;@mumuni2024automated;@gilpin2021chaos;@schroeder2019chemiresistive;@sylligardos2023choose;@mcdermott2023event;@petropoulos2022forecasting;@zhao2022ai]. `tsfresh` automates the process of calculating a diverse and large number of features from time series data. These features are derived from existing research and include distribution properties (e.g., mean, standard deviation, kurtosis), measures of autocorrelation (e.g., fast Fourier transform (FFT) and power spectral density coefficients), properties of linear regressors (e.g., gradient, standard error), energy, entropy, and stochasticity [@CHRIST201872]. After feature extraction, `tsfresh` also offers automated feature selection methods, such as using statistical tests like the Mann-Whitney U test, to identify and select the most relevant features for a specific task [@CHRIST201872].

Gridded time series data from satellites, climate models, camera feeds, and sensors contain rich temporal information for applications like crop type classification and yields, anomaly detection, robotics, quality control, environmental monitoring, and natural resource management [@delince2017handbook;@mumuni2024automated;@hufkens2019monitoring;@MANN201760;@mann2019predicting]. Traditional methods relied heavily on manually engineered features, which are time-consuming, require domain expertise, and often lack scalability.

Deep learning (DL) has largely supplanted traditional machine learning (ML) methods in a variety of fields, including the analysis of gridded remote sensing data [@LI2023103345;@MA2019166]. This shift is attributed to the capabilities of DL algorithms in handling the massive volume and complexity of remote sensing big data (RSBD) [@LI2023103345]. Traditional methods have largely relied on hand-crafted feature descriptors which are effective only in limited situations and struggle to balance generalization and robustness when dealing with RSBD [@delince2017handbook]. In contrast, deep learning models automatically extract rich hierarchical features directly from the data, learning weights from vast amounts of information. This ability to learn deep representative features is particularly effective for multi-temporal remote sensing data analysis, such as crop type mapping. Although traditional ML methods such as Random Forest, Decision Trees, and Support Vector Machines have been successfully applied to classify different crop types, they typically rely on shallow and hand-crafted features and struggle to capture the complex temporal patterns of crop growth [@zhang2024improving]. Compared to traditional ML models, deep learning models often have a much larger number of trainable parameters and, when trained with sufficient data or augmented techniques, can learn more comprehensive and generalized features, sometimes achieving comparable or superior accuracy even with fewer training samples [@zhang2024improving]. The rapid development of computing power and the availability of big data have enabled deep learning to surpass traditional methods in various remote sensing tasks, including scene classification, object detection, semantic segmentation, change detection, and land use and land cover (LULC) classification [@LI2023103345;@MA2019166;@LI2023103345]. Although traditional ML is still used, the remote sensing community has notably shifted its focus to deep learning since around 2014 due to its significant success in many image analysis tasks [@MA2019166].

Traditional machine learning approaches are well suited for analyzing gridded time series data from a variety of sources, including remote sensing platforms, in ssitu sensors, and climate reanalysis products. These methods can effectively leverage spatial and temporal patterns within the data, capturing key trends and anomalies across environmental variables  [@begue2018remote;@delince2017handbook]. Compared to deep learning architectures, traditional machine learning offers greater interpretability and lower computational overhead [@hohl2024recent]. In a recent study, `xr_fresh` ML methods outperformed a variety of deep learning methods for crop classification using monthly Sentinel-2 images [@venkatachalam2024temporal]. Their transparent decision-making processes can be easily examined by domain experts, communicated to non-technical stakeholders, and they typically require less training data and computational power, making them ideal for many operational or resource-constrained applications, such as on edge devices or research in developing countries [@hohl2024recent;@rs13132591;@LI2023103345;@MA2019166]. Currently, there is no method to rapidly extract a comprehensive set of features from gridded time series data, such as those derived from remote sensing imagery. Existing packages like `tsfresh` are not optimized for the unique characteristics of gridded time series data, which often include irregular sampling intervals, missing values, and high dimensionality. This limitation hinders the ability to efficiently analyze and model these datasets, particularly in the context of remote sensing applications where large volumes of data are generated.

To address this gap, `xr_fresh` automates the extraction of salient temporal and statistical features from each pixel time series. Using automated feature extraction, `xr_fresh` reduces manual intervention and improves reproducibility in remote sensing workflows.




@article{begue2018remote,
  title={Remote sensing and cropping practices: A review},
  author={B{\'e}gu{\'e}, Agn{\`e}s and Arvor, Damien and Bellon, Beatriz and Betbeder, Julie and De Abelleyra, Diego and PD Ferraz, Rodrigo and Lebourgeois, Valentine and Lelong, Camille and Sim{\~o}es, Margareth and R. Ver{\'o}n, Santiago},
  journal={Remote Sensing},
  volume={10},
  number={1},
  pages={99},
  year={2018},
  publisher={MDPI}
}

@article{CHRIST201872,
title = {Time Series FeatuRe Extraction on basis of Scalable Hypothesis tests (tsfresh – A Python package)},
journal = {Neurocomputing},
volume = {307},
pages = {72-77},
year = {2018},
issn = {0925-2312},
doi = {https://doi.org/10.1016/j.neucom.2018.03.067},
url = {https://www.sciencedirect.com/science/article/pii/S0925231218304843},
author = {Maximilian Christ and Nils Braun and Julius Neuffer and Andreas W. Kempa-Liehr},
keywords = {Feature engineering, Time series, Feature extraction, Feature selection, Machine learning},
abstract = {Time series feature engineering is a time-consuming process because scientists and engineers have to consider the multifarious algorithms of signal processing and time series analysis for identifying and extracting meaningful features from time series. The Python package tsfresh (Time Series FeatuRe Extraction on basis of Scalable Hypothesis tests) accelerates this process by combining 63 time series characterization methods, which by default compute a total of 794 time series features, with feature selection on basis automatically configured hypothesis tests. By identifying statistically significant time series characteristics in an early stage of the data science process, tsfresh closes feedback loops with domain experts and fosters the development of domain specific features early on. The package implements standard APIs of time series and machine learning libraries (e.g. pandas and scikit-learn) and is designed for both exploratory analyses as well as straightforward integration into operational data science applications.}
}

@article{delince2017handbook,
  title={Handbook on remote sensing for agricultural statistics},
  author={Delince, J and Lemoine, G and Defourny, P and Gallego, J and Davidson, A and Ray, S and Rojas, O and Latham, J and Achard, F},
  journal={GSARS: Rome, Italy},
  year={2017}
}
@article{faouzi2022time,
  title={Time series classification: A review of algorithms and implementations},
  author={Faouzi, Johann},
  journal={Machine Learning (Emerging Trends and Applications)},
  year={2022},
  publisher={Proud Pen}
}
@article{funk2015climate,
  title={The climate hazards infrared precipitation with stations—a new environmental record for monitoring extremes},
  author={Funk, Chris and Peterson, Pete and Landsfeld, Martin and Pedreros, Diego and Verdin, James and Shukla, Shraddhanand and Husak, Gregory and Rowland, James and Harrison, Laura and Hoell, Andrew and others},
  journal={Scientific data},
  volume={2},
  number={1},
  pages={1--21},
  year={2015},
  publisher={Nature Publishing Group}
}


@article{gilpin2021chaos,
  title={Chaos as an interpretable benchmark for forecasting and data-driven modelling},
  author={Gilpin, William},
  journal={arXiv preprint arXiv:2110.05266},
  year={2021}
}
@inproceedings{hohl2024recent,
  title={Recent Trends Challenges and Limitations of Explainable AI in Remote Sensing},
  author={H{\"o}hl, Adrian and Obadic, Ivica and Fern{\'a}ndez-Torres, Miguel-{\'A}ngel and Oliveira, Dario and Zhu, Xiao Xiang},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={8199--8205},
  year={2024}
}
@article{hufkens2019monitoring,
  title={Monitoring crop phenology using a smartphone based near-surface remote sensing approach},
  author={Hufkens, Koen and Melaas, Eli K and Mann, Michael L and Foster, Timothy and Ceballos, Francisco and Robles, Miguel and Kramer, Berber},
  journal={Agricultural and forest meteorology},
  volume={265},
  pages={327--337},
  year={2019},
  publisher={Elsevier}
}
@misc{jax2018github,
  author = {James Bradbury and Roy Frostig and Peter Hawkins and Matthew James Johnson and Chris Leary and Dougal Maclaurin and George Necula and Adam Paszke and Jake Vander{P}las and Skye Wanderman-{M}ilne and Qiao Zhang},
  title = {{JAX}: composable transformations of {P}ython+{N}um{P}y programs},
  year = {2018},
  version = {0.3.13}, 
  howpublished = {\url{http://github.com/google/jax}}
}
@article{jin2022automated,
  title={Automated dilated spatio-temporal synchronous graph modeling for traffic prediction},
  author={Jin, Guangyin and Li, Fuxian and Zhang, Jinlei and Wang, Mudan and Huang, Jincai},
  journal={IEEE Transactions on Intelligent Transportation Systems},
  volume={24},
  number={8},
  pages={8820--8830},
  year={2022},
  publisher={IEEE}
}
@article{lam2015numba,
  title={Numba: A LLVM-based Python JIT Compiler},
  author={Siu Kwan Lam and Antoine Pitrou and Stanley Seibert},
  journal={Proceedings of the Second Workshop on the LLVM Compiler Infrastructure in HPC},
  year={2015},
  pages={1--6},
  doi={https://doi.org/10.1145/2833157.2833162},
  url={https://numba.pydata.org/}
}

@article{LI2023103345,
title = {Cost-efficient information extraction from massive remote sensing data: When weakly supervised deep learning meets remote sensing big data},
journal = {International Journal of Applied Earth Observation and Geoinformation},
volume = {120},
pages = {103345},
year = {2023},
issn = {1569-8432},
doi = {https://doi.org/10.1016/j.jag.2023.103345},
url = {https://www.sciencedirect.com/science/article/pii/S156984322300167X},
author = {Yansheng Li and Xinwei Li and Yongjun Zhang and Daifeng Peng and Lorenzo Bruzzone},
keywords = {Remote sensing big data mining, Weakly supervised deep learning, Cost-efficient information extraction, Future research directions},
abstract = {With many platforms and sensors continuously observing the earth surface, the large amount of remote sensing data presents a big data challenge. While remote sensing data acquisition capability can fully meet the requirements of many application domains, there is still a need to further explore how to efficiently mine the useful information from remote sensing big data (RSBD). Many researchers in the remote sensing community have introduced deep learning in the process of RSBD, and deep learning-based methods have achieved better performance compared with traditional methods. However, there are still substantial obstacles to the application of deep learning in remote sensing. One of the major challenges is the generation of pixel-level labels with high quality for training samples, which is essential to deep learning models. Weakly supervised deep learning (WSDL) is a promising solution to address this problem as WSDL can utilize greedily labeled datasets that are easy to collect but not ideal to train the deep networks. In this review, we summarize the achievements of WSDL-driven cost-efficient information extraction from RSBD. We first analyze the opportunities and challenges of information extraction from RSBD. Based on the analysis of the theoretical foundations of WSDL in the computer vision (CV) domain, we conduct a survey on the WSDL-based information extraction methods under the data characteristic and task demand of RSBD in four different tasks: (i) scene classification, (ii) object detection, (iii) semantic segmentation and (iv) change detection. Finally, potential research directions are outlined to guide researchers to further exploit WSDL-based information extraction from RSBD.}
}
@article{li2020forecasting,
  title={Forecasting with time series imaging},
  author={Li, Xixi and Kang, Yanfei and Li, Feng},
  journal={Expert Systems with Applications},
  volume={160},
  pages={113680},
  year={2020},
  publisher={Elsevier}
}

@article{MA2019166,
title = {Deep learning in remote sensing applications: A meta-analysis and review},
journal = {ISPRS Journal of Photogrammetry and Remote Sensing},
volume = {152},
pages = {166-177},
year = {2019},
issn = {0924-2716},
doi = {https://doi.org/10.1016/j.isprsjprs.2019.04.015},
url = {https://www.sciencedirect.com/science/article/pii/S0924271619301108},
author = {Lei Ma and Yu Liu and Xueliang Zhang and Yuanxin Ye and Gaofei Yin and Brian Alan Johnson},
keywords = {Deep learning (DL), Remote sensing, LULC classification, Object detection, Scene classification},
abstract = {Deep learning (DL) algorithms have seen a massive rise in popularity for remote-sensing image analysis over the past few years. In this study, the major DL concepts pertinent to remote-sensing are introduced, and more than 200 publications in this field, most of which were published during the last two years, are reviewed and analyzed. Initially, a meta-analysis was conducted to analyze the status of remote sensing DL studies in terms of the study targets, DL model(s) used, image spatial resolution(s), type of study area, and level of classification accuracy achieved. Subsequently, a detailed review is conducted to describe/discuss how DL has been applied for remote sensing image analysis tasks including image fusion, image registration, scene classification, object detection, land use and land cover (LULC) classification, segmentation, and object-based image analysis (OBIA). This review covers nearly every application and technology in the field of remote sensing, ranging from preprocessing to mapping. Finally, a conclusion regarding the current state-of-the art methods, a critical conclusion on open challenges, and directions for future research are presented.}
}
@article{MANN201760,
title = {Ethiopian wheat yield and yield gap estimation: A spatially explicit small area integrated data approach},
journal = {Field Crops Research},
volume = {201},
pages = {60-74},
year = {2017},
issn = {0378-4290},
doi = {https://doi.org/10.1016/j.fcr.2016.10.014},
url = {https://www.sciencedirect.com/science/article/pii/S0378429016305238},
author = {Michael L. Mann and James M. Warner},
keywords = {Ethiopia, Agriculture, Data integration, Wheat productivity, Remote sensing, Smallholder agriculture, Panel data estimation, Yield gaps},
abstract = {Despite the routine collection of annual agricultural surveys and significant advances in GIS and remote sensing products, little econometric research has integrated these data sources in estimating developing nations’ agricultural yields. In this paper, we explore the determinants of wheat output per hectare in Ethiopia during the 2011–2013 principal Meher crop seasons at the kebele administrative area. Using a panel data approach, combining national agricultural field surveys with relevant GIS and remote sensing products, the model explains nearly 40% of the total variation in wheat output per hectare across the country. Reflecting on the high interannual variability in output per hectare, we explore whether these changes can be explained by weather, shocks to, and management of rain-fed agricultural systems. The model identifies specific contributors to wheat yields that include farm management techniques (e.g. area planted, improved seed, fertilizer, and irrigation), weather (e.g. rainfall), water availability (e.g. vegetation and moisture deficit indexes) and policy intervention. Our findings suggest that woredas produce between 9.8 and 86.5% of their locally attainable wheat yields given their altitude, weather conditions, terrain, and plant health. In conclusion, we believe the combination of field surveys with spatial data can be used to identify management priorities for improving production at a variety of administrative levels.}
}
@article{mann2019predicting,
  title={Predicting high-magnitude, low-frequency crop losses using machine learning: an application to cereal crops in Ethiopia},
  author={Mann, Michael L and Warner, James M and Malik, Arun S},
  journal={Climatic change},
  volume={154},
  number={1},
  pages={211--227},
  year={2019},
  publisher={Springer}
}
@article{mcdermott2023event,
  title={Event Stream GPT: a data pre-processing and modeling library for generative, pre-trained transformers over continuous-time sequences of complex events},
  author={McDermott, Matthew and Nestor, Bret and Argaw, Peniel and Kohane, Isaac S},
  journal={Advances in Neural Information Processing Systems},
  volume={36},
  pages={24322--24334},
  year={2023}
}
@article{mumuni2024automated,
  title={Automated data processing and feature engineering for deep learning and big data applications: a survey},
  author={Mumuni, Alhassan and Mumuni, Fuseini},
  journal={Journal of Information and Intelligence},
  year={2024},
  publisher={Elsevier}
}

@article{petropoulos2022forecasting,
  title={Forecasting: theory and practice},
  author={Petropoulos, Fotios and Apiletti, Daniele and Assimakopoulos, Vassilios and Babai, Mohamed Zied and Barrow, Devon K and Taieb, Souhaib Ben and Bergmeir, Christoph and Bessa, Ricardo J and Bijak, Jakub and Boylan, John E and others},
  journal={International Journal of forecasting},
  volume={38},
  number={3},
  pages={705--871},
  year={2022},
  publisher={Elsevier}
}
@article{schroeder2019chemiresistive,
  title={Chemiresistive sensor array and machine learning classification of food},
  author={Schroeder, Vera and Evans, Ethan D and Wu, You-Chi Mason and Voll, Constantin-Christian A and McDonald, Benjamin R and Savagatrup, Suchol and Swager, Timothy M},
  journal={ACS sensors},
  volume={4},
  number={8},
  pages={2101--2108},
  year={2019},
  publisher={ACS Publications}
}
@article{sylligardos2023choose,
  title={Choose wisely: An extensive evaluation of model selection for anomaly detection in time series},
  author={Sylligardos, Emmanouil and Boniol, Paul and Paparrizos, John and Trahanias, Panos and Palpanas, Themis},
  journal={Proceedings of the VLDB Endowment},
  volume={16},
  number={11},
  pages={3418--3432},
  year={2023},
  publisher={VLDB Endowment}
}
@article{scikit-learn,
  title={Scikit-learn: Machine Learning in {P}ython},
  author={Pedregosa, F. and Varoquaux, G. and Gramfort, A. and Michel, V.
          and Thirion, B. and Grisel, O. and Blondel, M. and Prettenhofer, P.
          and Weiss, R. and Dubourg, V. and Vanderplas, J. and Passos, A. and
          Cournapeau, D. and Brucher, M. and Perrot, M. and Duchesnay, E.},
  journal={Journal of Machine Learning Research},
  volume={12},
  pages={2825--2830},
  year={2011}
}
 
@software{geowombat,
  author       = {Graesser, Jordan and
                  Mann, Michael},
  title        = {GeoWombat (v2.1.22): Utilities for geospatial data},
  month        = may,
  year         = 2025,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.15483823},
  url          = {https://doi.org/10.5281/zenodo.15483823},
}
@article{hoyer2017xarray,
  title   = {xarray: {N-D} labeled arrays and datasets in {Python}},
  author  = {Hoyer, S. and J. Hamman},
  journal = {In revision, J. Open Res. Software},
  year    = {2017}
}

@article{virtanen2020scipy,
  title={SciPy 1.0: fundamental algorithms for scientific computing in Python},
  author={Virtanen, Pauli and Gommers, Ralf and Oliphant, Travis E and Haberland, Matt and Reddy, Tyler and Cournapeau, David and Burovski, Evgeni and Peterson, Pearu and Weckesser, Warren and Bright, Jonathan and others},
  journal={Nature methods},
  volume={17},
  number={3},
  pages={261--272},
  year={2020},
  publisher={Nature Publishing Group US New York}
}
@article{harris2020array,
  title={Array programming with NumPy},
  author={Charles R. Harris and K. Jarrod Millman and St{\'e}fan J. van der Walt and Ralf Gommers and Pauli Virtanen and David Cournapeau and Eric Wieser and Julian Taylor and Sebastian Berg and Nathaniel J. Smith and Robert Kern and Matti Picus and Stephan Hoyer and Marten H. van Kerkwijk and Matthew Brett and Allan Haldane and Jaime Fern{\'a}ndez del R{\'i}o and Mark Wiebe and Pearu Peterson and Pierre G{\'e}rard-Marchant and Kevin Sheppard and Tyler Reddy and Warren Weckesser and Hameer Abbasi and Christoph Gohlke and Travis E. Oliphant},
  journal={Nature},
  volume={585},
  pages={357--362},
  year={2020},
  doi={10.1038/s41586-020-2649-2},
  url={https://www.nature.com/articles/s41586-020-2649-2}
}
@inproceedings{hohl2024recent,
  title={Recent trends challenges and limitations of explainable ai in remote sensing},
  author={H{\"o}hl, Adrian and Obadic, Ivica and Fern{\'a}ndez-Torres, Miguel-Angel and Oliveira, Dario and Zhu, Xiao Xiang},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={8199--8205},
  year={2024}
}
@inproceedings{rocklin2015dask,
  title={Dask: Parallel computation with blocked algorithms and task scheduling},
  author={Matthew Rocklin},
  booktitle={Proceedings of the 14th Python in Science Conference},
  pages={130--136},
  year={2015},
  doi={10.25080/Majora-7b98e3ed-013},
  url={https://dask.org/}
}

@article{rs13132591,
AUTHOR = {Maxwell, Aaron E. and Warner, Timothy A. and Guillén, Luis Andrés},
TITLE = {Accuracy Assessment in Convolutional Neural Network-Based Deep Learning Remote Sensing Studies—Part 2: Recommendations and Best Practices},
JOURNAL = {Remote Sensing},
VOLUME = {13},
YEAR = {2021},
NUMBER = {13},
ARTICLE-NUMBER = {2591},
URL = {https://www.mdpi.com/2072-4292/13/13/2591},
ISSN = {2072-4292},
ABSTRACT = {Convolutional neural network (CNN)-based deep learning (DL) has a wide variety of applications in the geospatial and remote sensing (RS) sciences, and consequently has been a focus of many recent studies. However, a review of accuracy assessment methods used in recently published RS DL studies, focusing on scene classification, object detection, semantic segmentation, and instance segmentation, indicates that RS DL papers appear to follow an accuracy assessment approach that diverges from that of traditional RS studies. Papers reporting on RS DL studies have largely abandoned traditional RS accuracy assessment terminology; they rarely reported a complete confusion matrix; and sampling designs and analysis protocols generally did not provide a population-based confusion matrix, in which the table entries are estimates of the probabilities of occurrence of the mapped landscape. These issues indicate the need for the RS community to develop guidance on best practices for accuracy assessment for CNN-based DL thematic mapping and object detection. As a first step in that process, we explore key issues, including the observation that accuracy assessments should not be biased by the CNN-based training and inference processes that rely on image chips. Furthermore, accuracy assessments should be consistent with prior recommendations and standards in the field, should support the estimation of a population confusion matrix, and should allow for assessment of model generalization. This paper draws from our review of the RS DL literature and the rich record of traditional remote sensing accuracy assessment research while considering the unique nature of CNN-based deep learning to propose accuracy assessment best practices that use appropriate sampling methods, training and validation data partitioning, assessment metrics, and reporting standards.},
DOI = {10.3390/rs13132591}
}

 
@inproceedings {moritz2018ray,
author = {Philipp Moritz and Robert Nishihara and Stephanie Wang and Alexey Tumanov and Richard Liaw and Eric Liang and Melih Elibol and Zongheng Yang and William Paul and Michael I. Jordan and Ion Stoica},
title = {Ray: A Distributed Framework for Emerging {AI} Applications},
booktitle = {13th USENIX Symposium on Operating Systems Design and Implementation (OSDI 18)},
year = {2018},
isbn = {978-1-939133-08-3},
address = {Carlsbad, CA},
pages = {561--577},
url = {https://www.usenix.org/conference/osdi18/presentation/moritz},
publisher = {USENIX Association},
month = oct
}
@unpublished{venkatachalam2024temporal,
  title={Temporal Patterns and Pixel Precision: Satellite-Based Crop Classification Using Deep Learning and Machine Learning},
  author={Venkatachalam, Sairam and Kacha, Disha and Sheth, Devarsh and Mann, Michael and Jafari, Amir},
  note={Under review at \textit{IEEE Transactions on Geoscience and Remote Sensing}},
  year={2024},
  institution={George Washington University, Department of Geography \& Environment and Data Science Program}
}
@article{zhao2022ai,
  title={AI-based rainfall prediction model for debris flows},
  author={Zhao, Yan and Meng, Xingmin and Qi, Tianjun and Li, Yajun and Chen, Guan and Yue, Dongxia and Qing, Feng},
  journal={Engineering Geology},
  volume={296},
  pages={106456},
  year={2022},
  publisher={Elsevier}
}
@article{zhang2024improving,
  title={Improving crop type mapping by integrating LSTM with temporal random masking and pixel-set spatial information},
  author={Zhang, Xinyu and Cai, Zhiwen and Hu, Qiong and Yang, Jingya and Wei, Haodong and You, Liangzhi and Xu, Baodong},
  journal={ISPRS Journal of Photogrammetry and Remote Sensing},
  volume={218},
  pages={87--101},
  year={2024},
  publisher={Elsevier}
}