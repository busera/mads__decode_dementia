# Decode Dementia Project

- [Decode Dementia Project](#decode-dementia-project)
  - [1. Project Overview](#1-project-overview)
    - [1.1. Project Team](#11-project-team)
    - [1.2. Introduction and Objectives](#12-introduction-and-objectives)
  - [2. Dataset](#2-dataset)
    - [2.1. Data Description and Importance](#21-data-description-and-importance)
    - [2.2. Data Access and Usage Statement](#22-data-access-and-usage-statement)
  - [3. Project Structure](#3-project-structure)
  - [4. Workflow](#4-workflow)
  - [5. Installation and Setup](#5-installation-and-setup)
    - [5.1. Environment Setup](#51-environment-setup)
    - [5.2. Add Data](#52-add-data)
  - [6. References](#6-references)

## 1. Project Overview

### 1.1. Project Team

**UMICH MADS Capstone Team**:

- **Andre Buser** (busera)
- **Victor Adafinoaiei** (vadafino)

### 1.2. Introduction and Objectives

**Introduction.** Traumatic brain injuries (TBI) present a significant public health issue due to their immediate and enduring consequences on individuals, notably heightening the risk of neurodegenerative diseases like dementia. The intricate nature of TBIs, coupled with their diverse long-term impacts, especially in relation to dementia, remains not fully understood. This gap in knowledge constrains our capacity to devise and implement comprehensive preventative and treatment approaches.

**Objectives.** Our project was designed to enhance our understanding of the outcomes following traumatic brain injuries, focusing on their determinants and consequences. We used causal inference to investigate the complex relationships between TBI and the subsequent risk of dementia, considering a range of demographic and clinical factors. Our research was divided into two main steps:

- **Step 1:** Demographic and TBI-Related Data Analysis
In this phase, our goals were to (a) determine the association between TBI and the development of dementia and (b) examine and assess how age, gender, education level, APOE e4 gene status, and CERAD scores affected the risk of dementia after TBI.cWe employed both adjusted and unadjusted models, with and without applying group weights, to improve the accuracy of our findings.

  To ensure the success and relevance of our analysis in this phase, we focused on the following evaluation and success criteria:

  - Achieving Statistical Significance: Our goal was to obtain statistically significant correlations between TBI and dementia development. We aimed for p-values below an established threshold and confidence intervals that robustly demonstrated a significant association, confirming the validity of our findings.
  - Determining Substantial Effect Size: We sought to identify a meaningful effect size in the relationship between TBI and dementia onset. Success in this aspect meant quantifying an effect size that was not only statistically significant but also clinically meaningful, thereby underscoring the practical implications of TBI as a significant risk factor for dementia.
  - Ensuring Model Robustness: Our aim was to develop and validate robust statistical models that accurately represented the data and reliably predicted or explained the outcomes. This involved ensuring a good fit for the models, using appropriate statistical methods, and confirming that all underlying assumptions were met. Success here meant that our models would provide reliable and valid insights into the data, suitable for guiding further research or clinical applications.

- **Step 2:** Protein and Pathology Quantification Data Analysis and Clustering. In this phase, we conducted a comprehensive analysis of the protein and pathology quantification data to identify patterns and enhance our insights from Step 1. Our key steps and success criteria included:
  - **Statistical Significance of Clusters & High Silhouette Scores:** We aimed for statistically significant clusters, ensuring that the clustering method effectively grouped similar data points. Success was measured by achieving high Silhouette Scores, indicating well-matched data points within clusters and confirming the quality of cluster formation.
  - **Identification of Significant Differences:** Success involved identifying significant differences in protein expression across clusters using ANOVA tests, while considering TBI and dementia status in the analysis.
  - **Intuitive Cluster Representation:** We visually represented clusters using scatter plots and correlation heatmaps, enabling intuitive exploration of patterns and relationships within the data.
  - **Relevant Feature Identification:** We determined the top contributing features within each cluster, excluding PCA components, to gain insights into the key factors driving cluster formation.
  - **Statistical Significance in Protein Expression:** Success was achieved by revealing statistically significant differences in protein expression across clusters using ANOVA tests.

## 2. Dataset

Our project utilized a comprehensive dataset derived from the [Aging, Dementia, and Traumatic Brain Injury Study](http://aging.brain-map.org/overview/home) (Allen Institute for Brain Science, 2017). This dataset was accessible through the official portal of the ADTBI Study, encompassing a range of data files that provided insights into donor demographics, protein and pathology quantifications, and specific group weights for our analyses. Below, we detail the specific files used and their relevance to our project.

### 2.1. Data Description and Importance

For the project, we utilized the following data (files) from the study:

| Description and Importance | Records | Attributes |
| -------------------------- | :----: | :-----: |
| **DonorInformation.csv**<br>contains detailed information about individual donors, including various age-related characteristics.<br><br>This file allowed us to understand the baseline characteristics of our study population, providing a foundation for any associations or patterns we might observe in relation to TBI and dementia. | 107 | 19 |
| **ProteinAndPathologyQuantifications.csv**<br>offers quantified measurements related to proteins and pathologies associated with brain aging or neurodegenerative disorders.<br><br>The data from this file were instrumental in correlating molecular and pathological changes with clinical outcomes, helping us to uncover potential biomarkers or pathological processes associated with dementia post-TBI. | 377 | 33 |
| **group_weights.csv**<br>contains subject and sampling weights as calculated and described in the Allen Institute's 'Technical White Paper: Weighted Analyses' (2016). These weights are crucial for adjusting our analyses to be representative of the larger ACT cohort, thereby enhancing the validity of our findings. The Allen Institute (2016) provides detailed methodologies for the calculation of subject and sampling weights.<br><br>In the TBI study, participants with a history of traumatic brain injury (TBI) and loss of consciousness (LOC) were selected from the larger Adult Changes in Thought (ACT) cohort. To mitigate selection bias due to differences in mortality and autopsy consent between the autopsy-based sample and the entire ACT cohort, group weights were applied in the analyses. This methodology is a key strength of the study, as it addresses potential selection biases and ensures that the findings concerning the association between TBI and dementia risk, as well as the impact of other variables, are representative of the larger ACT cohort. By incorporating group weights, the study enhances the validity and generalizability of its results, providing a more accurate reflection of the broader population affected by TBI. | 107 | 2 |

### 2.2. Data Access and Usage Statement

In our project, we utilize data sourced from the Allen Institute. We adhere to the following guidelines, in line with the Allen Institute's Terms of Use:

- **Purpose of Use:** The data obtained from the Allen Institute is strictly for research and educational purposes. We ensure that all usage is noncommercial, aligning with the Allen Institute's policy.
- **Innovation and Improvements:** Our use of the Allen Institute's data respects their 'Freedom to Innovate' clause. Any enhancements or derivative works developed from this data are for academic advancement and do not infringe upon the Allen Institute's rights to innovate independently.
- **Citation and Acknowledgment:** Consistent with the Allen Institute's Citation Policy, we properly attribute the source of the data in all our publications and presentations. This acknowledgment is in line with standard academic and industry practices.
- **Restriction on Redistribution:** We understand that the redistribution of the Allen Institute's data or any derivative works for commercial purposes is not permitted without explicit written consent from the Allen Institute.
- **Intellectual Property Compliance:** We acknowledge that the Allen Institute holds the rights to the data. Our project's use of this data does not transfer any ownership, rights, title, or interest in the data to us.

## 3. Project Structure

```bash
PROJECT FOLDER
├── README.md
├── data
│   └── raw
│   └── interim
│   └── processed
├── docs
├── environment.yaml
├── logs
├── notebooks
├── poetry.lock
├── pyproject.toml
├── reports
│   ├── figures
│   └── html
```

- **README.md**: Project documentation.
- **data**: Data storage.
  - **interim**: Intermediate data storage.
  - **processed**: Processed data ready for modeling.
  - **raw**: Raw, unprocessed data.
- **docs**: Directory for the ACT study documentation.
- **environment.yaml**: YAML file that specifies the full development environment for the project, including dependencies and packages.
- **notebooks**: Directory for Jupyter notebooks used in the project.

## 4. Workflow

Our workflow for developing the final models, which were used for this project, is depicted in the figure below:

![Workflow](reports/figures/decode_dementia_workflow.jpg)


## 5. Installation and Setup

### 5.1. Environment Setup

We recommend using `conda` as the foundation because it simplifies the management of required Python versions. However, you can also consider `pyvenv` as an alternative. To create the project's conda environment use:

```bash
conda env create -f environment.yaml
```

Once the environment is set up, activate it and proceed to install the packages with **Poetry**:

```bash
poetry install
```

### 5.2. Add Data

The data utilized for this project can also be found in the GitHub repository, specifically in the "data" folder.

However, you can also download the data directly from the Allen Institute for Brain Science website.

Download all files from http://aging.brain-map.org/download/index, extract the zip-files and transfer all files to the data/raw folder (without any subfolders). Delete the zip-files.

The directory should appear as follows:

```bash
data
└── raw
    ├── DescriptionOfStains.csv
    ├── DonorInformation.csv
    ├── ProteinAndPathologyQuantifications.csv
    ├── README.txt
    ├── columns-samples.csv
    ├── fpkm_table_normalized.csv
    ├── fpkm_table_unnormalized.csv
    ├── group_weights.csv
    ├── rows-genes.csv
    ├── rsem_GRCh38.p2.gtf
    └── tbi_data_files.csv
```

## 6. References

- Allen Institute for Brain Science. (2017). Technical White Paper: Overview of the Aging, Dementia and Traumatic Brain Injury (TBI) Project. http://aging.brain-map.org/overview/home
