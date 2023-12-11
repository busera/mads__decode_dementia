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
  - [6. Conclusion](#6-conclusion)
  - [7. References](#7-references)

## 1. Project Overview

### 1.1. Project Team

**UMICH MADS Capstone Team**:

- **Andre Buser** (busera)
- **Victor Adafinoaiei** (vadafino)

### 1.2. Introduction and Objectives

Traumatic brain injuries (TBI) are a serious health issue that affects millions of people each year. These injuries can range from a mild bump on the head to severe trauma, and they can lead to long-term problems, including the risk of developing dementia – a condition that impairs memory and thinking skills. As our population ages, understanding the connection between TBI and dementia is becoming increasingly important.

The brain is a delicate and complex organ, and how it reacts to injuries is not fully clear. After a TBI, there can be immediate damage as well as a series of changes in the brain that may contribute to diseases like dementia. Studies have shown that people who have had a TBI may be more likely to develop dementia, but we don’t yet know exactly why this is the case. 

Our project was designed to enhance our understanding of the outcomes following traumatic brain injuries, focusing on their determinants and consequences. We used causal inference to investigate the complex relationships between TBI and the subsequent risk of dementia, considering a range of demographic and clinical factors.

Our project is split into two main parts:

**Step 1: Looking at People’s Backgrounds and TBI Details**
First, we’re studying how TBI is linked to dementia and how factors like age, gender, education, genetics, and brain health scores play a role. We’ll know we’re on the right track if we:
- Find clear evidence that TBI and dementia are connected.
- Measure how strong this connection is and show that it’s meaningful.

**Step 2: Studying Proteins and Brain Changes**
Next, we’re examining the proteins in the brain and other changes to find patterns that could explain the link between TBI and dementia. We’ll consider this part successful if we:
- Group the data in a way that makes sense and is backed up by the numbers.
- Show clear differences in protein levels between these groups.
- Present our findings in easy-to-understand format

## 2. Dataset

Our analysis incorporated a subset of the dataset from the Aging, Dementia, and Traumatic Brain Injury Study, a collaborative effort spearheaded by the University of Washington, Kaiser Permanente Washington Health Research Institute, and the Allen Institute for Brain Science.”

The dataset originates from a unique aged cohort from the Adult Changes in Thought (ACT) study, a longitudinal investigation into brain aging and dementia within the Seattle metropolitan area. The ACT study is managed by the Kaiser Permanente Washington Health Research Institute and has established protocols for sharing its data with external researchers.

### 2.1. Data Description and Importance

We focused on three key data files from the [Aging, Dementia, and Traumatic Brain Injury Study](http://aging.brain-map.org/overview/home) (Allen Institute for Brain Science, 2017):

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

Once the environment is set up, activate it:

```bash
conda activate dementia
```

Proceed to install the packages with **Poetry**:

```bash
poetry install
```

### 5.2. Add Data

The data utilized for this project can be found in the GitHub repository, specifically in the "data" folder.

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

## 6. Conclusion

In our study, we established a substantial link between traumatic brain injuries (TBI) and an elevated risk of dementia. Key findings indicate that TBIs, especially those involving loss of consciousness, alongside the presence of the APOE ε4 allele and male gender, significantly increase dementia risk, while higher education levels seem to offer some protective effects. 

Utilizing ANOVA, we identified five critical brain biomarkers related to protein patterns, with particular emphasis on the Aβ42/Aβ40 ratio and tau protein levels. These findings underscore the complex impact of TBI on cognitive health, supported by strong statistical evidence.

While our study provides extensive insights, it is not without limitations. The potential presence of unseen confounding variables and the reliance on retrospective data suggest a need for further research. Future studies should aim to include longitudinal analyses to better understand the connections between TBI and dementia and to explore the underlying mechanisms.

The implications of our findings could be far-reaching for public health policies, clinical practices, and individual decision-making. They emphasize the need for improved TBI prevention strategies, heightened awareness of its long-term risks, and the adoption of personalized medical approaches that take genetic and demographic factors into account. Additionally, our study calls for the establishment of ethical guidelines in research and application to prevent discrimination.
In conclusion, this study not only contributes to the growing body of knowledge on TBI and dementia but also opens new avenues for research and policy-making that can profoundly impact public health and individual well-being.

The complete report can be accessed here: [Decode Dementia Report](https://github.com/busera/MADS__SIADS699__Decode_Dementia/blob/main/docs/Decoding%20Dementia%20-%20Report.pdf)

## 7. References

- Allen Institute for Brain Science. (2017). Technical White Paper: Overview of the Aging, Dementia and Traumatic Brain Injury (TBI) Project. http://aging.brain-map.org/overview/home
