# AI-Hack-18-gwas

Submission for AI Hack 2018's [GWAS Challenge](http://aihack.org/the-hackathon/), organised by Imperial College Data Science Society and Math Soc. 

## 1. Challenge data 
Study individuals with genotype information across 861,473 single nucleotide polymorphisms (SNPs), surveyed between 5 years, where the case-control study was based on European ancestry severe angiographic cardiovascular disease (CAD) status. Datasets in the form of .csv files for:
- SNP matrices for all 22 chromosomes and patients 
- Clinical matrix for all patients 

![Imgur](https://i.imgur.com/OXJMRAm.png)

## 2. Task 
Determining the SNPs that are likely to be significant to changes in measures such as triglycerides, low-density lipoprotein (LDL) cholesterol and high-density lipoprotein (HDL) cholesterol, and which SNPs seem to be linked to the cardiovascular diseases the most. In particular, use linear and logistic regressions.

## 3. Brief outline of methodology
We looked at Chromosome 15 specifically. Our method outline is as follows:
1. Data pre-processing and flagging of SNPs with large amount of missing data.
2. PCA of the SNP data to obtain first 10 PCs. 
3. Logistic regression with the CAD, LDL, HDL clinical data to determine SNP p-values.
4. Manhattan plots. 

## 4. Submission files
1. SNP_pca.ipynb: Jupyter notebook containing code for PCA procedure and some interactive Manhattan plots 
2. SNP_regression.py: Loading of .csv data files, Logistic regression and Manhattan plots. Code here uses provided PCs by AI Hack but can also be used with our own calculated PCs. 
3. Team25_report.pdf: Submission report for the GWAS challenge (not proofread).
4. AI_Hack_slides.pptx: Demo slides. 

<br>

Interactive Manhattan plot
![Imgur](https://i.imgur.com/5bwaJYV.png)
