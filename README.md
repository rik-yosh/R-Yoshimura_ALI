# R-Yoshimura_ALI
This code is an analysis code for the paper "Stratifying and predicting progression to acute liver failure during the early phase of acute liver injury"

## Desciption
Mathematical models describing the PT%　dynamics are stored in monolix files.
See (https://lixoft.com/products/monolix/) for parameter estimation.

The complete dataset used in this study is available from "Data/miceData/df_afterMice_1".

## monolix
'monolix/timeSeriesPT.csv' is the time-series PT% of the patients with ALI/ALF.
'monolix/PT_NLMEMfit_Model.txt' is the description for the mathematical model of PT% dynamics.

'Virus_dynamics/simplemodel.txt' is the description for the mathematical model of virus dynamics.

These files are used to estimate the lesion and virus dynamics model parameters in MONOLIX2021R2.

## Data availability
The data used for the analysis is in the “Data” folder. "rawData.csv" is the actual raw data.

## Code availability
The code used for the analysis is in the “Code” folder.

