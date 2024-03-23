# FCM-based Maturity Model for Industry 4.0

## Getting Started

- Create a new [conda](https://docs.anaconda.com/free/miniconda/) environment:
    ```bash
    conda create -n pyfcm python=3.10
    conda activate pyfcm
    ```

- Install the dependencies:
    ```bash
    pip install -r requirements.py
    ```

## Run the code

- Define the activation levels (AL) of each technology (node) inside the [cases](cases) folder - [follow below instructions](#define-al-for-a-case). Or use one of the cases already available `[low, medium, high, mix]`.

- To run static analyses of the FCM you need to pass the `<case_name>` as argument:
    ```shell
    python SA_class.py <case_name>
    ```

- To run the FCM (inference analysis) you need to pass the `<case_name>` as argument:
    ```shell
    python FCM_class.py <case_name>
    ```

- To run what-if analysis (genetic algorithm) you need to pass the `<target_value>` you want to reach and the `<case_name>` as arguments:
    ```shell
    python GA_class.py <target_value> <case_name>
    ```

### Define AL for a new case

- Create a new folder with the name you want in [cases](cases).
- Create a new file `0_al.csv` for the AL of the main FCM with the following data.
    ```csv
    NA,1
    NA,2
    NA,3
    NA,4
    NA,5
    NA,0
    ```
- Create a new file `X_al.csv` for `X=[1,5]` to define the AL of each technology. Linguistic terms used available are `[NA, VL, L, M, H, VH] = [neutral, very low, low, medium, high, very high]`. The concept linked to the main FCM, the one in $(row=0,column=0)$ must have `NA` value. Below an example:
    ```csv
    NA,0
    L,0
    M,0
    H,0
    L,0
    L,0
    M,0
    ```
    To check the technologies linked to each node, have a look at the `.json` files in [model](model).