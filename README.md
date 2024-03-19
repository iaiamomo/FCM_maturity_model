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

- Define the activation levels (AL) of each technologies (nodes) inside the [cases](cases) folder.

- Run static analyses of the FCM
    ```shell
    python SA_class.py
    ```

- Run the FCM
    ```shell
    python FCM_class.py
    ```

- Run what-if analyses
    ```shell
    python GA_class.py
    ```

## Model

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
- Create a new file `X_al.csv` for `X=[1,5]` to define the AL of each technology. The concept linked to the main FCM is in $row=0$ and $column=0$.
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
