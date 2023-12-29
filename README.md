# FCM-based Maturity Model for Industry 4.0

## Preliminaries
We assume the user has **Python 3.10** installed

1. Download the repository
    ```shell
    git clone https://github.com/iaiamomo/FCM_maturity_model.git
    ```

- Install the Python packages
    ```shell
    pip install -r requirements.txt
    ```

## Run the code

- Define the activation levels (AL) of each technologies (nodes) inside the [models](models) folder. [Model](#model) describe how to define the ALs.

    **NOTE:** Please use [model5](models/model5/).

- Define the characteristics of the defined FCM into the configuration file [config.json](config.json)

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

0. Define the main $(index = 0)$ FCM.

1. Define the $n \times n$ weight matrix in `0_wm.csv`.
    - Rows and columns represent the $n$ concepts
    - Element in row $x$ and column $y$ represents the weight of the influence of concept $x$ over $y$.
    - Concept in rown and column $n-1$ represent the main concept.
    - Example:
    ```csv
    0,0,0,0,0,1
    0,0,0,0,0,1
    0,0,0,0,0,1
    0,0,0,0,0,1
    0,0,0,0,0,1
    0,0,0,0,0,0
    ```

2. Define the $n \times 2$ activation level matrix in `0_al.csv`.
    - Rows represent the $n$ concepts.
    - Columns represent:
        - Activation level.
        - Index of the sub-graph to which the concept in the row is linked to.
    - Example:
    ```csv
    0,1
    0,2
    0,3
    0,4
    0,5
    0,0
    ```

3. For each index $i\neq0$ defined in the main activation level matrix, define the weight and activation level matrices of the sub-graph $i$.
    - The weight matrix is defined in `<index>_wm.csv`.
    - The activation level matrix is defined in `<index>_al_<user>.csv`, where `<user>` is a particular user expressing its own activation level of the concepts.
    - The concept that is linked to the main FCM is in $row=0$ and $column=0$.
    - Weight matrix example of `index=1`:
    ```csv
    0,0,0,0,0,0,0
    1,0,0.5,0,0.1,0.5,0.5
    1,0,0,0,0.1,0.5,0.5
    1,1,0.5,0,0,0.1,0.5
    1,0,0,0,0,1,0
    1,0,0.5,0.1,1,0,0.5
    1,0,0.5,0.5,0,0.5,0
    ```
    - Activation level matrix example of `index=1`:
    ```csv
    0,0
    0.5,0
    0.9,0
    0.9,0
    0.5,0
    0.5,0
    0.9,0
    ```


4. For each FCM define a `<index>_desc.json` file listing the concepts related to that particular FCM. For instance main FCM of our maturity model is:
    ```json
    {
        "main": "Industry 4.0",
        "nodes": {
            "1": "CAD, CAM, PLM",
            "2": "CRM",
            "3": "ERP, SCM",
            "4": "WMS, TMS",
            "5": "MES",
            "6": "Industry 4.0"
        }
    }
    ```
    An instance of the describing file of a sub-graph is:
    ```json
    {
        "main": "CAD, CAM, PLM",
        "nodes": {
            "1": "CAD, CAM, PLM",
            "2": "Cloud Services",
            "3": "AI, CV, PM, RPA",
            "4": "Cyber Security",
            "5": "AR, VR",
            "6": "Digital Twin",
            "7": "BI, Big Data"
        }
    }
    ```
