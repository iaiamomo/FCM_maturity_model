import pandas as pd

systems = ['cad_cam_plm', 'crm', 'erp_scm', 'mes', 'wms_tms']
n_experts = {
    'cad_cam_plm': 6,
    'crm': 4,
    'erp_scm': 5,
    'mes': 4,
    'wms_tms': 3,

}
technologies = ['IIoT', 'Cloud Services', 'AI,CV,PM,RPA', 'Cyber Security', 'Robotics', 'AR/VR', 'Digital Twin', 'DM & BI']
linguistic_terms = {
    'nan': 0,
    'low': 0.25,
    'medium': 0.5,
    'high': 0.75,
}

for system in systems:
    csv_path = f'responses/final_{system}_csv.csv'
    df = pd.read_csv(csv_path)
    # get columns name
    cols = list(df.columns)[1:]
    print(cols)

    res = []
    for index, row in df.iterrows():
        col_mean = []
        technologies_system = []
        for technology in cols:
            elems = row[technology]
            if pd.isna(elems):
                col_mean.append(0.0)
                technologies_system.append(technology)
                continue
            elems = elems.split(',')
            elem_mean = 0
            for elem in elems:
                # make elem all lower case
                elem = elem.lower()
                elem_mean += linguistic_terms[elem]
            elem_mean /= n_experts[system]
            col_mean.append(elem_mean)
            technologies_system.append(technology)
        res.append(col_mean)

    res_df = pd.DataFrame(res, columns=technologies_system)
    res_df.to_csv(f'data/final_{system}_mean.csv', index=False)