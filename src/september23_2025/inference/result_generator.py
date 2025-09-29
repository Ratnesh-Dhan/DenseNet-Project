import os
import pandas as pd
import re

base_path = "../results/septmber25"
# model_names = os.listdir(base_path)
# print(model_names)
model_names = ['EarlyStoppedBestSeptmber24', 'Septmber24']

def sample_key(r):
    """Sorts samples like C1, C2... C20, then DBM1, DBM2... DBM20"""
    s = r['Sample']
    if s.startswith("DBM"):
        return ("DBM", int(s[3:]))   # everything after "DBM"
    elif s.startswith("C"):
        return ("C", int(s[1:]))     # everything after "C"
    else:
        # fallback for unknown formats
        m = re.match(r"([A-Za-z]+)(\d+)", s)
        if m:
            prefix, num = m.groups()
            return (prefix, int(num))
        return ("ZZZ", 9999)

for model_name in model_names:
    file = os.path.join(base_path, model_name, "final_output.txt")

    count = 0
    count2 = 3
    count3 = 0
    count4 = False
    row = []

    with open(file) as f:
        row_data = {}
        for lines in f:
            count = count + 1
            count2 = count2 + 1

            if count % 4 == 0:
                continue
            if count2 % 4 == 0:
                count2 = 0
                continue
            else:
                count3 = count3 + 1
                x = lines.split(':')
                count4 = not count4
                if count4:
                    row_data['Sample'] = x[0].strip().split(' ')[0].strip()
                y = x[0].strip().split(' ')
                row_data[f'{y[1].strip()} {y[2].strip()}'] = round(float(x[1].strip()), 4)

            if count3 % 2 == 0:
                row.append(row_data)
                row_data = {}

    # ✅ Sort using our safe key
    row.sort(key=sample_key)

    df = pd.DataFrame(row)
    output_file = "../results/Excel_Files"
    os.makedirs(output_file, exist_ok=True)
    df.to_excel(os.path.join(output_file, f'{model_name}.xlsx'), index=False)

    print(f"✅ Saved sorted Excel for {model_name} at {output_file}")
