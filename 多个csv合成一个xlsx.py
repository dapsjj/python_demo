import glob, os
import pandas as pd

from pandas import ExcelWriter

writer = ExcelWriter(r"C:/Users/XXX/Desktop/test/compiled.xlsx")
for filename in glob.glob(r"C:/Users/XXX/Desktop/test/*.csv"):
    df_csv = pd.read_csv(filename)
    (_, f_name) = os.path.split(filename)
    (f_shortname, _) = os.path.splitext(f_name)
    df_csv.to_excel(writer, f_shortname, index=False)
writer.save()
