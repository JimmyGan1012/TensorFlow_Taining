import chipwhisperer as cw
import pandas as pd
import numpy as np

proj = cw.open_project("../../Demo/collections/May_18_RevisedFirmware/Key_{}.cwp".format(1))

traces=[]

for trace in proj.traces:
    traces.append([trace.wave,trace.key,trace.textin,trace.textout])

df_Traces = pd.DataFrame(data = traces, columns=["wave","key","text-in","text-out"])

df_Traces.to_csv(r"Key_{}.csv".format(1))


