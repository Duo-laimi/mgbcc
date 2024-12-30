# 绘制 p - 内部指标/外部指标 图
import re
import pandas as pd
dataset = "Caltech101-20"
path = "./" + dataset + ".dat"
pattern = r"latent_dim=(\d+),\s*p=(\d+),\s*learning_rate=([\d.]+),\s*normalize=(True|False),\s*batch_size=(-?[\d.]+),\s*best_result=\{'epoch': (\d+), 'ACC': ([\d.]+), 'NMI': ([\d.]+), 'PUR': ([\d.]+), 'sh': ([\d.]+), 'ch': ([\d.]+), 'db': ([\d.]+)\},\s*final_result=\{'epoch': (\d+), 'ACC': ([\d.]+), 'NMI': ([\d.]+), 'PUR': ([\d.]+), 'sh': ([\d.]+), 'ch': ([\d.]+), 'db': ([\d.]+)\}"
pattern = re.compile(pattern)

headers = ["latent_dim", "p", "learning_rate", "normalize", "batch_size", "best_epoch", "best_acc", "best_nmi", "best_pur", "best_sh", "best_ch", "best_db", "final_epoch", "final_acc", "final_nmi", "final_pur", "final_sh", "final_ch", "final_db"]
df = pd.DataFrame(columns=headers)
# 打开文件
with open(path, "r") as f:
    # 逐行匹配
    for line in f:
        # 如果匹配
        match = pattern.match(line)
        if match:
            data_line = list(match.groups())
            for i in range(0, len(data_line)):
                data_item = data_line[i]
                if re.match(r"^-?\d+$", data_item):
                    data_line[i] = int(data_item)
                elif re.match(r"^-?\d+\.\d+$", data_item):
                    data_line[i] = float(data_item)
            df.loc[len(df)] = data_line

df.to_excel("./result_caltech101_20.xlsx", index=False)







