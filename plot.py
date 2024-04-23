import pandas as pd
import matplotlib.pyplot as plt
from icecream import ic

CSV_FILENAME = 'SIZE400.LR0.005.csv'
TITLE = 'LR 0.005'
OUTPUT_FILENAME = 'SIZE.400.BATCH.32.LR.0.005' + '.png'

df = pd.read_csv(CSV_FILENAME, index_col=[0])

df['val_acc'].plot()
df['val_year_acc'].plot()
df['val_make_acc'].plot()
df['val_type_acc'].plot()

idxmax = ic(df['val_acc'].idxmax())
cla_max = ic(df['val_acc'][idxmax]*100)
year_max = ic(df['val_year_acc'][idxmax]*100)
make_max = ic(df['val_make_acc'][idxmax]*100)
type_max = ic(df['val_type_acc'][idxmax]*100)

plt.title(f"Accuracy over Epochs\n{TITLE}")
plt.grid()
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend([f'Class (Max: {cla_max:.1f}%)', f'Year (Max: {year_max:.1f}%)', f'Make (Max: {make_max:.1f}%)', f'Type (Max: {type_max:.1f}%)'])
plt.savefig(OUTPUT_FILENAME)
