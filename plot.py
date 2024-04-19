import pandas as pd
import matplotlib.pyplot as plt
from icecream import ic

FILENAME = 'SIZE400.LOSS.0.1.csv'

df = pd.read_csv(FILENAME, index_col=[0])

df['val_acc'].plot()
df['val_year_acc'].plot()
df['val_make_acc'].plot()
df['val_type_acc'].plot()

idxmax = ic(df['val_acc'].idxmax())
ic(df['val_acc'][idxmax]*100)
ic(df['val_year_acc'][idxmax]*100)
ic(df['val_make_acc'][idxmax]*100)
ic(df['val_type_acc'][idxmax]*100)

plt.title("Accuracy over Epochs")
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Class', 'Year', 'Make', 'Type'])
plt.savefig(FILENAME.replace('csv', 'png'))
