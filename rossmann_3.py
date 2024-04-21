# %%
### Seção 0 - Carregar funções e dados
## Importações

import pandas as pd
import math
import calendar
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.core.display import HTML
from IPython.display import Image
import datetime

# %%
## Configurando os gráficos
def jupyter_settings():
    %matplotlib inline
    
    plt.style.use( 'bmh' )
    plt.rcParams['figure.figsize'] = [25, 12]
    plt.rcParams['font.size'] = 24
    display( HTML( '<style>.container { width:100% !important; }</style>') )
    pd.options.display.max_columns = None
    pd.options.display.max_rows = None
    pd.set_option( 'display.expand_frame_repr', False )
    sns.set_theme(style="ticks", color_codes=True)
jupyter_settings()

# %%
df3 = pd.read_csv('data/df2.csv')
# %%
