# %%
### Seção 0 - Carregar funções e dados
## Importações

import pandas as pd
import math
import calendar
import numpy as np
""" import seaborn as sns
import matplotlib.pyplot as plt
from IPython.core.display import HTML
from IPython.display import Image
import datetime """

# %%
## Configurando os gráficos
def jupyter_settings():
    %matplotlib inline
    
    """ plt.style.use( 'bmh' )
    plt.rcParams['figure.figsize'] = [25, 12]
    plt.rcParams['font.size'] = 24
    display( HTML( '<style>.container { width:100% !important; }</style>') ) """
    pd.options.display.max_columns = None
    pd.options.display.max_rows = None
    pd.set_option( 'display.expand_frame_repr', False )
    """ sns.set_theme(style="ticks", color_codes=True) """
jupyter_settings() 

# %%
df3 = pd.read_csv('data/df2.csv', index_col= [0])


# %%
df3.head(3)

# %%
df3.shape
# %%

## Filtragem de linhas

df3 = df3[(df3['open'] != 0) & (df3['sales'] > 0)]

# %%

## Filtragem de colunas
cols_drop = ['open', 'customers','month_map', 'promo_interval' ]
df3 = df3.drop(cols_drop, axis = 1)

# %%

df3.to_csv('data/df3.csv')
# %%
