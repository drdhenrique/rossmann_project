# %%
### Seção 0 - Carregar funções e dados
## Importações

import pandas as pd
import inflection
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
df4 = pd.read_csv('data/df3.csv', index_col= [ 0 ])
# %%
df4.sample(4)
# %%
## EDA

# %%
## Análise Univariada
resposta = df4['sales']
num_attributes = df4.select_dtypes(include=['int64', 'float64'])
cat_attributes = df4.select_dtypes(exclude=['int64', 'float64', 'datetime64[ns]'])

# %%
# Visualização da variável resposta 'sales'

sns.histplot(resposta, kde = True, bins = 40);
# %%
sns.histplot(df4['sales'],kde = True, bins = 50 );
# %%
## Visualização das variáveis numéricas

num_attributes.hist(bins = 50);

# %%
## Exploração das variáveis categóricas
cat_attributes.store_type.value_counts()

# %%
plt.subplot( 3, 2, 1 )
sns.countplot(x='state_holiday', data=df4[df4.state_holiday != 'regular_day'], hue='state_holiday', palette='Set1');

plt.subplot( 3, 2, 2 )
sns.kdeplot(x = 'sales', fill = True,  data= df4[df4['state_holiday'] == 'public_holiday'], label = 'regular_holiday');
sns.kdeplot(x = 'sales', fill = True,  data = df4[df4['state_holiday'] == 'easter_holiday'], label = 'easter_holiday');
sns.kdeplot(x = 'sales', label = 'christmas',fill = True, data = df4[df4['state_holiday'] == 'christmas']);
plt.legend()

plt.subplot( 3, 2, 3 )
sns.countplot(x='assortment', data=df4, hue='assortment', palette='Set1');

plt.subplot( 3, 2, 4 )
sns.kdeplot(x = 'sales', fill = True,  data= df4[df4.assortment == 'basic'], label = 'basic');
sns.kdeplot(x = 'sales', fill = True,  data= df4[df4.assortment == 'extended'], label = 'extended');
sns.kdeplot(x = 'sales', fill = True,  data= df4[df4.assortment == 'extra'], label = 'extra');
plt.legend()

plt.subplot( 3, 2, 5 )
sns.countplot(x='store_type', data=df4, hue='store_type', palette='Set1', legend=False);

plt.subplot( 3, 2, 6 )
sns.kdeplot(x = 'sales', fill = True,  data= df4[df4['store_type'] == 'a'], label = 'a');
sns.kdeplot(x = 'sales', fill = True,  data = df4[df4['store_type'] == 'b'], label = 'b');
sns.kdeplot(x = 'sales', label = 'c',fill = True, data = df4[df4['store_type'] == 'c']);
sns.kdeplot(x = 'sales', label = 'd',fill = True, data = df4[df4['store_type'] == 'd']);
plt.legend()
# %%

