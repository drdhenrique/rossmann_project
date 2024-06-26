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
cat_attributes.assortment.value_counts()

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
## Análise Bivariada

"""

H2. Lojas com maior variabilidade de produtos deveriam vender mais

"""

aux1 = df4[['assortment', 'sales']].groupby('assortment').mean().reset_index()

sns.barplot(x = 'assortment', y = 'sales', data= aux1, hue= 'assortment', palette= 'Set1');
# %%
aux2 = df4[['year_week', 'assortment', 'sales']].groupby(['assortment', 'year_week']).mean().reset_index()

aux3 = aux2.pivot(index= 'year_week', values='sales',columns='assortment').plot()

# Concluímos então que lojas com maior variedade vendem mais em média


# %%
"""
H1. Lojas maiores deveriam vender mais

"""
aux1 = df4[['store_type', 'sales']].groupby('store_type').mean().reset_index()

sns.barplot(x = 'store_type', y = 'sales', data= aux1, hue= 'store_type', palette= 'Set1');

# Aqui não conseguimos concluir muita coisa, devido ao fato de não sabermos o que significa cada classificação 'a', até 'd'.
# %%

"""

H3. Lojas com competidores mais próximos deveriam vender menos

"""
aux1 = df4[['competition_distance', 'sales']].groupby( 'competition_distance' ).mean().reset_index()

plt.subplot( 1, 3, 1 )
sns.scatterplot( x ='competition_distance', y='sales', data=aux1 );

plt.subplot( 1, 3, 2 )
bins = list( np.arange( 0, 20000, 1000) )
aux1['competition_distance_binned'] = pd.cut( aux1['competition_distance'],bins=bins )
aux2 = aux1[['competition_distance_binned', 'sales']].groupby('competition_distance_binned',observed=True ).sum().reset_index()
sns.barplot( x='competition_distance_binned', y='sales', data=aux2 );
plt.xticks( rotation=90);

plt.subplot( 1, 3, 3 )
sns.heatmap( df4[['competition_distance', 'sales']].corr(), annot= True);
# %%

"""

H4. Lojas com competidores à mais tempo deveriam vendem mais.

"""

plt.subplot( 1, 3, 1 )
aux1 = df4[['competition_time_months', 'sales']].groupby('competition_time_months' ).sum().reset_index()
aux2 = aux1[( aux1['competition_time_months'] < 120 ) & (aux1['competition_time_months'] != 0 )]
sns.barplot( x='competition_time_months', y='sales', data=aux2  );
plt.xticks( rotation=90 );
plt.subplot( 1, 3, 2 )
sns.regplot( x='competition_time_months', y='sales', data=aux2 );
plt.subplot( 1, 3, 3 )
x = sns.heatmap( aux1.corr(method='kendall'), annot=True );

# %%


