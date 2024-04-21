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
## Carregamento dos dados

df_sales_raw = pd.read_csv('data/train.csv', low_memory= False)
df_stores_raw = pd.read_csv('data/store.csv')

# Merge dos dados

df_raw = pd.merge(df_sales_raw, df_stores_raw, how= 'left', on = 'Store')

# %%
### Seção 1 - Descrição dos dados

## Cópia dos dados originais ao trocar de seção
df1 = df_raw.copy()

# %%
## Renomear colunas pra snake_case

df_raw.columns
# %%
cols_old = ['Store', 'DayOfWeek', 'Date', 'Sales', 'Customers', 
            'Open', 'Promo', 'StateHoliday', 'SchoolHoliday', 
            'StoreType', 'Assortment', 'CompetitionDistance', 
            'CompetitionOpenSinceMonth','CompetitionOpenSinceYear', 
            'Promo2', 'Promo2SinceWeek', 'Promo2SinceYear', 
            'PromoInterval']

snake_case = lambda x: inflection.underscore(x)

cols_new = list(map(snake_case, cols_old))
cols_new
# %%
df1.columns = cols_new
df1.columns
# %%
## Dimensão dos dados

print(f'Número de linhas: {df1.shape[0]}')
print(f'Número de colunas: {df1.shape[1]}')
# %%
## Verificação dos tipos dos dados
df1.dtypes

# Originalmente o 'tipo' de date estava como object, 
# por isso trocaremos para datetime

# %%
df1['date'] = pd.to_datetime(df1['date'])
df1.dtypes
# %%
## Verificação de NAs
df1.isna().sum()
# %%
## Preencher os NAs

# CompetitionDistance - distance in meters to the nearest competitor store
# Como competition_distance é a distância até o concorrente mais próximo
# vamos colocar um valor muito grande para preencher os NAs

# df1.competition_distance.max() retorna 75860.0, logo usaremos 200000.0 nos NAs

df1.competition_distance = df1.competition_distance.apply(lambda x: 200000.0 if math.isnan(x) else x)

# CompetitionOpenSince[Month/Year] - gives the approximate year and month of the time the 
# nearest competitor was opened 
# Aqui vamos inicialmente colocar o mês/ano de date nas entradas que estiverem com NA.  
# Talvez não seja a melhor escolha, mas voltaremos nisso numa segunda iteração do CRISP-DM 
# caso o resultado não seja satisfatório 

df1.competition_open_since_month = df1.apply(lambda x: x['date'].month if math.isnan(x['competition_open_since_month']) else x['competition_open_since_month'], axis = 1)
df1.competition_open_since_year = df1.apply(lambda x: x['date'].year if math.isnan(x['competition_open_since_year']) else x['competition_open_since_year'], axis = 1)  

# %%

# Promo - indicates whether a store is running a promo on that day
# Promo2 - Promo2 is a continuing and consecutive promotion for some stores: 0 = store is not

# participating, 1 = store is participating
# Promo2Since[Year/Week] - describes the year and calendar week when the store started 
# participating in Promo2
df1.promo2_since_week = df1.apply(lambda x: x['date'].week if math.isnan(x['promo2_since_week']) else x['promo2_since_week'], axis = 1)

df1.promo2_since_year = df1.apply(lambda x: x['date'].year if math.isnan(x['promo2_since_year']) else x['promo2_since_year'], axis = 1) 
# PromoInterval - describes the consecutive intervals Promo2 is started, naming the months 
# the promotion is started anew. E.g. "Feb,May,Aug,Nov" means each round starts in February, 
# May, August, November of any given year for that store  

month_map = {i:calendar.month_abbr[i] for i in range(1, 13)}

df1['promo_interval'].fillna(0, inplace= True)
df1['month_map'] = df1['date'].dt.month.map(month_map)

df1['is_promo'] = df1[['promo_interval','month_map']].apply(lambda x: 0 if x.promo_interval == 0 else 1 if x['month_map'] in x['promo_interval'].split(',') else 0, axis = 1)






# %%
## Conferindo novamente os tipos dos dados

df1.dtypes
# %%
df1.competition_open_since_month = df1.competition_open_since_month.astype( 'Int64' )
df1.competition_open_since_year = df1.competition_open_since_year.astype( 'Int64' )
df1.promo2_since_week = df1.promo2_since_week.astype( 'Int64' )
df1.promo2_since_year = df1.promo2_since_year.astype( 'Int64' )

# %%
## Estatísticas Descritivas
num_attributes = df1.select_dtypes(include=['int64', 'float64'])
cat_attributes = df1.select_dtypes(exclude=['int64', 'float64', 'datetime64[ns]'])

# %%
num_attributes.sample(2)

# %%

cat_attributes.sample(2)

# %%
# Medidas Resumo - Média e mediana
mr1 = pd.DataFrame(num_attributes.apply(np.mean)).T
mr2 = pd.DataFrame(num_attributes.apply(np.median)).T

# %%
# Medidas de Dispersão - Desvio-Padrão, máximo, mínimo, amplitude, curtose e viés

md1 = pd.DataFrame(num_attributes.apply(np.std)).T
md2 = pd.DataFrame(num_attributes.apply(min)).T
md3 = pd.DataFrame(num_attributes.apply(max)).T
md4 = pd.DataFrame(num_attributes.apply(lambda x: x.max()-x.min())).T
md5 = pd.DataFrame(num_attributes.apply(lambda x: x.skew())).T
md6 = pd.DataFrame(num_attributes.apply(lambda x: x.kurtosis())).T
# %%
## Entendendo os atributos numéricos
metricas = pd.concat([md2, md3, md4, mr1, mr2, md1, md5, md6]).T.reset_index()
metricas.columns = ['atributos', 'min', 'max', 'amplitude', 'média', 'mediana',
                    'desvio-padrão', 'assimetria', 'curtose']
metricas
# %%
sns.displot(df1.competition_distance);
# %%
sns.kdeplot(df1.sales);

# sns.pairplot(num_attributes)

# %%
## Entendendo os atributos categóricos

cat_attributes.apply(lambda x: x.unique())
# %%
## 
aux1 = df1[(df1.state_holiday != '0') & (df1.sales > 0)]

plt.subplot(1,3,1)
sns.boxplot(x='state_holiday',y='sales', data=aux1, hue = 'state_holiday');

plt.subplot(1,3,2)
sns.boxplot(x='store_type',y='sales', data=aux1, hue = 'store_type');

plt.subplot(1,3,3)
sns.boxplot(x='assortment',y='sales', data=aux1, hue = 'assortment');
# %%
