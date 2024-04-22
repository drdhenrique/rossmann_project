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
## Feature Engeneering

df2 = df1.copy()

# %%
## Mapa de Hipóteses
Image('images\mapa_de_hipoteses.png')

# %%
## Construção das hipóteses com base no mapa de hipóteses

# Criação inicial de hipóteses. Hipótese são apostas. Aqui podemos
# tentar criar muitas relações com todos os Agentes/atributos do
# mapa de hipóteses

"""
Lojas:
1. Lojas com mais funcionários deveriam vender mais
2. Lojas com estoque maior deveriam vender mais
3. Lojas maiores deveriam vender mais
4. Lojas com maior variabilidade de produtos deveriam vender mais
5. Lojas com competidores mais próximos deveriam vender menos

Tempo
1. Lojas que ficam fechadas em mais feriados deveriam vender menos
2. Lojas que abrem fds deveriam vender mais
3. Lojam vendem mais a partir do dia 10, em média
4. Durante os períodos de férias escolares, vende-se menos
5. No inverno vende-se mais

Produtos
1. Produtos em promoção aumentam o faturamento médio
2. Produtos/cat de produtos em promoção aumentam o faturamento
3. Lojas com preços maiores para determinados produtos vendem menos
4. Lojas com estoques maiores vendem mais
5. Produtos mais expostos vendem mais

Clientes
1. Clientes com mais filhos compram mais
2. Clientes com salário maior compram mais
3. Clientes mais velhos compram mais
4. Clientes que vão mais vezes até a loja compram mais

Localidade
1. Lojas perto de hospitais vendem mais
2. Lojas perto de escolas/universidades vendem mais
3. Lojas urbanas vendem mais que rurais
4. Lojas centrais vendem mais que lojas em bairros

"""
# %%
# Lista final de Hipóteses. Nessa primeira iteração do CRISP,
# vamos nos atentar apenas às hipóteses que podem ser respondidas
# com os dados disponíveis.

"""

1. Lojas maiores deveriam vender mais
2. Lojas com maior variabilidade de produtos deveriam vender mais
3. Lojas com competidores mais próximos deveriam vender menos
4. Lojas com competidores à mais tempo deveriam vendem mais.
5. Lojas que ficam fechadas em mais feriados deveriam vender menos
6. Lojas que abrem fds deveriam vender mais
7. Lojam vendem mais a partir do dia 10, em média
8. Durante os períodos de férias escolares, vende-se menos
9. No inverno vende-se mais
10. Lojas com promoção aumentam o faturamento médio
11. Lojas com promoções ativas por mais tempo deveriam vender mais.
12. Lojas com mais dias de promoção deveriam vender mais.
13. Lojas com mais promoções consecutivas deveriam vender mais.
14. Lojas abertas durante o feriado de Natal deveriam vender mais.
15. Lojas deveriam vender mais ao longo dos anos.
16. Lojas deveriam vender mais no segundo semestre do ano.
17. Lojas deveriam vender menos durante os feriados escolares.

"""
# Ano
df2['year'] = df2.date.dt.year

# Mês
df2['month'] = df2.date.dt.month

# Week
df2['week_of_year'] = df2.date.dt.isocalendar().week

# Dia
df2['day'] = df2.date.dt.day

# Year - week
df2['year_week'] = df2.date.dt.strftime( '%Y-%W')


# %%

# achando tempo em meses que a competição começou
df2['competition_since '] = df2.apply(lambda x: datetime.datetime(year=x['competition_open_since_year'], 
                  month= x['competition_open_since_month'], 
                  day=1), axis= 1)

df2['competition_time_months'] = ((df2['date'] - df2['competition_since '])/30).apply(lambda x: x.days).astype( 'Int64')
# %%

# promo since

df2['promo_since'] = df2['promo2_since_year'].astype( str ) + '-' + df2['promo2_since_week'].astype( str )
df2['promo_since'] = df2['promo_since'].apply( lambda x: datetime.datetime.strptime( x + '-1', '%Y-%W-%w' ) - datetime.timedelta( days=7 ) )
df2['weeks_promo2'] = ( ( df2['date'] - df2['promo_since'] )/7 ).apply( lambda x: x.days ).astype( int )

# %%

df2['assortment'] = df2['assortment'].apply(lambda x: 'basic' if x =='a'else
                                            'extra' if x == 'b' else 'extended' )

df2['state_holiday'] = df2['state_holiday'].apply( lambda x: 'public_holiday' if x == 'a' else 'easter_holiday' if x == 'b' else 'christmas' if x == 'c' else 'regular_day' )

# %%

df2.to_csv('data/df2.csv')
