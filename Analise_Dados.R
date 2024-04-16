library(tidyverse)
library(ggplot2)
library(glmnet)
library(pROC)
library(tibble)
library(glmnet)
library(Metrics)

library(rsample)
library(rpart)
library(partykit)
library(ISLR)
library(rpart.plot)
library(ranger) 

# A 
# Load the data into a DataFrame called 'df'
df <- read.csv('/Users/izadoraramos/Desktop/Insper/AprendizagemEstatisticaDeMaquina1/AnaliseDeDados/sao-paulo-properties-april-2019.csv')

# Calculando a frequência absoluta 
freq_abs <- table(df$Negotiation.Type)

# Calculando a frequência absoluta
freq_rel <- prop.table(freq_abs)

# B
ggplot(df, aes(x = Price, y = Condo, color = Negotiation.Type)) + 
  geom_point() +
  labs(title = "Gráfico de Dispersão", x = "Eixo X", y = "Eixo Y") 

# Indica o problema de escala porque tem um monte de valores em zero. 
# Existem apartamento muito caros sem valor de condomínio 
# Parece que existe uma relacao linear nos dados, pois a medida que o valor apartamento aumenta o valor do condomínio aumenta também 
# Faz sentido que o valor imóvel ser zero para quem está alugando o imóvel. 

# C
ggplot(df, aes(x = Price, y = Condo, color = Negotiation.Type)) + 
  geom_point() +
  labs(title = "Gráfico de Dispersão", x = "Eixo X", y = "Eixo Y") + 
  facet_wrap(~ Negotiation.Type, scales="free")

ggplot(df, aes(x = Price, y = Condo, color = Negotiation.Type)) + 
  geom_point() +
  labs(title = "Gráfico de Dispersão", x = "Eixo X", y = "Eixo Y") + 
  facet_wrap(~ Negotiation.Type)

# D
x <- df %>%
  group_by(District) %>%
  summarise(Count = n()) %>%
  arrange(desc(Count)) %>%
  head(10)

ggplot(x, aes(x = District, y = Count)) +
  geom_bar(stat = "identity", fill = "blue") +
  labs(title = "Título do Histograma", x = "Nome da Variável no Eixo X", y = "Frequência")


library(ggplot2)

ggplot(x, aes(x = reorder(District, -Count), y = Count)) +
  geom_bar(stat = "identity", fill = "blue") +
  labs(title = "Frequência", x = "Bairros", y = "Frequência") +
  theme_minimal()

# E

df_rent <- df %>% 
  filter(Negotiation.Type == 'rent')

summary(df_rent)

## deletar variáveia que tem somente uma observação 
library(skimr)
skim(df_rent)

df_rent <- subset(df_rent, select = -c(Negotiation.Type, Property.Type)) # esta dando errado - rever como excluir coluna

# A coluna District apresenta somente dois valores para Grajaú. Caso não fiquem separados na base de treino e teste darão problema na modelagem. Decidi exluir. 
table(df_rent$District)
#df_rent <- subset(df_rent, District != "Grajaú/São Paulo", -2)


# Separar a base em treino e teste  ---------------------------------------
set.seed(42)

partition <- sample(nrow(df_rent), size = .70 * nrow(df_rent), replace = FALSE)

# Preparando variáveis para glmnet ----------------------------------------

df_tr <- df_rent[partition, ]
x_tr <- model.matrix(~ . -Price, df_rent)[partition, ]
print(nrow(x_tr))
y_tr <- df_rent$Price[partition]
print(length(y_tr))

df_test <- df_rent[-partition, ]
x_test <- model.matrix(~ . - Price, df_rent)[-partition, ]
print(nrow(x_test))
y_test <- df_rent$Price[-partition]
print(length(y_test))

# Data frame resultados  --------------------------------------------------

resultados <- tibble(modelo = c("linear", "stepwise", "ridge", "lasso", "arvore_decisao", "floresta_aleatória"), 
                     rmse = NA, 
                     r2 = NA,
                     mae = NA, 
                     )

# Regressão Linear  -------------------------------------------------------

regressao_linear <- lm(Price ~ ., df_rent[partition, ])

summary_linear <- summary(regressao_linear )

predictions_linear <- predict(regressao_linear , newdata = df_rent[-partition, ])

rmse_linear <- rmse(df_rent[-partition, 'Price' ], predictions_linear)

r2_linear <- cor(df_rent[-partition, 'Price' ], predictions_linear)^2

mae_linear <- mean(abs(y_test - predictions_linear))

variaveis_modelo <- names(coef(regressao_linear))

print(summary_linear$r.squared)

#Plotando oo gráfico 
#Dados reais de Price
valores_reais <- df_rent[-partition, "Price"]

# Criando um data frame para plotagem
dados_plot <- data.frame(ValoresReais = valores_reais, Predicoes = predictions_linear)

library(ggplot2)
ggplot(dados_plot, aes(x = ValoresReais, y = predictions_linear)) +
  geom_point() +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "red") + # Linha y=x para referência
  labs(title = "Valores Preditos vs Valores Reais", x = "Valores Reais", y = "Valores Preditos") +
  theme_minimal()

# imputar erro da previsão no dataframe resultados 

resultados$rmse[resultados$modelo == "linear"] <- rmse_linear 

resultados$r2[resultados$modelo == "linear"] <- r2_linear 

resultados$mae[resultados$modelo == "linear"] <- mae_linear 


# Regressão stepwise  -----------------------------------------------------

# Carregar o pacote necessário
library(MASS)

# Executar a seleção para frente

modelo_completo <- lm(Price ~ ., data = df_rent[partition, ])

modelo_backward <- stepAIC(modelo_completo, direction="backward")

predictions_stepwise <- predict(modelo_backward, newdata = df_rent[-partition, ])

rmse_stepwise <- rmse(df_rent[-partition, 'Price' ], predictions_stepwise)

r2_stepwise <- cor(df_rent[-partition, 'Price' ], predictions_stepwise)^2

mae_stepwise <- mean(abs(y_test - predictions_stepwise ))

# ver quais variáveis foram mantidas 
variaveis_mantidas <- names(coef(modelo_backward))

# ver variáveis que foram retiradas do modelo 
variaveis_removidas  <- setdiff(variaveis_modelo, variaveis_mantidas)

#Plotando oo gráfico 
#Dados reais de Price
valores_reais <- df_rent[-partition, "Price"]

# Criando um data frame para plotagem
dados_plot <- data.frame(ValoresReais = valores_reais, Predicoes = predictions_stepwise)

library(ggplot2)
ggplot(dados_plot, aes(x = ValoresReais, y = predictions_stepwise)) +
  geom_point() +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "red") + # Linha y=x para referência
  labs(title = "Valores Preditos vs Valores Reais", x = "Valores Reais", y = "Valores Preditos") +
  theme_minimal()

# imputar erro da previsão no dataframe resultados 

resultados$rmse[resultados$modelo == "stepwise"] <- rmse_stepwise 

resultados$r2[resultados$modelo == "stepwise"] <- r2_stepwise

resultados$mae[resultados$modelo == "stepwise"] <- mae_stepwise 


# Ridge -------------------------------------------------------------------

set.seed(123)
# como tem hiperparametro podemos fazer cross validation para achar o lambida ótimo 

# validação cruzada me 10 lotes 

eqm_aux <- vector("numeric", 10) # vetor para armazenar o EQM a cada 10 lotes na validação cruzada 

(lotes <- sample(rep(1:10, each = 10)))

lambdas <- 10^seq(2, -2, by = -0.1)
resultados_lambda_ridge <- tibble(lambda = lambdas, eqm = NA_real_)

all_preds_ridge <- rep(NA, nrow(x_test))
for (i in 1:length(lambdas)) {
  for (v in 1:10) {
    # Ajustando regressão ridge 
    fit_ridge <- glmnet(x_tr[lotes != v,], 
                        y_tr[lotes != v], 
                        alpha = 0, 
                        lambda = lambdas[i])
    
    # Calcular o EQM no lote não incluído 
    pred <- predict(fit_ridge, newx = x_test[lotes == v,], s = lambdas[i])
    all_preds_ridge[lotes == v] <- pred
    eqm_aux[v] <- rmse(y_test[lotes == v], pred)
  }
  
summary_ridge <- summary(fit_ridge)
  
  # Colocando resultados no df
  resultados_lambda_ridge$eqm[i] <- mean(eqm_aux)
}


# Plotando gráficos  
ggplot(resultados_lambda_ridge, aes(x = log10(lambda), y = eqm)) + 
  geom_point() +
  geom_line() +
  labs(x = 'Log10(Lambda)', y = 'EQM',
       title = 'EQM por Lambda na Regressão Rodge')

ggplot(resultados_lambda_ridge, aes(x = (lambda), y = eqm)) + 
  geom_point() +
  geom_line() +
  labs(x = 'Lambda', y = 'EQM',
       title = 'EQM por Lambda na Regressão Ridge')

posicao_menor_eqm <- which.min(resultados_lambda_ridge$eqm)
linha_menor_eqm <- resultados_lambda_ridge[posicao_menor_eqm, ]
menor_lambda_ridge <- linha_menor_eqm$lambda

rmse_ridge <- linha_menor_eqm$eqm

r2_ridge <- cor(y_test, all_preds_ridge)^2

mae_ridge<- mean(abs(y_test - all_preds_ridge))

# imputando resultados 

resultados$rmse[resultados$modelo == "ridge"] <- rmse_ridge

resultados$r2[resultados$modelo == "ridge"] <- r2_ridge

resultados$mae[resultados$modelo == "ridge"] <- mae_ridge


# Lasso  ------------------------------------------------------------------

set.seed(123)

# validação cruzada me 10 lotes 

eqm_aux <- vector("numeric", 10) # vetor para armazenar o EQM a cada 10 lotes na validação cruzada 

(lotes <- sample(rep(1:10, each = 10)))

lambdas <- 10^seq(2, -2, by = -0.1)
resultados_lambda_lasso <- tibble(lambda = lambdas, eqm = NA_real_)

all_preds_lasso <- rep(NA, nrow(x_test))
for (i in 1:length(lambdas)) {
  for (v in 1:10) {
    # Ajustando regressão lasso
    fit_lasso <- glmnet(x_tr[lotes != v,], 
                        y_tr[lotes != v], 
                        alpha = 1, 
                        lambda = lambdas[i])
    
    # Calcular o EQM no lote não incluído 
    pred <- predict(fit_lasso, newx = x_test[lotes == v,], s = lambdas[i])
    all_preds_lasso[lotes == v] <- pred
    eqm_aux[v] <- rmse(y_test[lotes == v], pred)
  }
  
  # Colocando resultados no df
  resultados_lambda_lasso$eqm[i] <- mean(eqm_aux)
}


# Plotando gráficos  
ggplot(resultados_lambda_lasso, aes(x = log10(lambda), y = eqm)) + 
  geom_point() +
  geom_line() +
  labs(x = 'Log10(Lambda)', y = 'EQM',
       title = 'EQM por Lambda na Regressão Lasso')

ggplot(resultados_lambda_lasso, aes(x = (lambda), y = eqm)) + 
  geom_point() +
  geom_line() +
  labs(x = 'Lambda', y = 'EQM',
       title = 'EQM por Lambda na Regressão Lasso')

posicao_menor_eqm <- which.min(resultados_lambda_lasso$eqm)
linha_menor_eqm <- resultados_lambda_lasso[posicao_menor_eqm, ]
menor_lambda_lasso <- linha_menor_eqm$lambda
rmse_lasso <- linha_menor_eqm$eqm

# Obter os coeficientes para o valor escolhido de lambda
lasso_coef <- predict(fit_lasso, type="coefficients", s=menor_lambda_lasso)
# Converter a matriz esparsa em um vetor normal
lasso_coef <- as.numeric(lasso_coef)
# Nomes das variáveis (as colunas de `x`)
variaveis <- colnames(x)
# Variáveis que foram descartadas (coeficientes iguais a zero)
variaveis_descartadas <- variaveis[lasso_coef == 0]
# os resultados semelhantes de lasso e ridge pode ser explicado também porque lasso descartou somente uma variável


r2_lasso <- cor(y_test, all_preds_lasso)^2

mae_lasso <- mean(abs(y_test - all_preds_lasso))

# imputando resultados 
resultados$rmse[resultados$modelo == "lasso"] <- rmse_lasso

resultados$r2[resultados$modelo == "lasso"] <- r2_lasso

resultados$mae[resultados$modelo == "lasso"] <- mae_lasso


# Árvore de decisão  ------------------------------------------------------

arvore <- rpart(Price ~ ., data = df_tr, control = rpart.control(xval = 10, cp = 0))
plotcp(arvore)

arvore$cptable

corte <- arvore$cptable %>%
  as_tibble() %>%
  filter(xerror == min(xerror)) %>%
  transmute(corte = xerror + xstd)

cp_ot <- arvore$cptable %>%
  as_tibble() %>%
  filter(xerror <= corte[[1]])

# Encontrar o menor xerror associado ao xstd (desvio padrão)
linha_min_xerror <- cp_ot %>%
  filter(xerror == min(xerror))

# Mínimo xerror e  mínimo xerro std 
min_xerror <- min(cp_ot$xerror)
min_xerror_std <- cp_ot$xstd[which.min(cp_ot$xerror)]

# Aplicar a "1-SE rule"
upper_limit <- min_xerror + min_xerror_std
cp_1se <- max(cp_ot$CP[cp_ot$xerror <= upper_limit])

poda1 <- prune(arvore, cp = cp_1se)
rpart.plot(poda1, roundint = FALSE, compress = TRUE)


tibble(y_obs = df_test$Price, 
       y_pred = predict(poda1, newdata = df_test)) %>%
  ggplot(aes(y_obs, y_pred)) + geom_abline(slope = 1, intercept = 0, color = "red", size = 2) + geom_point(size = 3, alpha = .5)

predicoes_arvore <- predict(poda1, newdata = df_test)

rmse_arvore<- sqrt(mean((df_test$Price - predicoes_arvore)^2))
r2_arvore <- cor(df_test$Price, predicoes_arvore)^2
mae_arvore <- mean(abs(df_test$Price - predicoes_arvore))

resultados$rmse[resultados$modelo == "arvore_decisao"] <- rmse_arvore

resultados$r2[resultados$modelo == "arvore_decisao"] <- r2_arvore

resultados$mae[resultados$modelo == "arvore_decisao"] <- mae_arvore

# Floresta  ---------------------------------------------------------------
resultados_floresta <- crossing(mtry = c(2, 4, 8, 13),
                       n_arvores = c(1:10, seq(10, 500, 10)))

ajusta <- function(mtry, n_arvores) {
  rf <- ranger(Price ~ ., num.trees = n_arvores, mtry = mtry, data = df_tr)
  return(rf$prediction.error)
}
resultados_floresta <- resultados_floresta %>%
  mutate(mse = map2_dbl(mtry, n_arvores, ajusta))
head(resultados_floresta)

resultados_floresta %>%
  mutate(mtry = factor(mtry)) %>%
  ggplot(aes(n_arvores, mse, group = mtry, color = mtry)) +
  geom_line( size = 1.2) +
  labs(x = "Número de Árvores", y = "MSE (OOB)") +
  theme_bw()

rf <- ranger(Price ~ ., importance = "impurity", data = df_tr)
vip::vip(rf, aesthetics = list(fill = "#FF5757"))

rf <- ranger(Price~ ., importance = "permutation", data = df_tr)
vip::vip(rf, aesthetics = list(fill = "#FF5757"))

posicao_menor_mse <- which.min(resultados_floresta$mse)
linha_menor_mse <- resultados_floresta[posicao_menor_mse, ]
mtry_ot <- linha_menor_mse$mtry
n_arvores_ot <- linha_menor_mse$n_arvores


rf_otimo <- ranger(
  formula = Price ~ ., 
  data = df_tr, 
  num.trees = n_arvores_ot, 
  mtry = mtry_ot,
  importance = 'permutation'
)

predicoes_floresta <- predict(rf_otimo, data = df_test)$predictions

rmse_floresta <- sqrt(mean((df_test$Price - predicoes_floresta)^2))
r2_floresta <- cor(df_test$Price, predicoes_floresta)^2
mae_floresta <- mean(abs(df_test$Price - predicoes_floresta))

#mudar o nome do dataframe resultados da floresta 
resultados$rmse[resultados$modelo == "floresta_aleatória"] <- rmse_floresta

resultados$r2[resultados$modelo == "floresta_aleatória"] <- r2_floresta

resultados$mae[resultados$modelo == "floresta_aleatória"] <- mae_floresta

resultados
