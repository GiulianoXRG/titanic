# Desafio do Titanic (Kaggle): prever quem sobreviveu.
# Usa só titanic + tidyverse + rpart (já instalados).
library(titanic)
library(tidyverse)
library(rpart)

set.seed(2026)

# 1. Dados -------------------------------------------------------------------
# Junta treino e teste para tratar as variáveis de uma vez só.
dados <- bind_rows(
  mutate(titanic_train, conjunto = "treino"),
  mutate(titanic_test,  conjunto = "teste")
) |>
  rename(
    id_passageiro   = PassengerId,
    sobreviveu      = Survived,
    classe          = Pclass,
    nome            = Name,
    sexo            = Sex,
    idade           = Age,
    irmaos_conjuges = SibSp,
    pais_filhos     = Parch,
    bilhete         = Ticket,
    tarifa          = Fare,
    cabine          = Cabin,
    porto_embarque  = Embarked
  )

# 2. Variáveis ---------------------------------------------------------------
dados <- dados |>
  mutate(
    sexo = ifelse(sexo == "male", "masculino", "feminino"),

    # Título dentro do nome: "Braund, Mr. Owen Harris" -> "Mr"
    titulo = str_trim(str_extract(nome, "(?<=, )[^.]+")),
    titulo = case_when(
      titulo %in% c("Mlle", "Ms")  ~ "Miss",
      titulo == "Mme"              ~ "Mrs",
      titulo %in% c("Mr", "Mrs", "Miss", "Master") ~ titulo,
      TRUE                         ~ "Raro"
    ),

    tamanho_familia = irmaos_conjuges + pais_filhos + 1,
    tipo_familia = case_when(
      tamanho_familia == 1 ~ "sozinho",
      tamanho_familia <= 4 ~ "pequena",
      TRUE                 ~ "grande"
    ),

    tem_cabine     = as.integer(cabine != ""),
    porto_embarque = ifelse(porto_embarque == "", "S", porto_embarque)  # moda
  ) |>
  # Tarifa faltante: mediana da classe
  group_by(classe) |>
  mutate(tarifa = ifelse(is.na(tarifa), median(tarifa, na.rm = TRUE), tarifa)) |>
  # Idade faltante: mediana do título (Master = menino, Mr = adulto etc.)
  group_by(titulo) |>
  mutate(idade = ifelse(is.na(idade), median(idade, na.rm = TRUE), idade)) |>
  # Pessoas no mesmo bilhete viajavam juntas
  group_by(bilhete) |>
  mutate(pessoas_bilhete = n()) |>
  ungroup() |>
  mutate(
    log_tarifa = log1p(tarifa / pessoas_bilhete),  # tarifa por pessoa
    across(c(classe, sexo, titulo, tipo_familia, porto_embarque), factor)
  )

treino <- filter(dados, conjunto == "treino")
teste  <- filter(dados, conjunto == "teste")

# 3. Modelos -----------------------------------------------------------------
# (sexo fica de fora da logística: o título já carrega essa informação)
f_logit  <- sobreviveu ~ titulo + classe + tipo_familia + idade + log_tarifa +
  tem_cabine + porto_embarque
f_arvore <- sobreviveu ~ titulo + sexo + classe + tamanho_familia + idade +
  log_tarifa + tem_cabine + porto_embarque

ajusta <- list(
  logistica = function(d) glm(f_logit, data = d, family = binomial),
  arvore    = function(d) rpart(f_arvore, data = d, method = "class",
                                control = rpart.control(cp = 0.005, minbucket = 5))
)
preve <- list(
  logistica = function(m, d) as.integer(predict(m, d, type = "response") > 0.5),
  arvore    = function(m, d) as.integer(predict(m, d, type = "class") == "1")
)

# 4. Validação cruzada (10 partes, repetida 5 vezes) -------------------------
validacao <- map_dfr(1:5, function(rep) {
  parte <- sample(rep(1:10, length.out = nrow(treino)))
  map_dfr(1:10, function(k) {
    tr <- treino[parte != k, ]
    va <- treino[parte == k, ]
    map_dfr(names(ajusta), function(nm) {
      ajuste <- ajusta[[nm]](tr)
      tibble(modelo = nm, acuracia = mean(preve[[nm]](ajuste, va) == va$sobreviveu))
    })
  })
})

resumo <- validacao |>
  group_by(modelo) |>
  summarise(acuracia_media = mean(acuracia), desvio = sd(acuracia)) |>
  arrange(desc(acuracia_media))
print(resumo)

# Referência: "todas as mulheres sobrevivem, todos os homens morrem"
cat("Regra só do sexo:",
    round(mean((treino$sexo == "feminino") == treino$sobreviveu), 4), "\n")

# 5. Modelo final e submissão ------------------------------------------------
melhor <- resumo$modelo[1]
cat("Modelo escolhido:", melhor, "\n")

modelo_final <- ajusta[[melhor]](treino)
if (melhor == "logistica") print(summary(modelo_final))

submissao <- tibble(
  PassengerId = teste$id_passageiro,
  Survived    = preve[[melhor]](modelo_final, teste)
)
write_csv(submissao, "submission.csv")
cat("submission.csv gravado:", nrow(submissao), "linhas,",
    sum(submissao$Survived), "sobreviventes previstos\n")
