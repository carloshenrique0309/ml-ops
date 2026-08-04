# ml-ops 

> **Status deste documento:** grade estrutural (objetivo, dor motivadora, o que se constrói, tarefa de casa e ponto de atrito por encontro). Os **textos teóricos** de cada módulo — baseados no ml-ops.org — serão gerados depois, módulo a módulo, com a fonte verificada na hora da escrita. Este é o mapa; o conteúdo redacional vem em cima dele.

---

## 1. Dados do curso

| Item | Definição |
|---|---|
| **Público** | Último semestre de graduação (4 anos) em IA / Ciência de Dados. Sabem ML, Python e treinar modelos. Têm buracos em engenharia de software (Git, Docker, estrutura de projeto). |
| **Carga** | 15 encontros × 4h = **60h síncronas**  |
| **Objetivo terminal** | Ao fim do curso, o aluno é capaz de desenvolver um projeto de ML **de ponta a ponta**, atravessando os três pipelines (Dados, ML, Código) e implementando as principais ferramentas open-source de gestão de cada etapa. |
| **Escopo** | ML clássico (tabular). LLMOps/Agentes são apenas **citados** como extensão. Kubeflow/Kubernetes ficam no material assíncrono. |

---

## 2. Princípios de design (ler antes de aplicar a grade)

1. **Dor → ferramenta.** Nenhuma ferramenta é introduzida antes de o aluno sentir a dor que ela resolve. Docker entra quando "na minha máquina funciona" vira problema real; MLflow entra quando eles perdem qual experimento foi o melhor; Airflow entra quando rodar as etapas na mão fica insuportável. A ferramenta é a resposta a uma frustração já vivida, nunca um tópico solto.

2. **Projeto fixo, deliberadamente chato.** Todos constroem o **mesmo** sistema (churn tabular). Dataset e modelo são **fixados pelo professor** — os alunos não escolhem. O ponto pedagógico nunca é a acurácia; é o encanamento ao redor do modelo. Tirar essa liberdade é intencional: evita que a turma gaste tempo tunando modelo em vez de aprender MLOps.

3. **Sala invertida parcial.** Casa = **preparação de ambiente** (antes do encontro) + **consolidação/extensão** (depois) + **leitura teórica**. Sala = **construção assistida e desbloqueio ao vivo**. Setup acontece fora da sala; a sala fica livre para o que só o professor presencial resolve. Sem isso, ~35% de cada encontro evapora em "esperem, vou ajudar o fulano com o Docker".

4. **Checkpoint de sanidade** nos primeiros ~15min de cada encontro: o aluno chega com o ambiente da tarefa de casa de pé, ou com um erro específico para ser resolvido rápido. Isso é o que torna viável uma turma de 30+ sem o pesadelo de setup coletivo.
---

## 3. Fontes e vocabulário

| Uso | Fonte primária |
|---|---|
| Teoria, princípios, ciclo de vida, os 3 pipelines | **ml-ops.org** (INNOQ) |
| Vocabulário oficial de fases | **AWS Well-Architected ML Lens** |
| Ferramentas hands-on | Docs oficiais de cada ferramenta (open-source) |
| "Como isso vira serviço gerenciado" | **AWS** (análogos SageMaker etc.) |

**Tabela de tradução de vocabulário** (apresentar no Encontro 1 para evitar confusão terminológica):

| ml-ops.org (teoria) | AWS Well-Architected (fases) | Ferramenta do curso (open-source) | Análogo gerenciado AWS |
|---|---|---|---|
| Data Pipeline | Data Processing | DVC + Great Expectations/pandera | SageMaker Processing / Feature Store |
| ML Pipeline | Model Development | MLflow (tracking + registry) | SageMaker Experiments / Model Registry |
| (Orquestração — implícita) | (transversal) | Airflow | SageMaker Pipelines / Step Functions |
| Software Code Pipeline | Model Deployment | GitHub Actions + FastAPI + Docker | CodePipeline / SageMaker Endpoints |
| Monitoring & Logging | Model Monitoring | conceito + demo (Evidently) | SageMaker Model Monitor / Clarify |

> **Nota de coerência:** o diagrama-mapa do curso é o dos **três pipelines do ml-ops.org**. A âncora conceitual é *o modelo de três pipelines*, não "é da AWS". A AWS entra como vocabulário de fase e como catálogo de análogos gerenciados — não invocar "é o material da AWS" como justificativa para o diagrama, porque ele não é.

---

## 4. Kit de ferramentas (hands-on, presencial)

Git · Docker · DVC · MLflow · Airflow · GitHub Actions · FastAPI

Deliberadamente **open-source**: o aluno aprende o conceito onde vê o encanamento, e depois reconhece o serviço gerenciado AWS como "a mesma coisa, terceirizada". Isso também elimina o pesadelo logístico de contas AWS para 30+ alunos.

---

## 5. Projeto-fio-condutor

**Previsão de churn** (classificação binária tabular). Dataset e baseline fixados pelo professor.

**Arco:** o aluno começa no Encontro 1 com um notebook vergonhoso que "funciona na minha máquina" e termina no Encontro 15 com:

`dados versionados (DVC)` → `experimentos rastreados (MLflow)` → `pipeline orquestrado (Airflow)` → `modelo servido via API (FastAPI + Docker)` → `CI/CD automatizado (GitHub Actions)` → `monitoramento de drift (conceitual)`.

Cada encontro adiciona **uma** peça, motivada por uma dor sentida no encontro anterior. O diagrama do ml-ops.org é mostrado progressivamente — cada pipeline colorido "acende" quando a turma chega naquele módulo.

---
