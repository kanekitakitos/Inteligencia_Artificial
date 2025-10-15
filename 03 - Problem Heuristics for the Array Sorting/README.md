# Array Sorting with Heuristic Search

Este projeto explora a resolução de um problema de ordenação de arrays utilizando algoritmos de busca em espaços de estados. O objetivo é encontrar a sequência de trocas (swaps) de custo mínimo para transformar um array inicial em um array objetivo.

O custo de cada troca depende da paridade dos números envolvidos:
- **Par-Par**: Coste 2
- **Impar-Impar**: Coste 20
- **Par-Impar**: Coste 11

## Algoritmos Implementados

O projeto implementa e compara dois algoritmos de busca fundamentais:

1.  **`GSolver` (Busca de Custo Uniforme - UCS):**
    - É um algoritmo de busca **não informada**.
    - Explora o espaço de estados expandindo sempre o nó com o menor custo acumulado (`g(n)`) desde o início.
    - Garante encontrar a solução de custo ótimo, mas pode ser extremamente lento em problemas complexos, já que explora muitas rotas desnecessárias por não ter um "senso de direção".

2.  **`AStarSearch` (Busca A*):**
    - É um algoritmo de busca **informada**, muito mais eficiente.
    - Utiliza una función de evaluación `f(n) = g(n) + h(n)`, donde:
        - `g(n)` é o custo real desde o estado inicial até o estado `n`.
        - `h(n)` é uma **heurística** que estima o custo mínimo desde `n` até o estado objetivo.
    - A chave de seu desempenho reside na qualidade da heurística `h(n)`.

---

## A Heurística de A*: O Coração da Eficiência

A superioridade do algoritmo A* neste projeto se deve a uma heurística (`h(n)`) muito sofisticada e precisa, implementada na classe interna `ArrayCfg.Heuristic`. Esta heurística proporciona uma estimativa muito ajustada do custo real restante, permitindo ao algoritmo podar galhos inteiros da árvore de busca e encontrar a solução ótima de forma incrivelmente rápida.

A lógica se baseia no conceito matemático da **decomposição de permutações em ciclos disjuntos**.

### 1. O que é a Descomposição em Ciclos?

O problema de ordenar o array pode ser visto como transformar uma permutação de números em outra. Qualquer permutação pode ser decomposta em um conjunto de "ciclos" independentes.

**Ejemplo:**
- Array Atual: `[2, 3, 1]`
- Array Objetivo: `[1, 2, 3]`

Aqui, o elemento `2` está na posição de `1`, `3` está na posição de `2`, e `1` está na posição de `3`. Isto forma um único **3-ciclo**: `1 -> 2 -> 3 -> 1`.

A ideia fundamental é que **os ciclos são subproblemas independentes**. O custo total para ordenar o array é a soma dos custos para resolver cada ciclo separadamente. Um ciclo de comprimento `k` sempre requer um mínimo de `k-1` trocas para ser resolvido.

### 2. Estratégia Híbrida para Calcular o Custo dos Ciclos

A heurística não se conforma com uma estimativa simples. Utiliza uma **estratégia híbrida** para calcular o custo de cada ciclo com a máxima precisão possível, dependendo de seu tamanho:

#### a) 2-Ciclos (Trocas Simples)
- **Lógica:** Um ciclo de comprimento 2 (ex: `A` na posição de `B` e `B` na de `A`) se resolve com uma única troca.
- **Cálculo:** A heurística calcula o **custo exato** dessa única troca (`calculateCost(A, B)`). É a estimativa mais precisa possível.

#### b) Ciclos Pequenos (3 e 4 elementos)
- **Lógica:** Para ciclos de tamanho `k` de 3 ou 4, a heurística realiza uma **busca por força bruta** para encontrar o **custo ótimo real** para resolver esse subproblema.
- **Cálculo:** Explora todas as sequências válidas de `k-1` trocas entre os elementos do ciclo e seleciona a de menor custo. Embora seja computacionalmente intensivo, para um `k` tão pequeno o custo é trivial, mas o ganho em precisão para a heurística é enorme.

#### c) Ciclos Grandes (> 4 elementos)
- **Lógica:** Para ciclos maiores, a força bruta seria muito lenta. Em seu lugar, utiliza-se um **algoritmo guloso (greedy)** rápido e admissível, que varia segundo a composição do ciclo.
- **Cálculo:**
    - **Se o ciclo contém números pares:** A estratégia mais barata é usar um número par como "pivô". O custo é a soma de `(número de pares - 1)` trocas par-par (custo 2) e `(número de ímpares)` trocas par-ímpar (custo 11).
    - **Se o ciclo contém apenas números ímpares:** Duas opções são consideradas:
        1. Resolver o ciclo internamente com `k-1` trocas ímpar-ímpar (custo 20 cada).
        2. "Pegar emprestado" um número par de fora do ciclo, realizar `k` trocas par-ímpar (custo 11 cada) e depois devolver o número par. A heurística usa o **mínimo** entre `(k-1)*20` e `k*11`. Se não houver números pares no array, apenas a primeira opção é possível.

### 3. Admissibilidade: A Garantia de Otimalidade

A heurística é **admissível**, o que significa que **nunca superestima o custo real** para chegar ao objetivo. Isto é crucial, já que é a condição que garante que A* encontrará a solução ótima.

A admissibilidade se mantém porque:
- Para 2-ciclos, usa o custo **exato**.
- Para ciclos pequenos, encontra o custo **ótimo**.
- Para ciclos grandes, usa uma estimativa gulosa que representa o **melhor caso possível** (lower-bound).

Graças a esta combinação de precisão e eficiência, o algoritmo A* é capaz de resolver problemas muito complexos em milissegundos, enquanto um algoritmo não informado como `GSolver` levaria minutos, horas ou até mais.

## Como Executar o Projeto

1.  **Classe Principal:** O ponto de entrada é `Main.java`. Por padrão, ele executa a busca A*.
2.  **Entrada:** O programa espera duas linhas da entrada padrão:
    - A primeira linha é o array inicial (números separados por espaços).
    - A segunda linha é o array objetivo.

    **Ejemplo de entrada:**
    ```
    2 4 6 8 10 12 1 3 5 7 9 11
    1 3 5 7 9 11 2 4 6 8 10 12
    ```
3.  **Saída:** O programa imprimirá um único número: o custo total mínimo da solução.

## Estrutura do Projeto

- **`src/core`**: Contém as classes principais do motor de busca.
  - `AbstractSearch.java`: Classe base abstrata para os algoritmos de busca.
  - `AStarSearch.java`: Implementação do A*.
  - `GSolver.java`: Implementação da Busca de Custo Uniforme.
  - `ArrayCfg.java`: Representação do estado do problema e a lógica da heurística.
  - `Ilayout.java`: Interface que define a estrutura de um estado.
- **`src/test`**: Contém os testes unitários e de desempenho.
  - `AStarSearchTest.java`: Testes para o algoritmo A*.
  - `GSolverTest.java`: Testes para o algoritmo `GSolver`.
  - `ComparisonTest.java`: Testes de benchmark que comparam a velocidade de ambos os algoritmos em casos complexos.