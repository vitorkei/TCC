# Estudo de caso de *Deep Q-Learning*

Repositório contendo os códigos, modelos e ferramentas auxiliares referentes ao Trabalho de Formatura Supervisionado - 2018.

## ai/

Contém os códigos e modelos construídos ao longo dos últimos treinamentos feitos.

atari/ contém os códigos do *Pong* e do *Asteroids*.

gridworld/ contém os códigos do *Gridworld*.

models/ contém os modelos desenvolvidos nos últimos treinamentos. Há uma pasta para cada ambiente.

### Executando e treinando

Para poder rodar o agente do *Asteroids*, é necessário antes executar o comando:

```
python3 -m retro.import /path/to/ROMs/directory/
```

Basta rodar os comandos abaixo para executar os respectivos agentes.

```
python3 gridworld_agent.py

python3 pong_agent.py

python3 asteroids_agent.py
```

Eles estão configurados para apenas atuar no ambiente segundo os modelos do diretório ai/models/

Se quiser que ele treine, basta mudar a variável ```training``` na parte de hiper-parâmetros de cada agente.

Fazer isso sobrescreverá os modelos presentes no diretório ai/models/

### Renderizando

Por padrão, os ambientes não são renderizados.

Caso deseje ver o ambiente renderizado, basta alterar a variável ```render``` na parte de hiper-parâmetros de cada agente.

Também é possível acelerar ou retardar a renderização pela variável ```frame_gap```.
