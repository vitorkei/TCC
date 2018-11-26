# Estudo de caso de *Deep Q-Learning*

Repositório contendo os códigos e modelos referentes ao Trabalho de Formatura Supervisionado - 2018.

## ai/

Contém os códigos e modelos construídos ao longo dos últimos treinamentos feitos.

```atari/``` contém os códigos do *Pong* e do *Asteroids*.

```gridworld/``` contém os códigos do *Gridworld*.

```models/``` contém os modelos desenvolvidos nos últimos treinamentos. Há uma pasta para cada ambiente.

### Executando e treinando

Para poder rodar o *Pong*, é necessário ter o Gym e respectivos ambientes instalado: https://gym.openai.com/docs/

Para poder rodar o agente do *Asteroids*, é necessário ter o Gym-Retro e ambiente Atari intalados.
Além disso, é preciso importar a ROM:

```
python3 -m retro.import /path/to/ROMs/directory/
```

Basta rodar os comandos abaixo para executar os respectivos agentes.

```
python3 pong_agent.py

python3 asteroids_agent.py
```

Eles estão configurados para apenas atuar no ambiente segundo os modelos do diretório ```ai/models/```

Se quiser que ele treine, basta mudar a variável ```training``` na parte de hiper-parâmetros de cada agente.

Fazer isso sobrescreverá os modelos presentes no diretório ```ai/models/```

NOTA: O ambiente do *Gridworld* no Gym sofreu grandes alterações para os testes poderem ser feitos e, portanto, não está apto a ser executado.
Informações de como executá-lo serão disponibilizadas mais tarde.

### Renderizando

Por padrão, os ambientes não são renderizados.

Caso deseje ver o ambiente renderizado, basta alterar a variável ```render``` na parte de hiper-parâmetros de cada agente.

Também é possível acelerar ou retardar a renderização pela variável ```frame_gap```.
