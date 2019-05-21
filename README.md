# NPRG045

## Závislosti:

aspell neni potreba, kdyz mate denormalizovana data

Já používám python 3.7.2 - vsechny skripty automaticky volaji /bin/env python3

Na sandboxu je potřeba použít 3.6, funguje toto:

`virtualenv v --python=python3.6`

`cd v`

`source ./bin/activate`

Naklonujte repozitar (nebo klonujte rovnou fantastic-spoon):

`git clone https://github.com/knezi/NPRG045`

`cd NPRG045`

Zkompilujte fasttext:

```
git clone https://github.com/facebookresearch/fastText.git
cd fastText
make
cd ..
```

pro nainstalovani zavislosti v pipu:

```
pip install `cat pip_deps`
```

Dale predpoklada nainstalovane geneea.analyzer a geneea.utils (utils se museji instalovat prvni).


## spousteni

Dale k koreni repa vytvorte:

`mkdir graphs`

Pro data staci nakopirovat slozku /data/students/knizek/data/ do korenu. Nakonec staci 500MB. To jsou rovnou denormalizovane pouzite soubory.

`./process_data.py experiments.yaml data/data.json data/geneea.json | tee dump`

Po dobehnuti je graf v `graphs/Psummary.png` a ta sama data v csv dumpu:
`graphs/Psummary.csv`

Na stdout mame (nebo v `dump`) precision a recall jednotlivych velikosti bayesa.
