# NPRG045

## Závislosti:

aspell neni potreba, kdyz mate denormalizovana data

Já používám python 3.7.2 - vsechny skripty automaticky volaji /bin/env python3

Na sandboxu je potřeba použít 3.6, funguje toto:

`virtualenv v --python=python3.6`

`source ./v/bin/activate`

Pote naklonujte tento gitovy repozitar a v korenu zkompilujte fasttext:

```
git clone https://github.com/facebookresearch/fastText.git
cd fastText
make
```

pro nainstalovani zavislosti v pipu:

```
pip install `cat pip_deps`
```

Dale predpoklada nainstalovane geneea.analyzer a geneea.utils.


## spousteni

Dale staci spustit v korenu repozitare:

`mkdir graphs`


Pro data staci nakopirovat slozku /data/students/knizek/data/ do korenu. Nakonec staci 500MB. To jsou rovnou denormalizovane pouzite soubory.

`./process_data.py | tee dump`

Po dobehnuti je graf v `graphs/Psummary.png` a ta sama data v csv dumpu:
`graphs/Psummary.csv`

Na stdout mame (nebo v `dump`) precision a recall jednotlivych velikosti bayesa.
