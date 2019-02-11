# NPRG045

Nejdrive TODO yelp data

co data geenea?

TODO reqs
Závislosti:

aspell (en, de, fr - u mě balíky aspell, aspell-{en,de,fr} )

Já používám python 3.7.2 - vsechny skripty automaticky volaji /bin/env python3

pro nainstalovani zavislosti v pipu:

```
pip install `cat pip_deps`
```

Pote naklonujte gitovy repozitar.

Dale staci spustit v korenu repozitare:

`mkdir graphs`

Pro data staci nakopirovat slozku /data/students/knizek/data/ do korenu. Nakonec staci 500MB.


`./denormalise.sh cesta/ke_slozce_s_yelpem data/data.json`

(do data.json se uklada vysledek)

`./process_data.py | tee dump`

Po dobehnuti je graf v `graphs/Psummary.png` a ta sama data v csv dumpu:
`graphs/Psummary.csv`

Na stdout mame (nebo v `dump`) precision a recall jednotlivych velikosti bayesa.
