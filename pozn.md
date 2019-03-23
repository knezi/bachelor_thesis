# bakalářka
/data/students/knizek

## clanky
printed

## TODO programovani
geenea entity -> s/ngrams/entities/
log - tp/fp/tn/fn....
zkusit featuru obshujici klasifikaci na test masinerie
should I care about overwriting files and such?
116 n/a sentiment
TODO konstanty misto hodnot??
brat jenom slova delsi nez neco??
kouoknout rucne na data
feature rozdelit na dve
* Information Gain - snizit dimenzionality
Najit soouslovi - mutual information; compound detection
lingv ficury
podle mnozstvi dat a jake ficury
embedding - gensym
zkusit s plaintextem
try to plot error rate of useful vs not useful and blah blah blah...

Drobnost v reportu:
- Dejte mezi jednotlive casti nejaky oddelovac (pomlcky nebo prazdnou radku)
- Zaokrouhlite P a R na 3 desetinna mista
- napiste co je P a co R


## PROGRESS of docstring:
\*.sh - done
filter.py - done
statistics.py - done
join.py - done
load_data.py - done
exceptions.py - done
extract_ids.py - done
get_data_for_ids.py - waiting - on geneea server
reduce_user.py - necessary??
reduce_bus.py - necessary??
my_unittests.py - zbalit hezky a dodelat
compare_langs.py - useless?
crop_geneea.py - todo
process_data.py


## při hotovém programu
zkontrolovat, jestli masinerie nekde nenechava tmp soubory
30 minut na spusteni vsechno se vsim vsudy
requirements.txt - python, setup.py

## psaní bakalářky
 - pridat kecy o poctu ruznych review a tak, proc co jsem jak vybiral
 	-> grafy poctu slov/error rate and stuff
 - taking only business_review_count > 50
 - don't include word error unles you check the language
 - taking only attributes with at least 10 and at least 50 reviews
 - measure if reducing non-restaurants helps (no, doesn't)
 - zmerit ruzne konfigurace (kosinova vzdalenost...)

## zdroje
 - najit bakalarky podobneho zamezeni
 - najit sentiment analysis and classification
 - find state of the art of language modelingz
 - docist gensim, ft



PCA, SVD
- redukce dimenze
- centralizovat - kazdy sloupec - prumer nula + odchylky
- singularni matice - moc featur - 




* n-gramy
 - pridat 1 slova, 3 slova

korelace featur


cross validation?
learning curves


negativni cosin


accuracy - train_set
 - po pridani jedno slov z 95 -> 65
 - zkusit udelat jen suuuper frekventovana slova


 sklearn - pca randomisedPCA
 truncatedSVD


 svm - scikit
 maximum entropy





zeptat:
- joint probability & distribution
	Jakej je rozdil oproti conditional?
	-> generative vs discriminative
	generative class based on joint prob - distribution that pair X,Y falls into; makes more assumption about data, infers result indirectly; models joint probability and then from Bayes rule - calculates p{y|x] and picks the most probable label y (usual through Bayesan theorem which requires to compute p[x|y])
vs discriminative based on conditional probability (or none at all) P[X|Y], needs better data, but infers result directly; directly models or directly learns map x to y
	-> logistic regr vyssi presnost + Andrew Ng
	
- n-gram jsou proste Markovovy řetezy -> aplikace ve word2vec


HANA
Po potvrzení se nově objeví možnost zadání práce vytisknout (resp. vygenerovat zadání v PDF). Vedoucí práce (případně tajemník katedry) formulář se zadáním 3x vytiskne, nechá ho podepsat vedoucím své katedry (v případě pedagogických studijních oborů vedoucím KDF resp. KDM) a fakultní poštou odešle na studijní oddělení.
Studijní oddělení nechá vyplněný formulář schválit a podepsat děkanem a tím je zadání práce ukončeno. Jeden výtisk formuláře dostanete zpět poštou (budete ho ještě potřebovat při dokončování práce); jeden exemplář zadání vrátí studijní oddělení na pracoviště vedoucího a jeden založí do Vašeho studijního spisu.


