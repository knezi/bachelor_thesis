# bakalářka
/data/students/knizek

## clanky


## odevzdani
say how awesome jupyter is
\it -jmena algoritmu, postupu a tak
\bb - definice
- nezalomitelne mezery
- pomlcky bracho
50 min
- pdfa
- vazba nemusi byt nerozebiratelna
- pres 800 MB napsat HOffmanove
- uprava viz MJ's pruvodce
- kontext dira cila
- abstract nejdulezitejsi
- cil - uvod cilem teto prace je....
uzivatelska doc??
- diskuse a experiment nejdulezitejsi
	- pseudo nahoda - generator ze seedem
	-> metoda
	-> vysledky  vizualizace
	-> diskuse
- replikvatelnost
- hypoteza -> interpretace
- prvni osoboa mnozneho cisla
- nepiste detektivky
- komu pises - kolega matfyzak
	-> obecne znalosti, ale ne z oboru
- precist nahlas
- testujte na lidech
- vnimejte, jak pisi jini
- ruzne pomlcky
- desetinna tecka!
- nedelitelne mezery??
- latex -pdfx
- vlozene obrazky jako pdf nejsou pdfa (pdfix), asi nebude potreba
- sis to kontroluje pomoci verapdf.org (online servis http://www.validatepdfa.com)
- mj/bc
- povolene formaty priloh
- citace [2] - jako kdyby to mohlo byt vynechano - jak pise Balcar [1], i Jirasek [2]
	- lze i prednasku

## psani
https://grammarsherpa.wordpress.com/2011/08/13/capital-letters-in-titles-headline-styling/
capitalization
- dokumentace dat
- geneea - memory overflow, sorting and stuff - reducing memory demands
- precist smernice o bakalarce

## TODO programovani
yaml config file:
experiments:
 -
   name: 'a'
   classificator: 'name'
   features:
     - f1_name
	 - f2_name
	 - ...
   preprocessing:
     - preprocesing_function/class? TODO
	 - ...
   evaluation:
     - eval1 class name
	 - eval2
	 - ...
graphs:
 - TODO
 -
   filename: 'a'
   datalines:
     - 
	   dataline: experiment_name.eval1
	   [fmt: '']
	 - 
	   dataline: experiment_name2.eval2
	   [fmt: '']
	 - ...
   label_x TODO??
   label_y
 

 - running define:
	algorithm
	algo parameters:
		- features - simply set of features
		- preprocess
		- run
 	eval function
 - what to plot?

Classes:
 - eval functions
 	- receives classify instance and by classify gets shit done
	Class Evaluate:
		def __init__():

		@virtual
		def evaluate(aa: classify, test_sets: feature_dict) -> {"nameEvalFunction": val}

 - classificator:
 	 - gets parameters from yaml (only preprocessing though???? is this necessary? - change/think)
	 - train(feature_dict)
	 - classify(instance)
	 	 - return classified class
	 - name (readonly)

- move feature_matrix conversion and dump fasttext to outside classes
	-> class Data will only have one get_data method
	-> where the conversion functions will be??

## TODODDO TADY jsme skoncil
pridej klasifikovanou tridu do configu yamlu??
dodelat FASTTEXT klasifikator
nacitani YAMLu

ztratili jsme naivebayes train set accuracy - pridej to do konfigu, chcem to ale vubec?

split generate sample and features to allow feature change
add set gram_words, used_entities to check whether data feature_dict get has been changed
split tokenizer usage in and outside Data Class??
move entities/index and stuff outside Data class?
have setters getters for word/entities/... inclusion
a slova dle entropie ne vyskyu
v make... pridej execution unittesty
format directive with function calling
word2vec for classif?
fasttext - parameters tuning
geenea entity -> s/ngrams/entities/
log - tp/fp/tn/fn....
should I care about overwriting files and such?
116 n/a sentiment
TODO konstanty misto hodnot??
brat jenom slova delsi nez neco??
kouoknout rucne na data
feature rozdelit na dve
* Information Gain - snizit dimenzionality - or mutual information?
Najit soouslovi - mutual information; compound detection
lingv ficury
podle mnozstvi dat a jake ficury
embedding - gensym
zkusit s plaintextem
try to plot error rate of useful vs not useful and blah blah blah...
todo Makefile and clean

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
crop_geneea.py - done
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

