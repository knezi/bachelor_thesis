# bakalářka
https://unix.stackexchange.com/questions/332641/how-to-install-python-3-6
/data/students/knizek

## styl psani
- odstavece???
 - consistent past vs rpesent
 	- past for experiments
	- present for general stuff
- like vs as
- remove connections you don't need as many of them
- don't give summaries without thinking
- use respectivelly carefully
https://tex.stackexchange.com/questions/34155/autoref-does-not-capitalize-initial-character-in-sentence-when-referencing-lae

## clanky
error analysis - add section
	http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.441.6742&rep=rep1&type=pdf

exploring data
 - visualisation
 Kumar data Mining

 - najit bakalarky podobneho zamezeni
 - najit sentiment analysis and classification
 - find state of the art of language modelingz

## odevzdani
\it -jmena algoritmu, postupu a tak
\bb - definice

- nezalomitelne mezery
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


## experiment
 - zminit prumerny pocet slov a pismen
 - baseline
 - prior class distribution
 - test fasttext with diff parmas
 - different feature selection
 - tranovaci cas v zavislosti na poctu parametru a tak - zlepseni stoji za cas?

1st round
 - baseline
 - mi

2nd round
 - ficury - naive bayes
	- ruzne sady
	  	- top n dle tfidf
		- top n dle mutual info n-grams
		- top n entit

3rd round - this is a big maybe
 - best performing
 	- chi^2 vs mi

round
 - algoritmy
  - fasttext
  - naive
  - decision tree
  - pca
	- svm???


last round
 - velikost dat


 Zacit psat ze jsme pustili baseline, ale f-measure neni definovan
 Nekolik round - mi, pak sady
 Chteli bychom n*n but infeasible
 learnign curves at the end (asi je i pridat do evaluace??)

## psani
### later
* add somewhere mention that we have linguistics and non-ling features - mainly we use geneea for analysis
* summary
* typography
* details, credentials, my data
* experiments conducted
* fix citations
* citace???

- zminit jupyter a nastroje pouzite
 - pridat kecy o poctu ruznych review a tak, proc co jsem jak vybiral
 	-> grafy poctu slov/error rate and stuff
 - don't include word error unles you check the language
 - measure if reducing non-restaurants helps (no, doesn't)
 - zmerit ruzne konfigurace (kosinova vzdalenost...)

### diagramy
https://grammarsherpa.wordpress.com/2011/08/13/capital-letters-in-titles-headline-styling/
capitalization
- dokumentace dat
- geneea - memory overflow, sorting and stuff - reducing memory demands
- precist smernice o bakalarce


## programatorska dokumentace
- pridej yaml - compute mutula info
			  - learnign curves - space size test
prdej nekam specifikaci vsech feature a preprocesoru a klasifikatoru a extra_data
last preproces musi vratit tuple (whatever, label) for testing data - training go * yourself
YAML speci:
if you have extra_data, you need to get rid of them in preprocessing - classifier gets first element of the tuple
	- for training also label
config:
chunks: 1
tasks [
	{name,
	classificator,
	features: [],
	preprocessing: [],
	extra_data: [],
	config: {/* this depends of the classificator and preprocessor */}
]
graphs : [
	{name,
	data:
	 - namespace
	 	- line
	 	- line
	 - namespace
	 	- line
	}
]

## programovani
* exper_file - learning curves
* prodej pocitani a agregovani statistik
* debian python3.6
* pridej do readme kam patri data

### later
* make - expects data in ../data/datset
* add switch to do graphs and statisics (on one random dataset)
* dokumentovani kodu
* Information Gain - snizit dimenzionality - or mutual information?
* restrict used entities or all?

### possible todos
* point - error check if not enough dat
* prozkoumat zero instances
* vytvor podslozky, at je to vice prehledne
* fasttext - check files exist
* TODOB learning curves
* replace types manually withi FeatureDict type

### progress of docstring:
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
my_unittests.py - done
compare_langs.py - useless?
crop_geneea.py - done
process_data.py

### při hotovém programu
zkontrolovat, jestli masinerie nekde nenechava tmp soubory
30 minut na spusteni vsechno se vsim vsudy
requirements.txt - python, setup.py


## stats
[knezi@groot dataset]$ cat review.json | sed 's/^.*text":"//' | sed 's/".*//' | wc
5261669 506159364 2769188913

526 chars, 96 words


## zeptat
- diagram obrazek - feature correlation - v caption dodat zdroj
- nezalomitelne mezery
- citace - dost?
- uzivatelska doc

shrnuti zacatek a konec kapitoly

architektura
 - intro
 - high level + obrazek a pospat krabicky
 - potom konfigurovatelnost a make a...
 - samostatna stranka - conceptual overview - papir - 

 * pis
 dokonci 14.7

 add to readme - mkdir graphs


- DNES
 - eps funguje?
 - apendixy furt nefungujou
 - app E - threshold vypada hnusne


 spellchecker
 tabulky
 pdfa 
 citace i s carkama


 todo - highlight best results in the tables
