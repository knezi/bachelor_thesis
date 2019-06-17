# bakalářka
LOOP - fasttext

/data/students/knizek

## styl psani
 - we believe -> I believe
 - consistent past vs rpesent
 	- past for experiments
	- present for general stuff
- like vs as
- remove connections you don't need as many of them
- don't give summaries without thinking
- use respectivelly carefully
https://tex.stackexchange.com/questions/34155/autoref-does-not-capitalize-initial-character-in-sentence-when-referencing-lae
## clanky
tohle vypadata vazne dobre:
	Short text classification in twitter to improve information filtering
	short tet classification: A Survey Song

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

## experiment
	 - baseline
	 - prior class distribution

## psani


### later
* znimint knihovny?
* fasttext
* classificaiton
* classifiers
* summary
* my arch
* diagrams
* typography
* details, credentials, my data
* experiments conducted
* appendix - data
* freature engineering
* fix citations

- zminit jupyter a nastroje pouzite
 - pridat kecy o poctu ruznych review a tak, proc co jsem jak vybiral
 	-> grafy poctu slov/error rate and stuff
 - taking only business_review_count > 50
 - don't include word error unles you check the language
 - taking only attributes with at least 10 and at least 50 reviews
 - measure if reducing non-restaurants helps (no, doesn't)
 - zmerit ruzne konfigurace (kosinova vzdalenost...)
### diagramy
 - tikz - nefunguje?? - zeptat
https://grammarsherpa.wordpress.com/2011/08/13/capital-letters-in-titles-headline-styling/
capitalization
- dokumentace dat
- geneea - memory overflow, sorting and stuff - reducing memory demands
- precist smernice o bakalarce


## programatorska dokumentace
prdej nekam specifikaci vsech feature a preprocesoru a klasifikatoru a extra_data
last preproces musi vratit tuple (whatever, label) for testing data - training go fuck yourself
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
* make nefunguje
* data jsou pripravena jako data.json a ids
* zkontroluj process a jestli to matchuje
* prodej pocitani a agregovani statistik
* predelej geneea public API

### later
* make - expects data in ../data/datset
* pridej napojeni geeny
* README finish installation process
* pridej klasifikovanou tridu do configu yamlu??
* add switch to do graphs and statisics (on one random dataset)
* dokumentovani kodu
* Information Gain - snizit dimenzionality - or mutual information?
* restrict used entities or all?

### possible todos
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


