# pyconfirm
Pyconfirm uses an ensemble of a random forest and LSTM in order to confirm whether or not a given time-series from the  Cameras for All-Sky Meteor Surveillance (CAMS) survey is an actual meteor. This work was done as part of the NASA Frontier Development Lab 2017 in order to facilitate the detection of potentially hazardous, long-period comets.

# Usage
Pyconfirm is built of a modular codebase that does the work under the hood, as well as a simple script that can be run to use the models on actual CAMS data. 

## Getting Started

Pyconfirm is built of a modular codebase that does the work under the hood, as well as a simple script that can be run to use the models on actual CAMS data. These instructions will get you a copy of pyconfirm up and running on your local machine for development and testing purposes. 

### Prerequisites

```
numpy
scipy
statsmodels
scikit-learn
keras
pandas
```

### Installing

Simply install the prerequisites and clone the repository to your local machine, and you're good to go! Note: Currently the LSTM model won't fit on this repository. Feel free to contact me for the link to download that.

## Running the tests

You can test pyconfirm on your local machine by running meteor_clean on the example data file provided here:
Explain how to run the automated tests for this system

```
python meteor_clean.py
```

This should produce the file exFTP_detectinfo_cleaned.txt with only the meteors.

## Authors

* **Antonio Ordonez**

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Big shout out to Susana Zoghbi, Andres Plata-Stapper, and Marcelo De Cicco for the help on this project throughout our FDL journey.
* Also a big thanks to the NASA FDL team for making this possible
* Peter Jenniskens and Pete Gural also deserve significant mention for mentoring us on this difficult problem
